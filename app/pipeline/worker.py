"""Single-flight async worker for video processing."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable
from uuid import UUID

import cv2
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.detection import Detection, Frame, Plate, Session
from app.pipeline.detector import PlateDetector
from app.pipeline.frame_extractor import extract_frames, get_video_info
from app.pipeline.ocr import PlateReader
from app.pipeline.plate_validator import normalize_plate
from app.services.plate_service import upsert_plate
from app.services.ticket_lookup import lookup_ticket

logger = logging.getLogger(__name__)
UTC = timezone.utc

PROCESSING_SEMAPHORE = asyncio.Semaphore(1)

DbFactory = Callable[[], "AsyncSession | Awaitable[AsyncSession]"]


async def _open_db(db_factory: DbFactory) -> AsyncSession:
    db: Any = db_factory()
    if asyncio.iscoroutine(db):
        db = await db
    return db


async def _mark_failed(
    session_id: UUID, message: str, db_factory: DbFactory
) -> None:
    """Open a fresh DB session and mark the row failed. Truncates message."""
    db = await _open_db(db_factory)
    try:
        session = await db.get(Session, session_id)
        if session is not None:
            session.status = "failed"
            session.error_message = message[:500]
            session.ended_at = datetime.now(UTC)
            await db.commit()
    finally:
        await db.close()


async def process_video(
    session_id: UUID,
    video_path: str,
    fps_target: int,
    db_factory: DbFactory,
) -> None:
    """Process a video end-to-end. Only one video may run at a time."""
    if PROCESSING_SEMAPHORE.locked():
        logger.warning(
            "Rejecting session %s — another video is already processing", session_id
        )
        await _mark_failed(
            session_id,
            "Another video is currently processing. Please wait.",
            db_factory,
        )
        return

    async with PROCESSING_SEMAPHORE:
        db = await _open_db(db_factory)
        try:
            session = await db.get(Session, session_id)
            if session is None:
                raise ValueError(f"Session {session_id} not found")

            # (1) Mark running
            session.status = "running"
            started_at = datetime.now(UTC)
            if hasattr(session, "started_at"):
                session.started_at = started_at
            session.fps_target = float(fps_target)

            # (2) Video metadata
            info = get_video_info(video_path)
            session.total_frames = int(info.get("total_frames") or 0)
            await db.commit()

            # (3) Initialize models once
            detector = PlateDetector(settings.MODEL_PATH, settings.DETECTION_CONF)
            reader = PlateReader(gpu=False)

            crops_dir = Path(settings.CROPS_DIR)
            crops_dir.mkdir(parents=True, exist_ok=True)

            frames_processed = 0
            detections_since_commit = 0
            detections_total = 0

            # (4) Frame loop — any failure here marks the session as failed
            try:
                for frame_index, pts_ms, frame_img in extract_frames(
                    video_path, fps_target
                ):
                    frame_row = Frame(
                        session_id=session_id,
                        frame_index=frame_index,
                        pts_ms=pts_ms,
                    )
                    db.add(frame_row)
                    await db.flush()

                    for det in detector.detect(frame_img, frame_index=frame_index):
                        bbox = det["bbox"]
                        bx, by, bw, bh = bbox
                        confidence = float(det["confidence"])

                        try:
                            crop = PlateDetector.crop_plate(frame_img, bbox)
                        except ValueError:
                            continue

                        raw_text, conf_ocr = reader.read_plate(crop)
                        normalized_text: str | None = None
                        is_valid = False
                        if raw_text:
                            normalized_text, is_valid = normalize_plate(raw_text)

                        crop_filename = (
                            f"{session_id}_{frame_index}_{bx}_{by}.jpg"
                        )
                        cv2.imwrite(str(crops_dir / crop_filename), crop)

                        detection = Detection(
                            frame_id=frame_row.id,
                            bbox_x=bx,
                            bbox_y=by,
                            bbox_w=bw,
                            bbox_h=bh,
                            detection_confidence=confidence,
                            crop_image_path=crop_filename,
                            ocr_raw_text=raw_text or None,
                            ocr_confidence=conf_ocr,
                            ocr_normalized_text=normalized_text,
                            is_valid_ro_plate=is_valid,
                        )
                        db.add(detection)
                        await db.flush()

                        if normalized_text and is_valid:
                            plate = await upsert_plate(
                                normalized_text=normalized_text,
                                county_code=None,
                                county_name=None,
                                plate_type=None,
                                db=db,
                            )
                            detection.plate_id = plate.id

                            status, expires_at, _tid, _sid = await lookup_ticket(
                                plate_text=normalized_text,
                                zone_id=None,
                                checked_at=datetime.now(UTC),
                                db=db,
                                detection_id=detection.id,
                            )
                            detection.ticket_status = status
                            detection.ticket_expires_at = expires_at
                            await db.flush()

                        detections_since_commit += 1
                        detections_total += 1
                        if detections_since_commit >= 10:
                            await db.commit()
                            detections_since_commit = 0

                    frames_processed += 1

                    # (5) Periodic progress update
                    if frames_processed % 10 == 0:
                        session.frames_processed = frames_processed
                        await db.commit()
            except Exception as e:
                logger.exception(
                    "Frame loop failed for session %s", session_id
                )
                try:
                    await db.rollback()
                finally:
                    await db.close()
                await _mark_failed(session_id, str(e), db_factory)
                return

            session.frames_processed = frames_processed
            session.status = "done"
            session.ended_at = datetime.now(UTC)
            await db.commit()

            logger.info(
                "Session %s complete: %d frames, %d detections",
                session_id,
                frames_processed,
                detections_total,
            )
        except Exception as e:
            logger.exception("Processing setup failed for session %s", session_id)
            try:
                await db.rollback()
            finally:
                await db.close()
            await _mark_failed(session_id, str(e), db_factory)
            return
        await db.close()
