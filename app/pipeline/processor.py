"""
app/pipeline/processor.py

End-to-end session processor: frame extraction -> detection -> plate
record -> ticket lookup (audit).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.detection import Detection, Frame, Plate, Session
from app.pipeline.detector import PlateDetector
from app.pipeline.frame_extractor import extract_frames
from app.services.ticket_lookup import lookup_ticket

logger = logging.getLogger(__name__)

UTC = timezone.utc


class Recognizer(Protocol):
    """Anything that turns a crop image into a plate string (or None)."""

    def __call__(self, crop: np.ndarray) -> str | None: ...


def _noop_ocr(crop: np.ndarray) -> str | None:
    """Placeholder recognizer — always returns None."""
    return None


async def _get_or_create_plate(
    normalized_text: str,
    db: AsyncSession,
) -> Plate:
    """Return an existing Plate row or insert a new one."""
    result = await db.execute(
        select(Plate).where(Plate.normalized_text == normalized_text).limit(1)
    )
    plate = result.scalars().first()
    if plate is not None:
        plate.seen_count += 1
        plate.last_seen_at = datetime.now(UTC)
        await db.flush()
        return plate

    plate = Plate(normalized_text=normalized_text)
    db.add(plate)
    await db.flush()
    return plate


async def process_session(
    session_id: uuid.UUID,
    db: AsyncSession,
    *,
    ocr: Recognizer | None = None,
    zone_id: int | None = None,
) -> None:
    """Process every frame in a session's video file.

    Steps for each frame:
    1. Run YOLO detector to find licence-plate bounding boxes.
    2. Crop each detection and (optionally) run OCR.
    3. Persist ``Frame``, ``Detection`` and ``Plate`` rows.
    4. Call ``lookup_ticket`` with the ``detection_id`` so a
       ``TicketCheck`` audit row is written.
    5. Update the session status when done.

    Parameters
    ----------
    session_id:
        Primary key of an existing ``sessions`` row whose video has
        already been uploaded.
    db:
        An active ``AsyncSession``.  The function commits on success.
    ocr:
        Optional callable ``(crop: ndarray) -> str | None``.  When
        *None*, the built-in no-op stub is used (no text recognised).
    zone_id:
        Parking zone used for ticket validation.  ``None`` means
        "zone-agnostic" — tickets/subscriptions with *any* zone match.
    """
    recognize = ocr or _noop_ocr

    session = await db.get(Session, session_id)
    if session is None:
        raise ValueError(f"Session {session_id} not found")

    video_path = Path(settings.UPLOAD_DIR) / f"{session_id}_{session.source_filename}"
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    detector = PlateDetector(settings.MODEL_PATH, settings.DETECTION_CONF)

    session.status = "processing"
    await db.flush()

    frames_processed = 0

    for frame_index, pts_ms, frame_img in extract_frames(
        str(video_path), settings.FPS_TARGET
    ):
        # -- Frame record ---------------------------------------------------
        frame = Frame(session_id=session_id, frame_index=frame_index, pts_ms=pts_ms)
        db.add(frame)
        await db.flush()

        # -- Detection ------------------------------------------------------
        raw_detections = detector.detect(frame_img)

        for det in raw_detections:
            bbox_x, bbox_y, bbox_w, bbox_h = det["bbox"]
            confidence = det["confidence"]

            # Crop
            crop = frame_img[bbox_y : bbox_y + bbox_h, bbox_x : bbox_x + bbox_w]
            crop_filename = f"{frame.id}_{bbox_x}_{bbox_y}.jpg"
            crop_path = Path(settings.CROPS_DIR) / crop_filename
            cv2.imwrite(str(crop_path), crop)

            # OCR
            plate_text = recognize(crop)

            detection = Detection(
                frame_id=frame.id,
                bbox_x=bbox_x,
                bbox_y=bbox_y,
                bbox_w=bbox_w,
                bbox_h=bbox_h,
                detection_confidence=confidence,
                crop_image_path=crop_filename,
                ocr_raw_text=plate_text,
                ocr_normalized_text=plate_text,
            )
            db.add(detection)
            await db.flush()

            # Plate + ticket lookup (writes TicketCheck audit row)
            if plate_text:
                plate = await _get_or_create_plate(plate_text, db)
                detection.plate_id = plate.id

                status, expires_at, _tid, _sid = await lookup_ticket(
                    plate_text=plate_text,
                    zone_id=zone_id,
                    checked_at=datetime.now(UTC),
                    db=db,
                    detection_id=detection.id,
                )
                detection.ticket_status = status
                detection.ticket_expires_at = expires_at
                await db.flush()

        frames_processed += 1

    session.frames_processed = frames_processed
    session.status = "completed"
    session.ended_at = datetime.now(UTC)
    await db.commit()

    logger.info(
        "Session %s completed — %d frames processed", session_id, frames_processed
    )
