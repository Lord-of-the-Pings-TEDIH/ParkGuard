"""
app/pipeline/processor.py

End-to-end session processor with robust OCR:
frame extraction -> detection -> deskew/blur handling -> multi-angle OCR ->
temporal voting -> record -> ticket lookup (audit).
"""

from __future__ import annotations

import logging
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, TypeAlias

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.detection import Detection, Frame, Plate, Session
from app.pipeline.deskew import deskew_plate
from app.pipeline.detector import PlateDetector
from app.pipeline.frame_extractor import extract_frames
from app.pipeline.plate_validator import normalize_plate
from app.services.ticket_lookup import lookup_ticket

logger = logging.getLogger(__name__)

UTC = timezone.utc


@dataclass
class OCRCandidate:
    raw_text: str = ""
    confidence: float = 0.0
    normalized: str = ""
    is_valid: bool = False
    angle: float = 0.0


@dataclass
class PlateTrack:
    track_id: int
    bbox: tuple[int, int, int, int]
    last_frame: int
    observations: int = 0
    votes: Counter[str] = field(default_factory=Counter)
    conf_sums: dict[str, float] = field(default_factory=dict)


RecognizerOutput: TypeAlias = str | tuple[str | None, float] | None


class Recognizer(Protocol):
    """Anything that turns a crop image into plate text (+ optional confidence)."""

    def __call__(self, crop: np.ndarray) -> RecognizerOutput: ...


def _noop_ocr(crop: np.ndarray) -> RecognizerOutput:
    """Placeholder recognizer — always returns no text."""
    _ = crop
    return None


def _parse_ocr_angles(value: str) -> tuple[float, ...]:
    angles: list[float] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        angle = float(token)
        if angle < -45.0 or angle > 45.0:
            raise ValueError("OCR angles must be in [-45, 45] degrees.")
        if all(abs(existing - angle) > 1e-9 for existing in angles):
            angles.append(angle)

    if not angles:
        angles = [0.0]
    if all(abs(angle) > 1e-9 for angle in angles):
        angles.append(0.0)
    return tuple(angles)


def _read_with_recognizer(recognize: Recognizer, crop: np.ndarray) -> tuple[str, float]:
    raw = recognize(crop)
    if raw is None:
        return "", 0.0

    if isinstance(raw, tuple):
        if len(raw) < 2:
            return "", 0.0
        text = raw[0] if isinstance(raw[0], str) else ""
        confidence = float(raw[1]) if isinstance(raw[1], (int, float)) else 0.0
        return text.strip().upper(), confidence

    if isinstance(raw, str):
        return raw.strip().upper(), 1.0

    return "", 0.0


def _rotate_bound(image: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 1e-9:
        return image

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    bound_w = int((height * sin) + (width * cos))
    bound_h = int((height * cos) + (width * sin))
    matrix[0, 2] += bound_w / 2.0 - center[0]
    matrix[1, 2] += bound_h / 2.0 - center[1]

    return cv2.warpAffine(
        image,
        matrix,
        (bound_w, bound_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _sharpness_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _enhance_motion_blur(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    denoised = cv2.bilateralFilter(gray, 7, 50, 50)
    blurred = cv2.GaussianBlur(denoised, (0, 0), 1.0)
    sharpened = cv2.addWeighted(denoised, 1.6, blurred, -0.6, 0)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def _best_ocr_candidate(
    recognize: Recognizer,
    crop: np.ndarray,
    *,
    ocr_angles: tuple[float, ...],
) -> OCRCandidate:
    best_valid = OCRCandidate()
    best_any = OCRCandidate()

    for angle in ocr_angles:
        candidate_image = _rotate_bound(crop, angle)
        raw_text, confidence = _read_with_recognizer(recognize, candidate_image)
        if raw_text:
            normalized, is_valid = normalize_plate(raw_text)
            candidate = OCRCandidate(
                raw_text=raw_text,
                confidence=confidence,
                normalized=normalized,
                is_valid=is_valid,
                angle=angle,
            )
        else:
            candidate = OCRCandidate(angle=angle)

        if candidate.raw_text and candidate.confidence > best_any.confidence:
            best_any = candidate
        if candidate.is_valid and candidate.confidence > best_valid.confidence:
            best_valid = candidate

    return best_valid if best_valid.raw_text else best_any


def _bbox_iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    a_x2 = ax + aw
    a_y2 = ay + ah
    b_x2 = bx + bw
    b_y2 = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = aw * ah
    area_b = bw * bh
    union = max(1, area_a + area_b - inter_area)
    return float(inter_area / union)


def _prune_stale_tracks(
    active_tracks: dict[int, PlateTrack], *, frame_index: int, max_age: int
) -> None:
    stale_ids = [
        track_id
        for track_id, track in active_tracks.items()
        if frame_index - track.last_frame > max_age
    ]
    for track_id in stale_ids:
        del active_tracks[track_id]


def _assign_track(
    *,
    bbox: tuple[int, int, int, int],
    frame_index: int,
    active_tracks: dict[int, PlateTrack],
    all_tracks: dict[int, PlateTrack],
    next_track_id: int,
    max_age: int,
    min_iou: float,
) -> tuple[PlateTrack, int]:
    _prune_stale_tracks(active_tracks, frame_index=frame_index, max_age=max_age)

    best_track: PlateTrack | None = None
    best_iou = 0.0
    for track in active_tracks.values():
        iou = _bbox_iou(bbox, track.bbox)
        if iou > best_iou:
            best_iou = iou
            best_track = track

    if best_track is None or best_iou < min_iou:
        track = PlateTrack(
            track_id=next_track_id,
            bbox=bbox,
            last_frame=frame_index,
            observations=1,
        )
        active_tracks[next_track_id] = track
        all_tracks[next_track_id] = track
        return track, next_track_id + 1

    best_track.bbox = bbox
    best_track.last_frame = frame_index
    best_track.observations += 1
    return best_track, next_track_id


def _register_track_vote(track: PlateTrack, plate: str, confidence: float) -> None:
    track.votes[plate] += 1
    track.conf_sums[plate] = track.conf_sums.get(plate, 0.0) + confidence


def _track_best_plate(track: PlateTrack) -> tuple[str | None, int, float, float]:
    if not track.votes:
        return None, 0, 0.0, 0.0

    def _score(item: tuple[str, int]) -> tuple[int, float, str]:
        plate, votes = item
        avg_conf = track.conf_sums.get(plate, 0.0) / max(1, votes)
        return votes, avg_conf, plate

    best_plate, best_votes = max(track.votes.items(), key=_score)
    best_avg_conf = track.conf_sums.get(best_plate, 0.0) / max(1, best_votes)
    stability = best_votes / max(1, track.observations)
    return best_plate, best_votes, best_avg_conf, stability


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
    """Process every frame in a session's video file with robust OCR/voting."""
    recognize = ocr or _noop_ocr

    session = await db.get(Session, session_id)
    if session is None:
        raise ValueError(f"Session {session_id} not found")

    video_path = Path(settings.UPLOAD_DIR) / f"{session_id}_{session.source_filename}"
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    detector = PlateDetector(settings.MODEL_PATH, settings.DETECTION_CONF)

    min_ocr_conf = max(0.0, min(1.0, float(settings.OCR_MIN_CONF)))
    ocr_angles = _parse_ocr_angles(settings.OCR_ANGLES)
    min_sharpness = max(0.0, float(settings.OCR_MIN_SHARPNESS))
    track_max_age = max(1, int(settings.TRACK_MAX_AGE))
    track_min_iou = max(0.01, min(0.99, float(settings.TRACK_MIN_IOU)))
    min_track_votes = max(1, int(settings.MIN_TRACK_VOTES))

    session.status = "processing"
    session.fps_target = float(settings.FPS_TARGET)
    await db.flush()

    frames_processed = 0
    active_tracks: dict[int, PlateTrack] = {}
    all_tracks: dict[int, PlateTrack] = {}
    next_track_id = 1

    for frame_index, pts_ms, frame_img in extract_frames(
        str(video_path), settings.FPS_TARGET
    ):
        frame = Frame(session_id=session_id, frame_index=frame_index, pts_ms=pts_ms)
        db.add(frame)
        await db.flush()

        detections = detector.detect(frame_img, frame_index=frame_index)
        for det in detections:
            bbox_x, bbox_y, bbox_w, bbox_h = (
                int(det["bbox"][0]),
                int(det["bbox"][1]),
                int(det["bbox"][2]),
                int(det["bbox"][3]),
            )
            confidence = float(det["confidence"])
            bbox = (bbox_x, bbox_y, bbox_w, bbox_h)

            track, next_track_id = _assign_track(
                bbox=bbox,
                frame_index=frame_index,
                active_tracks=active_tracks,
                all_tracks=all_tracks,
                next_track_id=next_track_id,
                max_age=track_max_age,
                min_iou=track_min_iou,
            )

            try:
                crop = PlateDetector.crop_plate(frame_img, bbox, padding=0)
            except ValueError:
                continue

            crop_filename = f"{frame.id}_{bbox_x}_{bbox_y}.jpg"
            crop_path = Path(settings.CROPS_DIR) / crop_filename
            cv2.imwrite(str(crop_path), crop)

            candidate = OCRCandidate()
            rectified = deskew_plate(crop)
            if rectified.size == 0:
                rectified = crop

            sharpness = _sharpness_score(rectified)
            if sharpness >= min_sharpness:
                ocr_input = (
                    _enhance_motion_blur(rectified)
                    if sharpness < min_sharpness * 1.35
                    else rectified
                )
                candidate = _best_ocr_candidate(
                    recognize,
                    ocr_input,
                    ocr_angles=ocr_angles,
                )

                if (
                    candidate.raw_text
                    and candidate.confidence >= min_ocr_conf
                    and candidate.is_valid
                ):
                    _register_track_vote(track, candidate.normalized, candidate.confidence)

            best_plate, best_votes, _best_avg_conf, _stability = _track_best_plate(track)

            raw_plate_text = candidate.raw_text or None
            normalized_plate_text = candidate.normalized if candidate.raw_text else None
            is_valid_ro_plate = (
                bool(candidate.raw_text)
                and candidate.is_valid
                and candidate.confidence >= min_ocr_conf
            )

            if best_plate and best_votes >= min_track_votes:
                normalized_plate_text = best_plate
                is_valid_ro_plate = True

            detection = Detection(
                frame_id=frame.id,
                bbox_x=bbox_x,
                bbox_y=bbox_y,
                bbox_w=bbox_w,
                bbox_h=bbox_h,
                detection_confidence=confidence,
                crop_image_path=crop_filename,
                ocr_raw_text=raw_plate_text,
                ocr_normalized_text=normalized_plate_text,
                is_valid_ro_plate=is_valid_ro_plate,
            )
            db.add(detection)
            await db.flush()

            if normalized_plate_text and is_valid_ro_plate:
                plate = await _get_or_create_plate(normalized_plate_text, db)
                detection.plate_id = plate.id

                status, expires_at, _tid, _sid = await lookup_ticket(
                    plate_text=normalized_plate_text,
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

    finalized_tracks = sum(
        1 for track in all_tracks.values() if _track_best_plate(track)[1] >= min_track_votes
    )
    logger.info(
        "Session %s completed — %d frames processed, %d stable tracks",
        session_id,
        frames_processed,
        finalized_tracks,
    )
