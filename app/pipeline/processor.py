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
from app.pipeline.frame_extractor import extract_frames, get_video_info
from app.pipeline.geolocation import (
    CameraIntrinsics,
    PlateGeolocationCalculator,
    VehiclePose,
)
from app.pipeline.plate_color import PlateType, classify_plate_color
from app.pipeline.plate_validator import compact_plate, is_invalid_plate, normalize_plate
from app.services.parking_validation import validate_parking_at_location
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
    best_conf_plate: str | None = None
    best_confidence: float = 0.0
    best_conf_detection_id: uuid.UUID | None = None
    best_conf_bbox: tuple[int, int, int, int] | None = None
    last_detection_id: uuid.UUID | None = None
    detection_ids: list[uuid.UUID] = field(default_factory=list)
    finalized: bool = False


@dataclass(frozen=True)
class MobileLPRContext:
    """Per-session geolocation context.

    Bundling the calculator + pose + radius lets us pass a single optional
    object to ``process_session`` rather than half a dozen scalars.  When
    ``None`` the geolocation step is skipped entirely.
    """

    calculator: PlateGeolocationCalculator
    pose: VehiclePose
    search_radius_m: float


def build_mobile_lpr_context(
    session: Session,
    *,
    image_width: int | None,
    image_height: int | None,
) -> MobileLPRContext | None:
    """Construct a ``MobileLPRContext`` if ``session`` has GPS pose set.

    Returns ``None`` when the session lacks any of latitude/longitude/heading
    or when the image dimensions cannot be determined — in either case the
    pipeline runs without spot validation, exactly like a static camera.
    """
    if session.gps_latitude is None or session.gps_longitude is None:
        return None
    if session.gps_heading_deg is None:
        return None
    if not image_width or not image_height:
        return None

    intrinsics = CameraIntrinsics(
        height_m=float(settings.MOBILE_LPR_CAMERA_HEIGHT_M),
        focal_length_px=float(settings.MOBILE_LPR_CAMERA_FOCAL_PX),
        pitch_deg=float(settings.MOBILE_LPR_CAMERA_PITCH_DEG),
        image_width_px=int(image_width),
        image_height_px=int(image_height),
    )
    calculator = PlateGeolocationCalculator(
        intrinsics,
        plate_height_m=float(settings.MOBILE_LPR_PLATE_HEIGHT_M),
    )
    pose = VehiclePose(
        latitude=float(session.gps_latitude),
        longitude=float(session.gps_longitude),
        heading_deg=float(session.gps_heading_deg),
    )
    return MobileLPRContext(
        calculator=calculator,
        pose=pose,
        search_radius_m=float(settings.MOBILE_LPR_SEARCH_RADIUS_M),
    )


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


def _estimate_sampled_total_frames(
    *, total_frames: int | None, video_fps: float | None, fps_target: int
) -> int | None:
    """Estimate how many sampled frames extract_frames will yield."""
    if total_frames is None or total_frames <= 0:
        return None
    if video_fps is None or not np.isfinite(video_fps) or video_fps <= 0:
        video_fps = 25.0
    interval = max(1, round(float(video_fps) / max(1, int(fps_target))))
    return max(1, (int(total_frames) + interval - 1) // interval)


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
    plate_type: str | None = None,
) -> OCRCandidate:
    best_valid = OCRCandidate()
    best_any = OCRCandidate()

    for angle in ocr_angles:
        candidate_image = _rotate_bound(crop, angle)
        raw_text, confidence = _read_with_recognizer(recognize, candidate_image)
        if raw_text:
            normalized = normalize_plate(raw_text, plate_type=plate_type)
            is_valid = not is_invalid_plate(normalized)
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


def _prefer_candidate(current: OCRCandidate, fallback: OCRCandidate) -> OCRCandidate:
    if fallback.is_valid and not current.is_valid:
        return fallback
    if fallback.is_valid == current.is_valid and fallback.confidence > current.confidence:
        return fallback
    return current


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
) -> list[PlateTrack]:
    stale_ids = [
        track_id
        for track_id, track in active_tracks.items()
        if frame_index - track.last_frame > max_age
    ]
    stale_tracks = [active_tracks[track_id] for track_id in stale_ids]
    for track_id in stale_ids:
        del active_tracks[track_id]
    return stale_tracks


def _assign_track(
    *,
    bbox: tuple[int, int, int, int],
    frame_index: int,
    active_tracks: dict[int, PlateTrack],
    all_tracks: dict[int, PlateTrack],
    next_track_id: int,
    min_iou: float,
) -> tuple[PlateTrack, int]:
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


def _levenshtein_distance(a: str, b: str) -> int:
    """Compute the Levenshtein edit distance between strings *a* and *b*."""
    if len(a) < len(b):
        return _levenshtein_distance(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def _find_near_match(track: PlateTrack, plate: str) -> str | None:
    """Return the existing vote key within Levenshtein distance ≤1, if any.

    Prefers the variant with more accumulated votes, then by average OCR
    confidence, then by lexical order for deterministic tie-breaking.
    """
    if plate in track.votes:
        return plate

    candidates: list[tuple[int, float, str]] = []
    for existing, votes in track.votes.items():
        if len(existing) != len(plate):
            continue
        if _levenshtein_distance(existing, plate) > 1:
            continue
        avg_conf = track.conf_sums.get(existing, 0.0) / max(1, votes)
        candidates.append((votes, avg_conf, existing))

    if candidates:
        return max(candidates)[2]
    return None


def _find_registry_near_match(
    finalized_registry: dict[str, tuple[int, float]], plate: str
) -> str | None:
    if plate in finalized_registry:
        return plate

    candidates: list[tuple[int, float, str]] = []
    for existing, (votes, avg_conf) in finalized_registry.items():
        if len(existing) != len(plate):
            continue
        if _levenshtein_distance(existing, plate) > 1:
            continue
        candidates.append((votes, avg_conf, existing))

    if candidates:
        return max(candidates)[2]
    return None


def _register_track_vote(
    track: PlateTrack, plate: str, confidence: float, *, weight: float = 1.0
) -> None:
    """Register a vote, merging near-match plates (Levenshtein ≤1).

    When the new *plate* differs from an existing vote by at most 1 character,
    the votes are folded into the variant that already has more votes.  This
    prevents fragmentation like ``CJ12ABC`` vs ``CJ12ABG`` splitting the
    count for the same physical vehicle.
    """
    # Safety guard: invalid normalisation output must never influence voting.
    if not plate or is_invalid_plate(plate):
        return

    near = _find_near_match(track, plate)

    if near is not None and near != plate:
        # The existing variant already has votes — merge into it.
        # If the new plate is actually the one with higher total confidence,
        # rename the bucket.
        existing_votes = track.votes[near]
        new_weight = weight  # This is the first vote for `plate`
        if new_weight > existing_votes:
            # Promote new spelling: move old → new
            track.votes[plate] = track.votes.pop(near) + 1
            track.conf_sums[plate] = track.conf_sums.pop(near, 0.0) + confidence
        else:
            track.votes[near] += 1
            track.conf_sums[near] = track.conf_sums.get(near, 0.0) + confidence
    else:
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


async def _resolve_spot_match(
    *,
    track: PlateTrack,
    plate_text: str,
    mobile_lpr: MobileLPRContext,
    db: AsyncSession,
) -> tuple[float, float, str, float | None, int | None] | None:
    """Project the track's best bbox to GPS and validate against ParkingSpot.

    Returns ``None`` if the geometry is degenerate (plate above the horizon).
    Otherwise returns ``(target_lat, target_lon, status, distance_m, spot_id)``
    so the caller can stamp every detection in the track with the same result.
    """
    bbox = track.best_conf_bbox or track.bbox
    bbox_xywh = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

    try:
        target_lat, target_lon = mobile_lpr.calculator.project(
            mobile_lpr.pose, bbox_xywh
        )
    except ValueError as exc:
        logger.debug("Skipping geolocation for track %s: %s", track.track_id, exc)
        return None

    spot_match = await validate_parking_at_location(
        target_lat=target_lat,
        target_lon=target_lon,
        plate_text=plate_text,
        db=db,
        radius_m=mobile_lpr.search_radius_m,
    )
    return (
        target_lat,
        target_lon,
        spot_match.status,
        spot_match.distance_m,
        spot_match.spot.id if spot_match.spot is not None else None,
    )


async def _finalize_track_with_fallback(
    track: PlateTrack,
    *,
    db: AsyncSession,
    zone_id: int | None,
    min_track_votes: int,
    finalized_registry: dict[str, tuple[int, float]],
    mobile_lpr: MobileLPRContext | None = None,
) -> None:
    """Finalize unresolved tracks with the best-confidence valid plate.

    Every detection in the track is relabelled with the chosen canonical plate
    so the frontend groups them as a single card.  Near-matches against
    already-finalized plates (Levenshtein ≤1) collapse onto the existing
    canonical — this prevents split finals like ``SJ03FAL`` / ``SJ03FAU`` for
    the same physical vehicle when OCR disagrees slightly between tracks.
    """
    if track.finalized:
        return

    best_plate, best_votes, best_avg_conf, _stability = _track_best_plate(track)
    chosen_plate: str | None = None
    if best_plate and best_votes >= min_track_votes:
        chosen_plate = best_plate
    elif track.best_conf_plate:
        chosen_plate = track.best_conf_plate

    if not chosen_plate:
        return

    is_stable_choice = bool(best_plate and best_votes >= min_track_votes and chosen_plate == best_plate)
    # Always consult the cross-track registry so stable-vote finalizations
    # also collapse onto earlier near-match canonicals.  Without this, two
    # tracks of the same car can each reach MIN_TRACK_VOTES with slightly
    # different OCR reads and produce duplicate cards in the UI.
    near_canonical = _find_registry_near_match(finalized_registry, chosen_plate)
    if near_canonical is not None and near_canonical != chosen_plate:
        chosen_plate = near_canonical

    normalized = normalize_plate(chosen_plate)
    if is_invalid_plate(normalized):
        return

    representative_id = (
        track.best_conf_detection_id
        if chosen_plate == track.best_conf_plate
        else track.last_detection_id
    ) or track.last_detection_id
    if representative_id is None:
        return

    representative = await db.get(Detection, representative_id)
    if representative is None:
        return
    if representative.ticket_status is not None:
        track.finalized = True
        return

    compact_text = compact_plate(normalized)
    plate = await _get_or_create_plate(compact_text, db)

    status, expires_at, _tid, _sid = await lookup_ticket(
        plate_text=compact_text,
        zone_id=zone_id,
        checked_at=datetime.now(UTC),
        db=db,
        detection_id=representative.id,
    )

    # Mobile-LPR: project the plate's pixel position to a GPS coordinate and
    # check it against the registered parking spots.  Skipped for static
    # cameras (mobile_lpr is None) or when the geometry is degenerate.
    spot_result: tuple[float, float, str, float | None, int | None] | None = None
    if mobile_lpr is not None:
        spot_result = await _resolve_spot_match(
            track=track,
            plate_text=compact_text,
            mobile_lpr=mobile_lpr,
            db=db,
        )

    # Relabel every detection in the track.  ocr_raw_text is preserved so the
    # per-frame OCR remains visible; only the normalized view collapses.
    for detection_id in track.detection_ids:
        detection = await db.get(Detection, detection_id)
        if detection is None or detection.ticket_status is not None:
            continue
        detection.ocr_normalized_text = normalized
        detection.is_valid_ro_plate = True
        detection.plate_id = plate.id
        detection.ticket_status = status
        detection.ticket_expires_at = expires_at
        if spot_result is not None:
            (
                detection.target_latitude,
                detection.target_longitude,
                detection.spot_match_status,
                detection.target_distance_m,
                detection.matched_spot_id,
            ) = spot_result

    await db.flush()
    track.finalized = True

    evidence_votes = best_votes if is_stable_choice else max(1, best_votes)
    evidence_avg_conf = (
        track.conf_sums.get(best_plate, 0.0) / max(1, best_votes)
        if best_plate and best_votes > 0
        else track.best_confidence
    )
    current = finalized_registry.get(normalized)
    if current is None or (evidence_votes, evidence_avg_conf) > current:
        finalized_registry[normalized] = (evidence_votes, evidence_avg_conf)


async def process_session(
    session_id: uuid.UUID,
    db: AsyncSession,
    *,
    ocr: Recognizer | None = None,
    zone_id: int | None = None,
    mobile_lpr: MobileLPRContext | None = None,
) -> None:
    """Process every frame in a session's video file with robust OCR/voting.

    When ``mobile_lpr`` is provided, every finalised track also has its
    bounding box projected to a GPS coordinate and validated against the
    registered ``ParkingSpot`` rows; the resulting status/coords are stamped
    on every detection in the track.  Pass ``None`` for the static-camera
    flow (preserves existing behaviour).
    """
    recognize = ocr or _noop_ocr

    session = await db.get(Session, session_id)
    if session is None:
        raise ValueError(f"Session {session_id} not found")

    video_path = Path(settings.UPLOAD_DIR) / f"{session_id}_{session.source_filename}"
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    detector = PlateDetector(settings.MODEL_PATH, settings.DETECTION_CONF)
    fps_target = (
        max(1, int(round(session.fps_target)))
        if session.fps_target is not None and session.fps_target > 0
        else int(settings.FPS_TARGET)
    )

    min_ocr_conf = max(0.0, min(1.0, float(settings.OCR_MIN_CONF)))
    ocr_angles = _parse_ocr_angles(settings.OCR_ANGLES)
    min_sharpness = max(0.0, float(settings.OCR_MIN_SHARPNESS))
    track_max_age = max(1, int(settings.TRACK_MAX_AGE))
    track_min_iou = max(0.01, min(0.99, float(settings.TRACK_MIN_IOU)))
    min_track_votes = max(1, int(settings.MIN_TRACK_VOTES))
    progress_commit_every = 1
    session_crops_dir = Path(settings.CROPS_DIR) / str(session_id)
    session_crops_dir.mkdir(parents=True, exist_ok=True)

    sampled_total_frames = None
    try:
        video_info = get_video_info(str(video_path))
        sampled_total_frames = _estimate_sampled_total_frames(
            total_frames=video_info.get("total_frames"),
            video_fps=video_info.get("fps"),
            fps_target=fps_target,
        )
    except Exception:
        sampled_total_frames = None

    session.status = "processing"
    session.error_message = None
    session.fps_target = float(fps_target)
    session.frames_processed = 0
    if sampled_total_frames is not None:
        session.total_frames = sampled_total_frames
    session.ended_at = None
    await db.commit()

    frames_processed = 0
    active_tracks: dict[int, PlateTrack] = {}
    all_tracks: dict[int, PlateTrack] = {}
    finalized_plate_registry: dict[str, tuple[int, float]] = {}
    next_track_id = 1

    try:
        for frame_index, pts_ms, frame_img in extract_frames(
            str(video_path), fps_target
        ):
            stale_tracks = _prune_stale_tracks(
                active_tracks,
                frame_index=frame_index,
                max_age=track_max_age,
            )
            for stale_track in stale_tracks:
                await _finalize_track_with_fallback(
                    stale_track,
                    db=db,
                    zone_id=zone_id,
                    min_track_votes=min_track_votes,
                    finalized_registry=finalized_plate_registry,
                    mobile_lpr=mobile_lpr,
                )

            detections = detector.detect(frame_img, frame_index=frame_index)
            if not detections:
                frames_processed += 1
                session.frames_processed = frames_processed
                if frames_processed % progress_commit_every == 0:
                    await db.commit()
                continue

            frame = Frame(session_id=session_id, frame_index=frame_index, pts_ms=pts_ms)
            db.add(frame)
            await db.flush()

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
                    min_iou=track_min_iou,
                )
                # Adaptive padding: proportional to bbox size, clamped [2, 8].
                adaptive_pad = max(2, min(8, int(min(bbox_w, bbox_h) * 0.08)))
                try:
                    crop = PlateDetector.crop_plate(frame_img, bbox, padding=adaptive_pad)
                except ValueError:
                    continue

                crop_filename = f"{frame.id}_{bbox_x}_{bbox_y}.jpg"
                crop_relative_path = Path(str(session_id)) / crop_filename
                crop_path = session_crops_dir / crop_filename
                cv2.imwrite(str(crop_path), crop)

                # --- Color-based plate type classification ---
                color_hint = classify_plate_color(crop)
                plate_type_str: str | None = (
                    color_hint.value if color_hint != PlateType.UNKNOWN else None
                )

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
                        plate_type=plate_type_str,
                    )
                else:
                    candidate = _best_ocr_candidate(
                        recognize,
                        crop,
                        ocr_angles=ocr_angles,
                        plate_type=plate_type_str,
                    )

                if not candidate.raw_text:
                    fallback_candidate = _best_ocr_candidate(
                        recognize,
                        crop,
                        ocr_angles=ocr_angles,
                        plate_type=plate_type_str,
                    )
                    candidate = _prefer_candidate(candidate, fallback_candidate)

                if candidate.raw_text and candidate.confidence >= min_ocr_conf:
                    if candidate.is_valid:
                        _register_track_vote(
                            track, candidate.normalized, candidate.confidence
                        )

                raw_plate_text = candidate.raw_text or None
                normalized_plate_text = candidate.normalized if candidate.raw_text else None
                is_valid_ro_plate = (
                    bool(candidate.raw_text)
                    and candidate.is_valid
                    and candidate.confidence >= min_ocr_conf
                )

                detection = Detection(
                    frame_id=frame.id,
                    bbox_x=bbox_x,
                    bbox_y=bbox_y,
                    bbox_w=bbox_w,
                    bbox_h=bbox_h,
                    detection_confidence=confidence,
                    crop_image_path=crop_relative_path.as_posix(),
                    ocr_raw_text=raw_plate_text,
                    ocr_normalized_text=normalized_plate_text,
                    is_valid_ro_plate=is_valid_ro_plate,
                )
                db.add(detection)
                await db.flush()
                track.last_detection_id = detection.id
                track.detection_ids.append(detection.id)

                if (
                    candidate.raw_text
                    and candidate.confidence >= min_ocr_conf
                    and candidate.is_valid
                    and candidate.confidence >= track.best_confidence
                ):
                    track.best_confidence = candidate.confidence
                    track.best_conf_plate = candidate.normalized
                    track.best_conf_detection_id = detection.id
                    track.best_conf_bbox = bbox

            frames_processed += 1
            session.frames_processed = frames_processed
            if frames_processed % progress_commit_every == 0:
                await db.commit()

        for track in all_tracks.values():
            await _finalize_track_with_fallback(
                track,
                db=db,
                zone_id=zone_id,
                min_track_votes=min_track_votes,
                finalized_registry=finalized_plate_registry,
                mobile_lpr=mobile_lpr,
            )

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
    except Exception as exc:
        session.frames_processed = frames_processed
        session.status = "failed"
        session.error_message = str(exc)
        session.ended_at = datetime.now(UTC)
        await db.commit()
        logger.exception("Session %s failed during processing", session_id)
        raise
