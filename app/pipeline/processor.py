"""
app/pipeline/processor.py

End-to-end session processor with robust OCR:
frame extraction -> detection -> deskew/blur handling -> multi-angle OCR ->
temporal voting -> record -> ticket lookup (audit).
"""

from __future__ import annotations

import asyncio
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
from app.core.events import publish
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
from app.pipeline.plate_validator import (
    compact_plate,
    extract_county,
    is_invalid_plate,
    normalize_plate,
)
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
        max_distance_m=float(settings.MOBILE_LPR_MAX_DISTANCE_M),
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


class BatchRecognizer(Protocol):
    """Optional batched OCR — one call per frame instead of one per detection.

    Implementations should return one ``(text, confidence)`` tuple per input
    crop, in the same order, so the per-frame loop can pre-seed the candidate
    for each detection without an extra Python/Paddle round-trip.
    """

    def __call__(
        self,
        crops: list[np.ndarray],
        *,
        plate_types: list[str | None] | None = None,
    ) -> list[tuple[str, float]]: ...


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
    early_exit_conf: float | None = None,
) -> OCRCandidate:
    """Iterate angles and return the best (valid + highest-conf) candidate.

    When ``early_exit_conf`` is set and the *first* angle that produces a
    valid plate hits ``confidence >= early_exit_conf``, the remaining
    angles are skipped — most plates are not tilted, so this short-circuit
    drops the average angle-sweep from 5 OCR calls to 1 without affecting
    the cases that actually need rotation.  Pass ``None`` to keep the old
    behaviour (sweep every angle).
    """
    best_valid = OCRCandidate()
    best_any = OCRCandidate()

    # Run the un-rotated angle first so the early-exit shortcut fires on
    # the most common case.  ``ocr_angles`` already canonicalises 0° to be
    # present in the list (see ``_parse_ocr_angles``).
    ordered_angles = sorted(ocr_angles, key=lambda a: abs(a))

    for angle in ordered_angles:
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

        if (
            early_exit_conf is not None
            and best_valid.is_valid
            and best_valid.confidence >= early_exit_conf
        ):
            # First valid+high-conf hit wins — no need to keep sweeping.
            return best_valid

    return best_valid if best_valid.raw_text else best_any


def _per_detection_ocr_sweep(
    recognize: Recognizer,
    crop: np.ndarray,
    *,
    detection_confidence: float,
    plate_type: str | None,
    ocr_angles: tuple[float, ...],
    early_exit_conf: float,
    min_sharpness: float,
) -> OCRCandidate:
    """Run the deskew + multi-angle / multi-pass OCR sweep for one detection.

    Pulled out of the per-frame loop so the whole CPU-bound chain (deskew,
    sharpness, motion-blur enhance, multi-angle OCR) can be dispatched with
    a single ``asyncio.to_thread(...)`` call — concurrent sessions then
    interleave their DB writes / SSE pushes on the event loop while their
    crops are crunched in worker threads.
    """
    rectified = deskew_plate(crop)
    if rectified.size == 0:
        rectified = crop

    angles_for_detection = (
        (0.0,) if detection_confidence >= 0.85 else ocr_angles
    )

    sharpness = _sharpness_score(rectified)
    if sharpness >= min_sharpness:
        ocr_input = (
            _enhance_motion_blur(rectified)
            if sharpness < min_sharpness * 1.35
            else rectified
        )
        sweep_candidate = _best_ocr_candidate(
            recognize,
            ocr_input,
            ocr_angles=angles_for_detection,
            plate_type=plate_type,
            early_exit_conf=early_exit_conf,
        )
    else:
        sweep_candidate = _best_ocr_candidate(
            recognize,
            crop,
            ocr_angles=angles_for_detection,
            plate_type=plate_type,
            early_exit_conf=early_exit_conf,
        )

    if not sweep_candidate.raw_text:
        fallback_candidate = _best_ocr_candidate(
            recognize,
            crop,
            ocr_angles=ocr_angles,
            plate_type=plate_type,
            early_exit_conf=early_exit_conf,
        )
        sweep_candidate = _prefer_candidate(sweep_candidate, fallback_candidate)

    return sweep_candidate


def _track_is_stable(
    track: PlateTrack,
    *,
    min_votes: int,
    min_confidence: float,
) -> bool:
    """Return True when a track has converged enough to skip per-frame OCR.

    Stable means: we have observed the track at least twice the vote
    threshold, the leading plate has reached the vote threshold, and the
    best per-detection confidence is high.  Once stable, additional OCR
    runs are pure overhead — the finalizer relabels every detection in
    the track with the canonical plate, so leaving the per-frame OCR text
    unset costs nothing downstream.
    """
    if track.observations < min_votes * 2:
        return False
    if track.best_conf_plate is None:
        return False
    if track.best_confidence < min_confidence:
        return False
    if not track.votes:
        return False
    leading_votes = max(track.votes.values(), default=0)
    if leading_votes < min_votes:
        return False
    # Require ≥60% of votes on a single plate — prevents locking in a plate
    # when OCR is split between two near-miss variants (e.g. CJ12ABC / CJ12ABG).
    total_votes = sum(track.votes.values())
    return (leading_votes / max(1, total_votes)) >= 0.60


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

    def _score(item: tuple[str, int]) -> tuple[float, int, float, str]:
        plate, votes = item
        conf_total = track.conf_sums.get(plate, 0.0)
        avg_conf = conf_total / max(1, votes)
        # Primary key: total accumulated confidence — 3 reads at 0.95 (2.85)
        # beats 5 reads at 0.50 (2.50).  Ties broken by vote count then avg.
        return conf_total, votes, avg_conf, plate

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
        # Backfill county_code on existing rows that pre-date the registry
        # filter — keeps the sidebar's county column populated without a
        # migration script.
        if plate.county_code is None:
            plate.county_code = extract_county(normalized_text)
        await db.flush()
        return plate

    plate = Plate(
        normalized_text=normalized_text,
        county_code=extract_county(normalized_text),
    )
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
    session_id: str | None = None,
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
    #
    # GPS handling is split between per-detection and per-track:
    #   • target_latitude / target_longitude are computed from each detection's
    #     own bbox so different cars in the same track sit at different points.
    #     If eager projection ran at frame-time the value is already set; for
    #     older rows (or when eager projection raised) we retry here.
    #   • spot_match_status / target_distance_m / matched_spot_id are a
    #     track-level decision driven by the best-confidence frame, so they
    #     apply identically to every detection in the track.
    if spot_result is not None:
        _, _, track_status, track_distance, track_spot_id = spot_result
    else:
        track_status = None
        track_distance = None
        track_spot_id = None

    # Fetch all detections for this track in one query instead of N individual
    # db.get() calls — after the per-frame commits the identity map is expired,
    # so each db.get() would otherwise hit the database separately.
    batch_result = await db.execute(
        select(Detection).where(Detection.id.in_(track.detection_ids))
    )
    detections_by_id = {d.id: d for d in batch_result.scalars().all()}

    for detection_id in track.detection_ids:
        detection = detections_by_id.get(detection_id)
        if detection is None or detection.ticket_status is not None:
            continue
        detection.ocr_normalized_text = normalized
        detection.is_valid_ro_plate = True
        detection.plate_id = plate.id
        detection.ticket_status = status
        detection.ticket_expires_at = expires_at

        if mobile_lpr is not None and (
            detection.target_latitude is None
            or detection.target_longitude is None
        ):
            try:
                d_lat, d_lon = mobile_lpr.calculator.project(
                    mobile_lpr.pose,
                    (
                        float(detection.bbox_x),
                        float(detection.bbox_y),
                        float(detection.bbox_w),
                        float(detection.bbox_h),
                    ),
                )
            except ValueError:
                pass
            else:
                detection.target_latitude = d_lat
                detection.target_longitude = d_lon

        if spot_result is not None:
            detection.spot_match_status = track_status
            detection.target_distance_m = track_distance
            detection.matched_spot_id = track_spot_id

    await db.flush()
    track.finalized = True

    if session_id is not None:
        publish(session_id, "detection_finalized", {"plate": normalized})

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
    ocr_batch: BatchRecognizer | None = None,
    zone_id: int | None = None,
    mobile_lpr: MobileLPRContext | None = None,
) -> None:
    """Process every frame in a session's video file with robust OCR/voting.

    When ``mobile_lpr`` is provided, every finalised track also has its
    bounding box projected to a GPS coordinate and validated against the
    registered ``ParkingSpot`` rows; the resulting status/coords are stamped
    on every detection in the track.  Pass ``None`` for the static-camera
    flow (preserves existing behaviour).

    When ``ocr_batch`` is provided AND a frame yields ≥2 non-stable detections,
    a single batched OCR call covers them as a fast first pass.  Detections
    that don't reach the early-exit confidence on the batch result fall back
    to the existing per-detection multi-pass / multi-angle path, so accuracy
    is preserved.
    """
    recognize = ocr or _noop_ocr
    recognize_batch = ocr_batch

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
    early_exit_conf = max(0.0, min(1.0, float(settings.OCR_EARLY_EXIT_CONF)))
    skip_stable_tracks = bool(settings.OCR_SKIP_STABLE_TRACKS)
    skip_stable_conf = max(0.0, min(1.0, float(settings.OCR_SKIP_STABLE_CONF)))
    progress_commit_every = max(1, int(settings.PROCESSOR_COMMIT_EVERY_FRAMES))
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
                    session_id=str(session_id),
                )

            # Dispatch the YOLO inference to a worker thread so other sessions'
            # async work (DB writes, SSE pushes, frame reads) can interleave
            # while this one's frame is being analysed.  Each session has its
            # own ``detector`` instance so the model state is not shared.
            detections = await asyncio.to_thread(
                detector.detect, frame_img, frame_index=frame_index
            )
            if not detections:
                frames_processed += 1
                session.frames_processed = frames_processed
                if frames_processed % progress_commit_every == 0:
                    await db.commit()
                    publish(str(session_id), "frame_processed", {
                        "frames_processed": frames_processed,
                        "total_frames": session.total_frames,
                    })
                continue

            frame = Frame(session_id=session_id, frame_index=frame_index, pts_ms=pts_ms)
            db.add(frame)
            await db.flush()

            # ---- Per-frame staging ----
            # Build the per-detection state up front so we can (optionally)
            # fire one batched OCR call for the whole frame before the
            # per-detection inner loop runs its multi-angle / multi-pass
            # fallback.  This is the [B1] amortisation: when a frame contains
            # 2+ non-stable plates, batching saves N-1 PaddleOCR round-trips.
            staged: list[dict] = []  # one entry per kept detection
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
                # Adaptive padding: proportional to bbox size, clamped [4, 16].
                adaptive_pad = max(4, min(16, int(min(bbox_w, bbox_h) * 0.12)))
                try:
                    crop = PlateDetector.crop_plate(frame_img, bbox, padding=adaptive_pad)
                except ValueError:
                    continue

                crop_filename = f"{frame.id}_{bbox_x}_{bbox_y}.jpg"
                crop_relative_path = Path(str(session_id)) / crop_filename
                crop_path = session_crops_dir / crop_filename
                # Disk I/O off the event loop — even at 5–10 ms per crop this
                # adds up across a long video and pins the loop.
                await asyncio.to_thread(cv2.imwrite, str(crop_path), crop)

                color_hint = classify_plate_color(crop)
                plate_type_str: str | None = (
                    color_hint.value if color_hint != PlateType.UNKNOWN else None
                )

                # Once a track has converged we skip OCR entirely — the
                # finalizer relabels every detection in the track anyway.
                track_stable = skip_stable_tracks and _track_is_stable(
                    track,
                    min_votes=min_track_votes,
                    min_confidence=skip_stable_conf,
                )

                staged.append(
                    {
                        "bbox": bbox,
                        "bbox_x": bbox_x,
                        "bbox_y": bbox_y,
                        "bbox_w": bbox_w,
                        "bbox_h": bbox_h,
                        "confidence": confidence,
                        "crop": crop,
                        "crop_relative_path": crop_relative_path,
                        "track": track,
                        "plate_type": plate_type_str,
                        "track_stable": track_stable,
                    }
                )

            # ---- Optional batched first-pass OCR ----
            batch_results: dict[int, tuple[str, float]] = {}
            if recognize_batch is not None:
                batch_indices = [
                    idx
                    for idx, item in enumerate(staged)
                    if not item["track_stable"]
                ]
                # Only worth batching when ≥2 crops share the call — a single
                # crop pays the same overhead either way and avoids any risk
                # from batch-API quirks.
                if len(batch_indices) >= 2:
                    batch_crops = [staged[idx]["crop"] for idx in batch_indices]
                    batch_types = [staged[idx]["plate_type"] for idx in batch_indices]
                    try:
                        results = recognize_batch(
                            batch_crops, plate_types=batch_types
                        )
                    except Exception as exc:
                        logger.debug(
                            "Frame %d batched OCR failed (%s); falling back to per-detection",
                            frame_index, exc,
                        )
                        results = []
                    if len(results) == len(batch_indices):
                        for idx, payload in zip(batch_indices, results):
                            batch_results[idx] = payload

            # ---- Per-detection OCR + DB writes ----
            for staged_idx, item in enumerate(staged):
                bbox = item["bbox"]
                bbox_x = item["bbox_x"]
                bbox_y = item["bbox_y"]
                bbox_w = item["bbox_w"]
                bbox_h = item["bbox_h"]
                confidence = item["confidence"]
                crop = item["crop"]
                crop_relative_path = item["crop_relative_path"]
                track = item["track"]
                plate_type_str = item["plate_type"]
                track_stable = item["track_stable"]

                candidate = OCRCandidate()
                if not track_stable:
                    # Step 1 — try the batched first-pass result if we have one.
                    batched = batch_results.get(staged_idx)
                    if batched is not None:
                        raw_text, raw_conf = batched
                        if raw_text:
                            normalized = normalize_plate(raw_text, plate_type=plate_type_str)
                            candidate = OCRCandidate(
                                raw_text=raw_text,
                                confidence=raw_conf,
                                normalized=normalized,
                                is_valid=not is_invalid_plate(normalized),
                                angle=0.0,
                            )

                    # Step 2 — fall back to the per-detection multi-angle /
                    # deskew pass when the batched result was empty or didn't
                    # clear the early-exit threshold.
                    needs_fallback = not (
                        candidate.is_valid
                        and candidate.confidence >= early_exit_conf
                    )
                    if needs_fallback:
                        rectified = deskew_plate(crop)
                        if rectified.size == 0:
                            rectified = crop

                        angles_for_detection = (
                            (0.0,) if confidence >= 0.85 else ocr_angles
                        )

                        sharpness = _sharpness_score(rectified)
                        if sharpness >= min_sharpness:
                            ocr_input = (
                                _enhance_motion_blur(rectified)
                                if sharpness < min_sharpness * 1.35
                                else rectified
                            )
                            sweep_candidate = _best_ocr_candidate(
                                recognize,
                                ocr_input,
                                ocr_angles=angles_for_detection,
                                plate_type=plate_type_str,
                                early_exit_conf=early_exit_conf,
                            )
                        else:
                            sweep_candidate = _best_ocr_candidate(
                                recognize,
                                crop,
                                ocr_angles=angles_for_detection,
                                plate_type=plate_type_str,
                                early_exit_conf=early_exit_conf,
                            )

                        if not sweep_candidate.raw_text:
                            fallback_candidate = _best_ocr_candidate(
                                recognize,
                                crop,
                                ocr_angles=ocr_angles,
                                plate_type=plate_type_str,
                                early_exit_conf=early_exit_conf,
                            )
                            sweep_candidate = _prefer_candidate(
                                sweep_candidate, fallback_candidate
                            )

                        candidate = _prefer_candidate(candidate, sweep_candidate)

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
                # Eagerly project this detection's bbox to a GPS coordinate so
                # the live (pre-finalization) UI can show per-car GPS.  The
                # spot-match status is left for finalization, when the canonical
                # plate text is known and the cross-track registry has settled.
                if mobile_lpr is not None:
                    try:
                        proj_lat, proj_lon = mobile_lpr.calculator.project(
                            mobile_lpr.pose,
                            (float(bbox_x), float(bbox_y), float(bbox_w), float(bbox_h)),
                        )
                    except ValueError:
                        proj_lat = proj_lon = None
                    if proj_lat is not None and proj_lon is not None:
                        detection.target_latitude = proj_lat
                        detection.target_longitude = proj_lon
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
                publish(str(session_id), "frame_processed", {
                    "frames_processed": frames_processed,
                    "total_frames": session.total_frames,
                })

        for track in all_tracks.values():
            await _finalize_track_with_fallback(
                track,
                db=db,
                zone_id=zone_id,
                min_track_votes=min_track_votes,
                finalized_registry=finalized_plate_registry,
                mobile_lpr=mobile_lpr,
                session_id=str(session_id),
            )

        session.frames_processed = frames_processed
        session.status = "completed"
        session.ended_at = datetime.now(UTC)
        await db.commit()
        publish(str(session_id), "session_completed", {
            "frames_processed": frames_processed,
            "total_frames": session.total_frames,
        })

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
        # When the failure originates inside a flush, the session sits in a
        # rolled-back state — committing without rolling back first raises
        # PendingRollbackError and the status update never lands, leaving the
        # row stuck on "processing" and the frontend hanging on a dead task.
        await db.rollback()
        fresh_session = await db.get(Session, session_id)
        if fresh_session is not None:
            fresh_session.frames_processed = frames_processed
            fresh_session.status = "failed"
            fresh_session.error_message = str(exc)
            fresh_session.ended_at = datetime.now(UTC)
            await db.commit()
        publish(str(session_id), "session_failed", {"error": str(exc)})
        logger.exception("Session %s failed during processing", session_id)
        raise
