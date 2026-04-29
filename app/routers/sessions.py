import asyncio
import json
import logging
import uuid
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, List

import aiofiles
import numpy as np
from fastapi import APIRouter, Depends, Form, HTTPException, Query, Response, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import AsyncSessionLocal, get_db
from app.models.parking import ParkingZone
from app.core.events import publish, subscribe, unsubscribe
from app.models.detection import Detection, Frame, Session
from app.models.parking import TicketCheck
from app.pipeline.ocr import PlateReader
from app.pipeline.processor import build_mobile_lpr_context, process_session
from app.schemas.detection import DetectionOut, SessionOut
from app.pipeline.frame_extractor import get_video_info
from app.pipeline.gps_validation import (
    GpsSource,
    GpsValidationError,
    GpsValidationStatus,
    ResolvedGpsPose,
    ValidatedCoordinates,
    cross_validate_sources,
    validate_coordinates,
    validate_heading,
)
from app.pipeline.video_metadata import MediaGpsMetadata, extract_media_gps_metadata

router = APIRouter()
_plate_reader: PlateReader | None = None
_processing_tasks: dict[uuid.UUID, asyncio.Task[None]] = {}
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov"}
# Streaming-write chunk size for uploads.  At 4 MiB we cap RAM use at a
# constant level even for multi-GB videos, while still amortising syscall
# overhead.  Avoids OOM from `await file.read()` on large files.
_UPLOAD_CHUNK_BYTES = 4 * 1024 * 1024
_SESSION_UPLOAD_PREFIX = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}_.+"
)
logger = logging.getLogger(__name__)
UTC = timezone.utc


def _get_plate_reader() -> PlateReader:
    global _plate_reader
    if _plate_reader is None:
        _plate_reader = PlateReader.from_settings()
    return _plate_reader


def _recognize_plate(crop: np.ndarray) -> tuple[str | None, float]:
    text, confidence = _get_plate_reader().read_plate(crop)
    return (text or None, confidence)


def _recognize_plates_batch(
    crops: list[np.ndarray],
    *,
    plate_types: list[str | None] | None = None,
) -> list[tuple[str, float]]:
    """Batched OCR — one PaddleOCR call for every crop in a frame."""
    return _get_plate_reader().read_plates_batch(crops, plate_types=plate_types)


def _is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in _VIDEO_EXTENSIONS


def _is_session_generated_upload(filename: str) -> bool:
    return bool(_SESSION_UPLOAD_PREFIX.match(filename))


def _resolve_test_upload_file(filename: str) -> Path:
    if not filename or Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    source = Path(settings.UPLOAD_DIR) / filename
    if not source.exists() or not source.is_file():
        raise HTTPException(status_code=404, detail="Test file not found")
    if not _is_video_file(source):
        raise HTTPException(status_code=400, detail="Unsupported test file format")
    return source


def _effective_fps_target(value: float | None) -> float:
    if value is None or value <= 0:
        return float(max(1, int(settings.FPS_TARGET)))
    return float(max(1, int(round(value))))


def _estimate_sampled_total_frames(video_info: dict, fps_target: float) -> int | None:
    total_frames_raw = video_info.get("total_frames")
    video_fps_raw = video_info.get("fps")
    if not isinstance(total_frames_raw, (int, float)) or total_frames_raw <= 0:
        return None
    total_frames = int(total_frames_raw)

    if not isinstance(video_fps_raw, (int, float)) or video_fps_raw <= 0:
        video_fps = 25.0
    else:
        video_fps = float(video_fps_raw)

    interval = max(1, round(video_fps / max(1.0, fps_target)))
    return max(1, (total_frames + interval - 1) // interval)


async def _resolve_zone_id(explicit: int | None, db: AsyncSession) -> int | None:
    """Pick a zone for the session in priority order.

    1. explicit value from the form submission
    2. DEFAULT_ZONE_ID from settings
    3. first row in parking_zones (single DB round-trip)
    4. None — no zones exist yet; ticket lookup will return "unknown"
    """
    if explicit is not None:
        return explicit
    if settings.DEFAULT_ZONE_ID is not None:
        return settings.DEFAULT_ZONE_ID
    result = await db.execute(
        select(ParkingZone.id).order_by(ParkingZone.id).limit(1)
    )
    return result.scalar_one_or_none()


def _build_mobile_lpr_for_session(session: Session) -> object | None:
    """Resolve the geolocation context for a session, if GPS pose is set.

    The image dimensions are read from the source video on disk; if the file
    is missing or unreadable we still return ``None`` so processing falls back
    to the static-camera flow rather than crashing.
    """
    if (
        session.gps_latitude is None
        or session.gps_longitude is None
        or session.gps_heading_deg is None
    ):
        return None

    video_path = Path(settings.UPLOAD_DIR) / f"{session.id}_{session.source_filename}"
    try:
        video_info = get_video_info(str(video_path))
    except Exception:
        return None

    return build_mobile_lpr_context(
        session,
        image_width=int(video_info.get("width") or 0) or None,
        image_height=int(video_info.get("height") or 0) or None,
    )


async def _run_processing_job(session_id: uuid.UUID) -> None:
    try:
        async with AsyncSessionLocal() as db:
            session = await db.get(Session, session_id)
            mobile_lpr = _build_mobile_lpr_for_session(session) if session else None
            ocr_batch = _recognize_plates_batch if settings.OCR_BATCH_PER_FRAME else None
            zone_id = (session.zone_id if session else None) or settings.DEFAULT_ZONE_ID
            await process_session(
                session_id,
                db,
                ocr=_recognize_plate,
                ocr_batch=ocr_batch,
                mobile_lpr=mobile_lpr,
                zone_id=zone_id,
            )
    except asyncio.CancelledError:
        async with AsyncSessionLocal() as db:
            session = await db.get(Session, session_id)
            if session is not None:
                session.status = "failed"
                session.error_message = "Cancelled by user"
                session.ended_at = datetime.now(UTC)
                await db.commit()
        publish(str(session_id), "session_failed", {"error": "Cancelled by user"})
        raise
    except Exception:
        logger.exception("Background processing failed for session %s", session_id)
    finally:
        _processing_tasks.pop(session_id, None)


@router.get("", response_model=List[SessionOut])
async def list_sessions(db: AsyncSession = Depends(get_db)):
    """Return all sessions."""
    result = await db.execute(select(Session).order_by(Session.created_at.desc()))
    sessions = result.scalars().all()
    return sessions


def _validated_explicit_pose(
    lat: float | None, lon: float | None, heading: float | None
) -> tuple[float, float, float] | None:
    """Validate explicit form-supplied lat/lon/heading.

    Mobile-LPR mode is all-or-nothing: a partial pose would silently disable
    spot validation and the user would never know why.  Returns ``None`` when
    no explicit pose was supplied; raises ``HTTPException(400)`` for any
    validation error so the operator gets immediate feedback.
    """
    provided = [v for v in (lat, lon, heading) if v is not None]
    if not provided:
        return None
    if len(provided) != 3:
        raise HTTPException(
            status_code=400,
            detail="Mobile-LPR requires gps_latitude, gps_longitude and gps_heading_deg together.",
        )
    try:
        coords = validate_coordinates(lat, lon)  # type: ignore[arg-type]
        heading_norm = validate_heading(heading)  # type: ignore[arg-type]
    except GpsValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return (coords.latitude, coords.longitude, heading_norm)


def _validated_metadata_pose(
    metadata: MediaGpsMetadata | None,
) -> tuple[MediaGpsMetadata, list[str]] | None:
    """Validate GPS embedded in the video container.

    Returns ``(metadata, warnings)`` when the fix is usable.  Returns
    ``None`` when the container had no GPS or the GPS failed any hard
    rule — the caller falls back to the next provenance source (config
    default) rather than aborting upload.  Soft warnings (e.g. low
    accuracy) are surfaced through the warnings list.
    """
    if metadata is None:
        return None
    try:
        validate_coordinates(metadata.latitude, metadata.longitude)
    except GpsValidationError as exc:
        logger.info(
            "Discarding video-metadata GPS (%s, %s): %s",
            metadata.latitude,
            metadata.longitude,
            exc,
        )
        return None

    warnings: list[str] = []
    # iOS' horizontal-accuracy estimate is in metres — anything beyond ~25 m
    # (consumer GPS in an urban canyon) represents significant uncertainty
    # relative to the 40 m spot-match radius, so we keep it but warn.
    if metadata.accuracy_m is not None and metadata.accuracy_m > 25.0:
        warnings.append(
            f"video metadata GPS accuracy {metadata.accuracy_m:.1f} m "
            "exceeds 25 m — projected spot match may be unreliable"
        )
    return (metadata, warnings)


def _validated_default_pose() -> tuple[float, float, float] | None:
    """Validate the configured ``MOBILE_LPR_DEFAULT_*`` fallback pose.

    The defaults bypass the API layer, so we cannot rely on the form
    validators — apply the same hard rules here.  An invalid default is
    a deployment misconfiguration; we log loudly and fall back to "no
    pose" rather than projecting plates to nonsense.
    """
    lat = settings.MOBILE_LPR_DEFAULT_LATITUDE
    lon = settings.MOBILE_LPR_DEFAULT_LONGITUDE
    heading = settings.MOBILE_LPR_DEFAULT_HEADING_DEG
    if lat is None or lon is None or heading is None:
        return None
    try:
        coords = validate_coordinates(float(lat), float(lon))
        heading_norm = validate_heading(float(heading))
    except GpsValidationError as exc:
        logger.error(
            "MOBILE_LPR_DEFAULT_* configuration is invalid (%s, %s, %s): %s",
            lat,
            lon,
            heading,
            exc,
        )
        return None
    return (coords.latitude, coords.longitude, heading_norm)


def _resolve_session_pose(
    *,
    explicit_lat: float | None,
    explicit_lon: float | None,
    explicit_heading: float | None,
    video_path: Path,
) -> ResolvedGpsPose | None:
    """Pick the best Mobile-LPR pose for a new session.

    Precedence (first usable wins):
      1. The explicit lat/lon/heading triple sent on the form.
      2. The GPS fix embedded in the uploaded video's container metadata
         (e.g. ``com.apple.quicktime.location.ISO6709`` from iOS .MOV files),
         combined with the configured default heading (or 0 = North).
      3. The configured default pose from ``settings.MOBILE_LPR_DEFAULT_*``.

    When both explicit and video-metadata coordinates are present, the
    explicit pose wins but the two are cross-checked: a >50 m disagreement
    becomes a soft warning so the operator knows they may have paired the
    wrong coordinates with the wrong upload.

    Returns ``None`` when no source produced a usable pose — the session is
    then created without GPS and processed as a static-camera capture.
    """
    metadata = extract_media_gps_metadata(video_path)
    metadata_validated = _validated_metadata_pose(metadata)

    explicit = _validated_explicit_pose(
        explicit_lat, explicit_lon, explicit_heading
    )
    if explicit is not None:
        lat, lon, heading = explicit
        warnings: list[str] = []
        if metadata_validated is not None:
            meta, _meta_warnings = metadata_validated
            agree, distance = cross_validate_sources(
                ValidatedCoordinates(latitude=lat, longitude=lon),
                ValidatedCoordinates(
                    latitude=meta.latitude, longitude=meta.longitude
                ),
            )
            if not agree:
                warnings.append(
                    f"explicit GPS disagrees with video metadata by "
                    f"{distance:.0f} m — verify the uploaded file matches "
                    "the supplied coordinates"
                )
        return ResolvedGpsPose(
            latitude=lat,
            longitude=lon,
            heading_deg=heading,
            source=GpsSource.EXPLICIT,
            status=(
                GpsValidationStatus.WARNING
                if warnings
                else GpsValidationStatus.OK
            ),
            warnings=warnings,
        )

    if metadata_validated is not None:
        meta, meta_warnings = metadata_validated
        # Heading precedence: EXIF GPSImgDirection (when the device wrote
        # one) → configured default → 0° (North).  Most ffprobe-sourced
        # video metadata does not carry a heading, so the default usually
        # wins for video uploads; EXIF photos from a phone often do.
        heading_default = settings.MOBILE_LPR_DEFAULT_HEADING_DEG
        try:
            if meta.heading_deg is not None:
                heading_norm = validate_heading(float(meta.heading_deg))
            elif heading_default is not None:
                heading_norm = validate_heading(float(heading_default))
            else:
                heading_norm = 0.0
        except GpsValidationError:
            heading_norm = 0.0
        return ResolvedGpsPose(
            latitude=meta.latitude,
            longitude=meta.longitude,
            heading_deg=heading_norm,
            source=GpsSource.VIDEO_METADATA,
            status=(
                GpsValidationStatus.WARNING
                if meta_warnings
                else GpsValidationStatus.OK
            ),
            accuracy_m=meta.accuracy_m,
            warnings=meta_warnings,
        )

    default = _validated_default_pose()
    if default is not None:
        lat, lon, heading = default
        return ResolvedGpsPose(
            latitude=lat,
            longitude=lon,
            heading_deg=heading,
            source=GpsSource.CONFIG_DEFAULT,
            status=GpsValidationStatus.OK,
            warnings=[
                "GPS pose came from MOBILE_LPR_DEFAULT_* — every session "
                "without an explicit pose will be projected to this fixed "
                "point; verify the default matches your deployment"
            ],
        )

    return None


def _apply_pose_to_session(
    session_kwargs: dict,
    pose: ResolvedGpsPose | None,
) -> None:
    """Merge a ``ResolvedGpsPose`` (or ``None``) into Session constructor kwargs.

    Centralises the field-by-field copy so both upload endpoints stay in
    sync as columns are added to the model.
    """
    if pose is None:
        session_kwargs.update(
            gps_latitude=None,
            gps_longitude=None,
            gps_heading_deg=None,
            gps_source=None,
            gps_accuracy_m=None,
            gps_validation_status=None,
            gps_validation_warnings=None,
        )
        return
    session_kwargs.update(
        gps_latitude=pose.latitude,
        gps_longitude=pose.longitude,
        gps_heading_deg=pose.heading_deg,
        gps_source=pose.source.value,
        gps_accuracy_m=pose.accuracy_m,
        gps_validation_status=pose.status.value,
        gps_validation_warnings=(
            "\n".join(pose.warnings) if pose.warnings else None
        ),
    )


@router.post("", response_model=SessionOut, status_code=201)
async def create_session(
    file: UploadFile,
    fps_target: float | None = Form(default=None),
    gps_latitude: float | None = Form(default=None),
    gps_longitude: float | None = Form(default=None),
    gps_heading_deg: float | None = Form(default=None),
    zone_id: int | None = Form(default=None),
    db: AsyncSession = Depends(get_db),
):
    session_id = uuid.uuid4()
    filename = Path(file.filename or "unknown_file").name
    dest = Path(settings.UPLOAD_DIR) / f"{session_id}_{filename}"

    # Stream chunks instead of `await file.read()` so multi-GB uploads don't
    # have to fit in memory.
    async with aiofiles.open(dest, "wb") as out:
        while chunk := await file.read(_UPLOAD_CHUNK_BYTES):
            await out.write(chunk)

    effective_fps_target = _effective_fps_target(fps_target)
    total_frames = None
    try:
        video_info = get_video_info(str(dest))
        total_frames = _estimate_sampled_total_frames(video_info, effective_fps_target)
    except Exception:
        pass

    pose = _resolve_session_pose(
        explicit_lat=gps_latitude,
        explicit_lon=gps_longitude,
        explicit_heading=gps_heading_deg,
        video_path=dest,
    )

    resolved_zone = await _resolve_zone_id(zone_id, db)
    session_kwargs = dict(
        id=session_id,
        source_filename=filename,
        status="pending",
        fps_target=effective_fps_target,
        frames_processed=0,
        total_frames=total_frames,
        zone_id=resolved_zone,
    )
    _apply_pose_to_session(session_kwargs, pose)
    new_session = Session(**session_kwargs)
    db.add(new_session)
    await db.commit()
    await db.refresh(new_session)

    return new_session


@router.get("/test-files", response_model=List[str])
async def list_test_files():
    upload_dir = Path(settings.UPLOAD_DIR)
    if not upload_dir.exists():
        return []

    test_files = [
        path.name
        for path in upload_dir.iterdir()
        if _is_video_file(path) and not _is_session_generated_upload(path.name)
    ]
    return sorted(test_files)


@router.post("/test-files/{filename}", response_model=SessionOut, status_code=201)
async def create_session_from_test_file(
    filename: str,
    gps_latitude: float | None = Form(default=None),
    gps_longitude: float | None = Form(default=None),
    gps_heading_deg: float | None = Form(default=None),
    zone_id: int | None = Form(default=None),
    db: AsyncSession = Depends(get_db),
):
    source = _resolve_test_upload_file(filename)

    session_id = uuid.uuid4()
    dest = Path(settings.UPLOAD_DIR) / f"{session_id}_{source.name}"
    shutil.copy2(source, dest)

    effective_fps_target = _effective_fps_target(None)
    total_frames = None
    try:
        video_info = get_video_info(str(dest))
        total_frames = _estimate_sampled_total_frames(video_info, effective_fps_target)
    except Exception:
        pass

    pose = _resolve_session_pose(
        explicit_lat=gps_latitude,
        explicit_lon=gps_longitude,
        explicit_heading=gps_heading_deg,
        video_path=dest,
    )

    resolved_zone = await _resolve_zone_id(zone_id, db)
    session_kwargs = dict(
        id=session_id,
        source_filename=source.name,
        status="pending",
        fps_target=effective_fps_target,
        frames_processed=0,
        total_frames=total_frames,
        zone_id=resolved_zone,
    )
    _apply_pose_to_session(session_kwargs, pose)
    new_session = Session(**session_kwargs)
    db.add(new_session)
    await db.commit()
    await db.refresh(new_session)
    return new_session


@router.get("/{session_id}", response_model=SessionOut)
async def get_session(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    session = await db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.post("/{session_id}/process", response_model=SessionOut)
async def process_session_endpoint(
    session_id: uuid.UUID, db: AsyncSession = Depends(get_db)
):
    """Start the ML pipeline on an uploaded session's video."""
    session = await db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    existing_task = _processing_tasks.get(session_id)
    if existing_task is not None and not existing_task.done():
        await db.refresh(session)
        return session

    if session.status in {"processing", "completed"}:
        return session

    session.status = "processing"
    session.error_message = None
    session.ended_at = None
    await db.commit()
    await db.refresh(session)

    _processing_tasks[session_id] = asyncio.create_task(_run_processing_job(session_id))
    await db.refresh(session)
    return session


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    active_task = _processing_tasks.get(session_id)
    if active_task is not None and not active_task.done():
        raise HTTPException(
            status_code=409,
            detail="Session is processing. Cancel it before deleting.",
        )

    session = await db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    detection_ids_stmt = (
        select(Detection.id)
        .join(Frame, Detection.frame_id == Frame.id)
        .where(Frame.session_id == session_id)
    )
    await db.execute(
        delete(TicketCheck).where(TicketCheck.detection_id.in_(detection_ids_stmt))
    )

    # Must run before the session row is deleted: SpotOccupancyEvent.session_id
    # has ondelete=SET NULL, so after deletion we can't identify which events
    # belonged to this session.
    from app.services.occupancy import cleanup_session_occupancy
    await cleanup_session_occupancy(session_id, db)

    source_filename = session.source_filename
    await db.delete(session)
    await db.commit()

    upload_path = Path(settings.UPLOAD_DIR) / f"{session_id}_{source_filename}"
    if upload_path.exists():
        upload_path.unlink()
    session_crops_dir = Path(settings.CROPS_DIR) / str(session_id)
    if session_crops_dir.exists():
        shutil.rmtree(session_crops_dir, ignore_errors=True)

    return Response(status_code=204)


@router.post("/{session_id}/cancel", response_model=SessionOut)
async def cancel_session(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    session = await db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    active_task = _processing_tasks.get(session_id)
    if active_task is None or active_task.done():
        if session.status == "processing":
            session.status = "failed"
            session.error_message = "Cancelled by user"
            session.ended_at = datetime.now(UTC)
            await db.commit()
            await db.refresh(session)
        return session

    active_task.cancel()
    session.status = "failed"
    session.error_message = "Cancelled by user"
    session.ended_at = datetime.now(UTC)
    await db.commit()
    await db.refresh(session)
    return session


@router.get("/{session_id}/detections", response_model=List[DetectionOut])
async def list_detections(
    session_id: uuid.UUID,
    since: str | None = Query(default=None, description="ISO 8601 timestamp; return only detections created after this point"),
    db: AsyncSession = Depends(get_db),
):
    """Return detections for a session.

    Pass ``since`` (ISO 8601) to get only rows created after that timestamp —
    useful for incremental polling after an SSE ``detection_finalized`` event.
    """
    session = await db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    stmt = (
        select(Detection)
        .join(Frame, Detection.frame_id == Frame.id)
        .where(Frame.session_id == session_id)
        .order_by(Detection.created_at.desc())
    )
    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            stmt = stmt.where(Detection.created_at > since_dt)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid 'since' timestamp; expected ISO 8601")

    result = await db.execute(stmt)
    detections = result.scalars().all()
    return detections


@router.get("/{session_id}/events")
async def session_events(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Stream live pipeline events via Server-Sent Events.

    Event types emitted by the backend:
      * ``frame_processed``    – progress tick; data: {frames_processed, total_frames}
      * ``detection_finalized`` – a track was finalised; data: {plate}
      * ``session_completed``  – processing finished; data: {frames_processed, total_frames}
      * ``session_failed``     – processing failed; data: {error}

    The connection is kept alive with comment keepalives every 25 s.
    """
    session = await db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    sid = str(session_id)

    async def _stream() -> AsyncGenerator[str, None]:
        # If already in a terminal state, send one event and close immediately.
        if session.status == "completed":
            yield (
                f"event: session_completed\n"
                f"data: {json.dumps({'frames_processed': session.frames_processed, 'total_frames': session.total_frames})}\n\n"
            )
            return
        if session.status == "failed":
            yield (
                f"event: session_failed\n"
                f"data: {json.dumps({'error': session.error_message})}\n\n"
            )
            return

        q = await subscribe(sid)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=25.0)
                    yield f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
                    if event["type"] in ("session_completed", "session_failed"):
                        break
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            unsubscribe(sid, q)

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
