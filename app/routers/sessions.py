import asyncio
import logging
import uuid
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import aiofiles
import numpy as np
from fastapi import APIRouter, Depends, Form, HTTPException, Response, UploadFile
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import AsyncSessionLocal, get_db
from app.models.detection import Detection, Frame, Session
from app.models.parking import TicketCheck
from app.pipeline.ocr import PlateReader
from app.pipeline.processor import build_mobile_lpr_context, process_session
from app.schemas.detection import DetectionOut, SessionOut
from app.pipeline.frame_extractor import get_video_info
from app.pipeline.video_metadata import extract_video_gps

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
            await process_session(
                session_id,
                db,
                ocr=_recognize_plate,
                mobile_lpr=mobile_lpr,
            )
    except asyncio.CancelledError:
        async with AsyncSessionLocal() as db:
            session = await db.get(Session, session_id)
            if session is not None:
                session.status = "failed"
                session.error_message = "Cancelled by user"
                session.ended_at = datetime.now(UTC)
                await db.commit()
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


def _default_gps_pose() -> tuple[float | None, float | None, float | None]:
    """Return the configured default Mobile-LPR pose, if all three are set.

    When MOBILE_LPR_DEFAULT_LATITUDE / LONGITUDE / HEADING_DEG are all
    configured in .env, every session created without an explicit pose falls
    back to these defaults so the GPS projection runs and per-car coordinates
    appear in the UI.  Returns ``(None, None, None)`` if any default is unset.
    """
    lat = settings.MOBILE_LPR_DEFAULT_LATITUDE
    lon = settings.MOBILE_LPR_DEFAULT_LONGITUDE
    heading = settings.MOBILE_LPR_DEFAULT_HEADING_DEG
    if lat is None or lon is None or heading is None:
        return (None, None, None)
    return (float(lat), float(lon), float(heading) % 360.0)


def _validate_explicit_gps_pose(
    lat: float | None, lon: float | None, heading: float | None
) -> tuple[float | None, float | None, float | None]:
    """Validate user-provided GPS pose form fields.

    Mobile-LPR mode is all-or-nothing: a partial pose would silently disable
    spot validation in the pipeline and the user would never know why.
    Returns ``(None, None, None)`` when nothing was supplied (so the caller
    can decide whether to fall back to video metadata or config defaults).
    """
    provided = [v for v in (lat, lon, heading) if v is not None]
    if not provided:
        return (None, None, None)
    if len(provided) != 3:
        raise HTTPException(
            status_code=400,
            detail="Mobile-LPR requires gps_latitude, gps_longitude and gps_heading_deg together.",
        )
    if not (-90.0 <= float(lat) <= 90.0):  # type: ignore[arg-type]
        raise HTTPException(status_code=400, detail="gps_latitude out of range")
    if not (-180.0 <= float(lon) <= 180.0):  # type: ignore[arg-type]
        raise HTTPException(status_code=400, detail="gps_longitude out of range")
    return (float(lat), float(lon), float(heading) % 360.0)  # type: ignore[arg-type]


def _resolve_session_pose(
    *,
    explicit_lat: float | None,
    explicit_lon: float | None,
    explicit_heading: float | None,
    video_path: Path,
) -> tuple[float | None, float | None, float | None]:
    """Pick the best Mobile-LPR pose for a new session.

    Precedence (first usable wins):
      1. The explicit lat/lon/heading triple sent on the form.
      2. The GPS fix embedded in the uploaded video's container metadata
         (e.g. ``com.apple.quicktime.location.ISO6709`` from iOS .MOV files),
         combined with the configured default heading (or 0 = North).
      3. The configured default pose from ``settings.MOBILE_LPR_DEFAULT_*``.
    """
    explicit = _validate_explicit_gps_pose(
        explicit_lat, explicit_lon, explicit_heading
    )
    if explicit[0] is not None:
        return explicit

    metadata = extract_video_gps(video_path)
    if metadata is not None:
        meta_lat, meta_lon = metadata
        # The container only stores lat/lon — heading falls back to the
        # configured default, or 0° (North) when no default is set.
        heading_default = settings.MOBILE_LPR_DEFAULT_HEADING_DEG
        heading = float(heading_default) if heading_default is not None else 0.0
        return (meta_lat, meta_lon, heading % 360.0)

    return _default_gps_pose()


@router.post("", response_model=SessionOut, status_code=201)
async def create_session(
    file: UploadFile,
    fps_target: float | None = Form(default=None),
    gps_latitude: float | None = Form(default=None),
    gps_longitude: float | None = Form(default=None),
    gps_heading_deg: float | None = Form(default=None),
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

    lat, lon, heading = _resolve_session_pose(
        explicit_lat=gps_latitude,
        explicit_lon=gps_longitude,
        explicit_heading=gps_heading_deg,
        video_path=dest,
    )

    new_session = Session(
        id=session_id,
        source_filename=filename,
        status="pending",
        fps_target=effective_fps_target,
        frames_processed=0,
        total_frames=total_frames,
        gps_latitude=lat,
        gps_longitude=lon,
        gps_heading_deg=heading,
    )
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

    lat, lon, heading = _resolve_session_pose(
        explicit_lat=gps_latitude,
        explicit_lon=gps_longitude,
        explicit_heading=gps_heading_deg,
        video_path=dest,
    )

    new_session = Session(
        id=session_id,
        source_filename=source.name,
        status="pending",
        fps_target=effective_fps_target,
        frames_processed=0,
        total_frames=total_frames,
        gps_latitude=lat,
        gps_longitude=lon,
        gps_heading_deg=heading,
    )
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
async def list_detections(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Return all detections for a session, with a computed crop_image_url."""
    # Verify session exists
    session = await db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Join detections through frames belonging to this session
    stmt = (
        select(Detection)
        .join(Frame, Detection.frame_id == Frame.id)
        .where(Frame.session_id == session_id)
        .order_by(Detection.created_at.desc())
    )
    result = await db.execute(stmt)
    detections = result.scalars().all()
    return detections
