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
from app.pipeline.processor import process_session
from app.schemas.detection import DetectionOut, SessionOut
from app.pipeline.frame_extractor import get_video_info

router = APIRouter()
_plate_reader: PlateReader | None = None
_processing_tasks: dict[uuid.UUID, asyncio.Task[None]] = {}
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov"}
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


async def _run_processing_job(session_id: uuid.UUID) -> None:
    try:
        async with AsyncSessionLocal() as db:
            await process_session(session_id, db, ocr=_recognize_plate)
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


@router.post("", response_model=SessionOut, status_code=201)
async def create_session(
    file: UploadFile,
    fps_target: float | None = Form(default=None),
    db: AsyncSession = Depends(get_db),
):
    session_id = uuid.uuid4()
    filename = Path(file.filename or "unknown_file").name
    dest = Path(settings.UPLOAD_DIR) / f"{session_id}_{filename}"

    async with aiofiles.open(dest, "wb") as out:
        content = await file.read()
        await out.write(content)

    effective_fps_target = _effective_fps_target(fps_target)
    total_frames = None
    try:
        video_info = get_video_info(str(dest))
        total_frames = _estimate_sampled_total_frames(video_info, effective_fps_target)
    except Exception:
        pass

    new_session = Session(
        id=session_id,
        source_filename=filename,
        status="pending",
        fps_target=effective_fps_target,
        frames_processed=0,
        total_frames=total_frames,
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

    new_session = Session(
        id=session_id,
        source_filename=source.name,
        status="pending",
        fps_target=effective_fps_target,
        frames_processed=0,
        total_frames=total_frames,
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
