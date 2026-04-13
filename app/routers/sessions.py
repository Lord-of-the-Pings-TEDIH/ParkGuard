import uuid
from pathlib import Path
from typing import List

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import AsyncSessionLocal, get_db
from app.models.detection import Detection, Frame, Session
from app.pipeline.processor import process_session
from app.pipeline.worker import process_video
from app.schemas.detection import DetectionOut, SessionOut
from app.pipeline.frame_extractor import get_video_info

router = APIRouter()


@router.get("", response_model=List[SessionOut])
async def list_sessions(db: AsyncSession = Depends(get_db)):
    """Return all sessions."""
    result = await db.execute(select(Session))
    sessions = result.scalars().all()
    return sessions


@router.post("", response_model=SessionOut, status_code=201)
async def create_session(file: UploadFile, db: AsyncSession = Depends(get_db)):
    session_id = uuid.uuid4()
    filename = file.filename or "unknown_file"
    dest = Path(settings.UPLOAD_DIR) / f"{session_id}_{filename}"

    async with aiofiles.open(dest, "wb") as out:
        content = await file.read()
        await out.write(content)

    total_frames = None
    try:
        video_info = get_video_info(str(dest))
        total_frames = video_info.get("total_frames")
    except Exception:
        pass

    new_session = Session(
        id=session_id,
        source_filename=filename,
        status="pending",
        frames_processed=0,
        total_frames=total_frames,
    )
    db.add(new_session)
    await db.commit()
    await db.refresh(new_session)

    return new_session


@router.post("/{session_id}/process", response_model=SessionOut, status_code=202)
async def process_session_endpoint(
    session_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Kick off the ML pipeline as a FastAPI BackgroundTask.

    Returns immediately with the session in its current (``pending``) state;
    the worker flips it to ``running`` → ``done``/``failed`` asynchronously.
    """
    session = await db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    video_path = Path(settings.UPLOAD_DIR) / f"{session_id}_{session.source_filename}"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Uploaded video file missing")

    background_tasks.add_task(
        process_video,
        session_id=session_id,
        video_path=str(video_path),
        fps_target=settings.FPS_TARGET,
        db_factory=AsyncSessionLocal,
    )
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
    )
    result = await db.execute(stmt)
    detections = result.scalars().all()
    return detections
