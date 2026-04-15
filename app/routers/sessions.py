import uuid
from pathlib import Path
from typing import List

import aiofiles
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.models.detection import Detection, Frame, Session
from app.pipeline.ocr import PlateReader
from app.pipeline.processor import process_session
from app.schemas.detection import DetectionOut, SessionOut
from app.pipeline.frame_extractor import get_video_info

router = APIRouter()
_plate_reader: PlateReader | None = None


def _get_plate_reader() -> PlateReader:
    global _plate_reader
    if _plate_reader is None:
        _plate_reader = PlateReader(
            gpu=False,
            use_angle_cls=settings.OCR_USE_ANGLE_CLS,
            lang="en",
        )
    return _plate_reader


def _recognize_plate(crop: np.ndarray) -> tuple[str | None, float]:
    text, confidence = _get_plate_reader().read_plate(crop)
    return (text or None, confidence)


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


@router.post("/{session_id}/process", response_model=SessionOut)
async def process_session_endpoint(
    session_id: uuid.UUID, db: AsyncSession = Depends(get_db)
):
    """Run the ML pipeline on an uploaded session's video."""
    session = await db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    await process_session(session_id, db, ocr=_recognize_plate)
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
    )
    result = await db.execute(stmt)
    detections = result.scalars().all()
    return detections
