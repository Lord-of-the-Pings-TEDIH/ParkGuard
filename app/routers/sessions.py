import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.models.detection import Session
from app.schemas.detection import SessionOut

router = APIRouter()


@router.post("/sessions", response_model=SessionOut)
async def create_session(file: UploadFile, db: AsyncSession = Depends(get_db)):
    session_id = uuid.uuid4()
    filename = file.filename or "unknown_file"
    dest = Path(settings.UPLOAD_DIR) / f"{session_id}_{filename}"

    async with aiofiles.open(dest, "wb") as out:
        content = await file.read()
        await out.write(content)

    new_session = Session(
        id=session_id,
        source_filename=filename,
        status="pending",
        frames_processed=0,
    )
    db.add(new_session)
    await db.commit()
    await db.refresh(new_session)

    return new_session
