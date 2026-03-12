import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, UploadFile

from app.core.config import settings

router = APIRouter()


@router.post("/sessions")
async def create_session(file: UploadFile):
    session_id = str(uuid.uuid4())
    dest = Path(settings.UPLOAD_DIR) / f"{session_id}_{file.filename}"

    async with aiofiles.open(dest, "wb") as out:
        content = await file.read()
        await out.write(content)

    return {
        "session_id": session_id,
        "filename": file.filename,
        "size_bytes": dest.stat().st_size,
    }
