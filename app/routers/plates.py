import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.detection import Plate
from app.schemas.detection import PlateOut

router = APIRouter()


@router.get("", response_model=List[PlateOut])
async def list_plates(db: AsyncSession = Depends(get_db)):
    """Return all unique plates observed across sessions."""
    result = await db.execute(select(Plate))
    plates = result.scalars().all()
    return plates


@router.get("/{plate_id}", response_model=PlateOut)
async def get_plate(plate_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Return a single plate by its UUID."""
    plate = await db.get(Plate, plate_id)
    if plate is None:
        raise HTTPException(status_code=404, detail="Plate not found")
    return plate
