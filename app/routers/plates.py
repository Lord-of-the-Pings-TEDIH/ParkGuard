import re
import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.detection import Plate
from app.schemas.detection import PlateOut

router = APIRouter()

_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]")


@router.get("", response_model=List[PlateOut])
async def list_plates(
    q: str | None = Query(default=None, description="Substring of the plate text"),
    county: str | None = Query(default=None, description="County prefix (e.g. CJ, B)"),
    db: AsyncSession = Depends(get_db),
):
    """Return unique plates observed across sessions, optionally filtered.

    The frontend "Registry" sidebar sends both ``q`` (free-text plate search)
    and ``county`` (leading prefix); without these filters the endpoint used
    to dump every row, which made the search box pointless once a few sessions
    had run.  Plates are stored in compact uppercase form (e.g. ``CJ05XYO``),
    so we normalise the inputs the same way before doing the LIKE match —
    keeps the comparison portable across PostgreSQL and the SQLite test DB.
    """
    stmt = select(Plate).order_by(Plate.last_seen_at.desc())

    if q:
        compact_q = _NON_ALNUM_RE.sub("", q.upper())
        if compact_q:
            stmt = stmt.where(Plate.normalized_text.like(f"%{compact_q}%"))

    if county:
        compact_county = _NON_ALNUM_RE.sub("", county.upper())
        if compact_county:
            stmt = stmt.where(Plate.normalized_text.like(f"{compact_county}%"))

    result = await db.execute(stmt)
    plates = result.scalars().all()
    return plates


@router.get("/{plate_id}", response_model=PlateOut)
async def get_plate(plate_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Return a single plate by its UUID."""
    plate = await db.get(Plate, plate_id)
    if plate is None:
        raise HTTPException(status_code=404, detail="Plate not found")
    return plate
