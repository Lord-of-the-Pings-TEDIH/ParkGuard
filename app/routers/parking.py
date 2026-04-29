from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.parking import ParkingTicket, ParkingZone
from app.models.parking_spot import ParkingSpot
from app.schemas.occupancy import SuspiciousOccupancyOut
from app.schemas.parking import (
    ParkingSpotIn,
    ParkingSpotOut,
    ParkingTicketIn,
    ParkingTicketOut,
    ParkingZoneOut,
)

router = APIRouter()


@router.get("/zones", response_model=List[ParkingZoneOut])
async def list_zones(db: AsyncSession = Depends(get_db)):
    """Return all parking zones (used by the zone selector in the upload form)."""
    result = await db.execute(select(ParkingZone).order_by(ParkingZone.id))
    return result.scalars().all()


@router.get("/tickets", response_model=List[ParkingTicketOut])
async def list_tickets(db: AsyncSession = Depends(get_db)):
    """Return all parking tickets."""
    result = await db.execute(select(ParkingTicket))
    tickets = result.scalars().all()
    return tickets


@router.post("/tickets", response_model=ParkingTicketOut, status_code=201)
async def create_ticket(payload: ParkingTicketIn, db: AsyncSession = Depends(get_db)):
    """Create a new parking ticket."""
    ticket = ParkingTicket(**payload.model_dump())
    db.add(ticket)
    await db.commit()
    await db.refresh(ticket)
    return ticket


@router.get("/tickets/{ticket_id}", response_model=ParkingTicketOut)
async def get_ticket(ticket_id: int, db: AsyncSession = Depends(get_db)):
    """Return a single parking ticket by ID."""
    ticket = await db.get(ParkingTicket, ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


@router.get("/spots", response_model=List[ParkingSpotOut])
async def list_spots(db: AsyncSession = Depends(get_db)):
    """Return all parking spots (used by the Mobile-LPR map view)."""
    result = await db.execute(select(ParkingSpot).order_by(ParkingSpot.id))
    return result.scalars().all()


@router.post("/spots", response_model=ParkingSpotOut, status_code=201)
async def create_spot(payload: ParkingSpotIn, db: AsyncSession = Depends(get_db)):
    """Register a new parking spot with optional GPS centre + assigned plate."""
    spot = ParkingSpot(**payload.model_dump())
    db.add(spot)
    await db.commit()
    await db.refresh(spot)
    return spot


@router.get("/spots/suspicious", response_model=List[SuspiciousOccupancyOut])
async def list_suspicious_occupancies(
    flagged_only: bool = True,
    db: AsyncSession = Depends(get_db),
):
    """Return spots with suspicious foreign-plate occupancy, ranked by score.

    Pass ``flagged_only=false`` to include records below the flag threshold.
    """
    from app.services.occupancy import get_suspicious_records
    return await get_suspicious_records(db, flagged_only=flagged_only)


@router.delete("/spots/suspicious/{record_id}", status_code=204)
async def dismiss_suspicious_occupancy(
    record_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Clear the suspicion score and flag for a record (false-positive dismissal).

    Raw events are preserved for audit. Use when a flagged record is not a
    genuine violation.
    """
    from app.services.occupancy import reset_occupancy_record
    found = await reset_occupancy_record(record_id, db)
    if not found:
        raise HTTPException(status_code=404, detail="Record not found")
    await db.commit()


@router.get("/spots/{spot_id}", response_model=ParkingSpotOut)
async def get_spot(spot_id: int, db: AsyncSession = Depends(get_db)):
    spot = await db.get(ParkingSpot, spot_id)
    if spot is None:
        raise HTTPException(status_code=404, detail="Parking spot not found")
    return spot


@router.put("/spots/{spot_id}", response_model=ParkingSpotOut)
async def update_spot(
    spot_id: int,
    payload: ParkingSpotIn,
    db: AsyncSession = Depends(get_db),
):
    spot = await db.get(ParkingSpot, spot_id)
    if spot is None:
        raise HTTPException(status_code=404, detail="Parking spot not found")
    for key, value in payload.model_dump().items():
        setattr(spot, key, value)
    await db.commit()
    await db.refresh(spot)
    return spot


@router.delete("/spots/{spot_id}", status_code=204)
async def delete_spot(spot_id: int, db: AsyncSession = Depends(get_db)):
    spot = await db.get(ParkingSpot, spot_id)
    if spot is None:
        raise HTTPException(status_code=404, detail="Parking spot not found")
    await db.delete(spot)
    await db.commit()
