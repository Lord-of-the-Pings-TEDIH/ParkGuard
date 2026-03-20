from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.parking import ParkingTicket
from app.schemas.parking import ParkingTicketIn, ParkingTicketOut

router = APIRouter()


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
