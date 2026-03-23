"""Pydantic schemas for parking-related models."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class ParkingTicketIn(BaseModel):
    plate_text: str
    valid_from: datetime
    valid_until: datetime
    zone_id: Optional[int] = None
    ticket_ref: str
    payment_method: Optional[str] = None


class ParkingTicketOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    plate_text: str
    valid_from: datetime
    valid_until: datetime
    zone_id: Optional[int] = None
    ticket_ref: str
    payment_method: Optional[str] = None
    created_at: datetime
