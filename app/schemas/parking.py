"""Pydantic schemas for parking-related models."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


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


class ParkingSpotIn(BaseModel):
    spot_label: str = Field(..., max_length=20)
    parking_lot_id: int
    latitude: Optional[float] = Field(default=None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(default=None, ge=-180.0, le=180.0)
    assigned_plate: Optional[str] = Field(default=None, max_length=20)
    is_occupied: bool = False


class ParkingSpotOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    spot_label: str
    parking_lot_id: int
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    assigned_plate: Optional[str] = None
    is_occupied: bool
    updated_at: datetime
