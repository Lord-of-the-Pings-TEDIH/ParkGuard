"""Pydantic schemas for parking-related models."""

import re
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]")


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
    spot_sequence: Optional[int] = Field(default=None, ge=1)
    latitude: Optional[float] = Field(default=None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(default=None, ge=-180.0, le=180.0)
    assigned_plate: Optional[str] = Field(default=None, max_length=20)
    is_occupied: bool = False

    @field_validator("assigned_plate", mode="before")
    @classmethod
    def _normalize_plate(cls, v: str | None) -> str | None:
        if v is None:
            return None
        compact = _NON_ALNUM_RE.sub("", v.strip().upper())
        if not compact:
            raise ValueError("assigned_plate must contain at least one alphanumeric character")
        return compact


class ParkingSpotOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    spot_label: str
    parking_lot_id: int
    spot_sequence: Optional[int] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    assigned_plate: Optional[str] = None
    is_occupied: bool
    updated_at: datetime
