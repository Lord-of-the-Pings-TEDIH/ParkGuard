"""Pydantic schemas for parking-related models."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.pipeline.plate_validator import compact_plate, is_invalid_plate, normalize_plate


class ParkingZoneOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    code: str
    grace_period_min: int
    is_active: bool
    created_at: datetime


class ParkingTicketIn(BaseModel):
    plate_text: str
    valid_from: datetime
    valid_until: datetime
    zone_id: Optional[int] = None
    ticket_ref: str
    payment_method: Optional[str] = None

    @field_validator("plate_text", mode="before")
    @classmethod
    def _validate_plate(cls, v: str) -> str:
        normalized = normalize_plate(str(v).strip())
        if is_invalid_plate(normalized):
            raise ValueError(f"plate_text is not a valid Romanian plate: {normalized}")
        return compact_plate(normalized)


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


def _validate_plate_list(plates: list[str] | None) -> list[str] | None:
    """Normalize and deduplicate a list of allowed plate strings."""
    if not plates:
        return None
    result = []
    for raw in plates:
        normalized = normalize_plate(str(raw).strip())
        if is_invalid_plate(normalized):
            raise ValueError(f"allowed_plates entry is not a valid Romanian plate: {normalized}")
        compact = compact_plate(normalized)
        if compact and compact not in result:
            result.append(compact)
    return result or None


class ParkingSpotIn(BaseModel):
    spot_label: str = Field(..., max_length=20)
    parking_lot_id: int
    spot_sequence: Optional[int] = Field(default=None, ge=1)
    latitude: Optional[float] = Field(default=None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(default=None, ge=-180.0, le=180.0)
    assigned_plate: Optional[str] = Field(default=None, max_length=20)
    allowed_plates: Optional[List[str]] = None
    is_occupied: bool = False

    @field_validator("assigned_plate", mode="before")
    @classmethod
    def _normalize_plate(cls, v: str | None) -> str | None:
        if v is None:
            return None
        normalized = normalize_plate(str(v).strip())
        if is_invalid_plate(normalized):
            raise ValueError(f"assigned_plate is not a valid Romanian plate: {normalized}")
        return compact_plate(normalized)

    @field_validator("allowed_plates", mode="before")
    @classmethod
    def _normalize_allowed_plates(cls, v: list | None) -> list[str] | None:
        if v is None:
            return None
        return _validate_plate_list([str(p) for p in v])


class ParkingSpotOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    spot_label: str
    parking_lot_id: int
    spot_sequence: Optional[int] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    assigned_plate: Optional[str] = None
    allowed_plates: Optional[List[str]] = None
    is_occupied: bool
    updated_at: datetime
