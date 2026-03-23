"""Pydantic schemas for API request/response models."""

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    source_filename: str
    status: str
    fps_target: Optional[float]
    frames_processed: int
    created_at: datetime
    ended_at: Optional[datetime]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


class DetectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    frame_id: uuid.UUID
    ocr_normalized_text: Optional[str]
    ocr_raw_text: Optional[str]
    is_valid_ro_plate: bool
    detection_confidence: float
    crop_image_url: Optional[str]
    ticket_status: Optional[str]
    ticket_expires_at: Optional[datetime]
    created_at: datetime


# ---------------------------------------------------------------------------
# Plate
# ---------------------------------------------------------------------------


class PlateOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    normalized_text: str
    county_code: Optional[str]
    county_name: Optional[str]
    plate_type: Optional[str]
    first_seen_at: datetime
    last_seen_at: datetime
    seen_count: int


# ---------------------------------------------------------------------------
# Parking Ticket (input)
# ---------------------------------------------------------------------------


class ParkingTicketIn(BaseModel):
    plate_text: str
    valid_from: datetime
    valid_until: datetime
    zone_id: Optional[int]
    ticket_ref: str
    payment_method: Optional[str] = None
