from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class CameraIn(BaseModel):
    name: str = Field(..., max_length=150)
    stream_url: str = Field(..., max_length=500)
    parking_lot_id: int
    is_active: bool = True


class CameraOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    stream_url: str
    parking_lot_id: int
    is_active: bool
    created_at: datetime
