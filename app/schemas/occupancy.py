from datetime import datetime

from pydantic import BaseModel, ConfigDict


class SuspiciousOccupancyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    spot_id: int
    spot_label: str | None
    assigned_plate: str | None
    plate_text: str
    event_count: int
    distinct_days: int
    accumulated_score: float
    first_seen_at: datetime
    last_seen_at: datetime
    is_flagged: bool
