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
    # Circular concentration of detection hours in [0, 1]:
    # 1 = always same hour (e.g. always 3am), 0 = uniformly spread.
    hour_concentration: float | None = None
    # Circular mean of hour-of-day for all events (0–24).
    typical_hour: float | None = None
    first_seen_at: datetime
    last_seen_at: datetime
    is_flagged: bool
