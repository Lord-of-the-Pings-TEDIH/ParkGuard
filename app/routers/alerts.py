"""Alert management endpoints.

Alerts are created automatically when the occupancy-scoring pipeline crosses
FLAG_THRESHOLD for a (spot, plate) pair for the first time.  Operators use
this router to list pending alerts and resolve false positives.
"""

from typing import List
from datetime import datetime

import uuid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.alert import Alert

router = APIRouter()


class AlertOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    detection_id: uuid.UUID | None = None
    alert_type: str
    message: str | None = None
    is_resolved: bool
    created_at: datetime


@router.get("", response_model=List[AlertOut])
async def list_alerts(
    resolved: bool | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Return alerts, optionally filtered by resolution status.

    - ``resolved=false`` → only open alerts (default behaviour for dashboards)
    - ``resolved=true``  → only resolved alerts
    - omitted           → all alerts
    """
    stmt = select(Alert).order_by(Alert.created_at.desc())
    if resolved is not None:
        stmt = stmt.where(Alert.is_resolved == resolved)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.post("/{alert_id}/resolve", response_model=AlertOut)
async def resolve_alert(alert_id: int, db: AsyncSession = Depends(get_db)):
    """Mark an alert as resolved (false positive or handled)."""
    alert = await db.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.is_resolved = True
    await db.commit()
    await db.refresh(alert)
    return alert


@router.delete("/{alert_id}", status_code=204)
async def delete_alert(alert_id: int, db: AsyncSession = Depends(get_db)):
    """Permanently delete an alert."""
    alert = await db.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    await db.delete(alert)
    await db.commit()
