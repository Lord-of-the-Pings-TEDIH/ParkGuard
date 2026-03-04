from datetime import datetime

from sqlalchemy import String, Boolean, ForeignKey, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(primary_key=True)
    detection_id: Mapped[int] = mapped_column(
        ForeignKey("detections.id"), nullable=False
    )
    alert_type: Mapped[str] = mapped_column(String(50), nullable=False)
    message: Mapped[str | None] = mapped_column(String(500))
    is_resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    detection: Mapped["Detection"] = relationship(back_populates="alerts")  
