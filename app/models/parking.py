from datetime import date, datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    func,
)

from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

class ParkingZone(Base):
    __tablename__ = "parking_zones"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    grace_period_min: Mapped[int] = mapped_column(
        Integer, default=10, server_default="10"
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default="true"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    tickets: Mapped[list["ParkingTicket"]] = relationship(back_populates="zone")
    subscriptions: Mapped[list["ParkingSubscription"]] = relationship(
        back_populates="zone"
    )


class ParkingTicket(Base):
    __tablename__ = "parking_tickets"

    id: Mapped[int] = mapped_column(primary_key=True)
    plate_text: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    zone_id: Mapped[int | None] = mapped_column(
        ForeignKey("parking_zones.id"), nullable=True, index=True
    )
    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    valid_until: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    ticket_ref: Mapped[str] = mapped_column(String(100), nullable=False)
    payment_method: Mapped[str | None] = mapped_column(String(50))
    extra_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    zone: Mapped["ParkingZone | None"] = relationship(back_populates="tickets")
    ticket_checks: Mapped[list["TicketCheck"]] = relationship(
        back_populates="matched_ticket",
        foreign_keys="TicketCheck.matched_ticket_id",
    )


class ParkingSubscription(Base):
    __tablename__ = "parking_subscriptions"

    id: Mapped[int] = mapped_column(primary_key=True)
    plate_text: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    zone_id: Mapped[int | None] = mapped_column(
        ForeignKey("parking_zones.id"), nullable=True, index=True
    )
    subscription_type: Mapped[str] = mapped_column(String(50), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    holder_name: Mapped[str | None] = mapped_column(String(200))
    ref_number: Mapped[str | None] = mapped_column(String(100))
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default="true"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    zone: Mapped["ParkingZone | None"] = relationship(back_populates="subscriptions")
    ticket_checks: Mapped[list["TicketCheck"]] = relationship(
        back_populates="matched_sub",
        foreign_keys="TicketCheck.matched_sub_id",
    )


class TicketCheck(Base):
    __tablename__ = "ticket_checks"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )
    detection_id: Mapped[int] = mapped_column(
        ForeignKey("detections.id"), nullable=False, index=True
    )
    plate_text: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    checked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    result_status: Mapped[str] = mapped_column(String(30), nullable=False)
    matched_ticket_id: Mapped[int | None] = mapped_column(
        ForeignKey("parking_tickets.id"), nullable=True, index=True
    )
    matched_sub_id: Mapped[int | None] = mapped_column(
        ForeignKey("parking_subscriptions.id"), nullable=True, index=True
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    detection: Mapped["Detection"] = relationship()
    matched_ticket: Mapped["ParkingTicket | None"] = relationship(
        back_populates="ticket_checks",
        foreign_keys=[matched_ticket_id],
    )
    matched_sub: Mapped["ParkingSubscription | None"] = relationship(
        back_populates="ticket_checks",
        foreign_keys=[matched_sub_id],
    )
