from datetime import datetime, time, timedelta, timezone

from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

UTC = timezone.utc

from app.models.parking import ParkingTicket, ParkingSubscription, ParkingZone, TicketCheck
from app.models.detection import Plate


async def check_active_ticket(
    plate_text: str,
    zone_id: int,
    checked_at: datetime,
    db: AsyncSession,
) -> tuple[int, datetime] | None:
    """Return (ticket.id, ticket.valid_until) for the latest active ticket, or None."""
    stmt = (
        select(ParkingTicket)
        .where(
            ParkingTicket.plate_text == plate_text,
            ParkingTicket.valid_from <= checked_at,
            ParkingTicket.valid_until >= checked_at,
            or_(
                ParkingTicket.zone_id == None,  # noqa: E711 – SQLAlchemy IS NULL
                ParkingTicket.zone_id == zone_id,
            ),
        )
        .order_by(ParkingTicket.valid_until.desc())
        .limit(1)
    )

    result = await db.execute(stmt)
    ticket = result.scalars().first()

    if ticket is None:
        return None

    return (ticket.id, ticket.valid_until)


async def _check_active_subscription(
    plate_text: str,
    zone_id: int,
    checked_at: datetime,
    db: AsyncSession,
) -> tuple[int, datetime] | None:
    """Return (sub.id, end-of-day datetime) for the latest active subscription, or None."""
    check_date = checked_at.date()

    stmt = (
        select(ParkingSubscription)
        .where(
            ParkingSubscription.plate_text == plate_text,
            ParkingSubscription.is_active == True,  # noqa: E712
            ParkingSubscription.start_date <= check_date,
            ParkingSubscription.end_date >= check_date,
            or_(
                ParkingSubscription.zone_id == None,  # noqa: E711 – SQLAlchemy IS NULL
                ParkingSubscription.zone_id == zone_id,
            ),
        )
        .order_by(ParkingSubscription.end_date.desc())
        .limit(1)
    )

    result = await db.execute(stmt)
    sub = result.scalars().first()

    if sub is None:
        return None

    expires_at = datetime.combine(sub.end_date, time.max, tzinfo=UTC)
    return (sub.id, expires_at)


async def _check_grace_period(
    plate_text: str,
    zone_id: int | None,
    checked_at: datetime,
    db: AsyncSession,
    grace_minutes: int | None = None,
) -> tuple[int, datetime] | None:
    """Return (ticket.id, ticket.valid_until) if the plate is within the grace window."""
    # Resolve grace window: explicit arg > zone row > global default
    if grace_minutes is None:
        if zone_id is not None:
            zone_result = await db.execute(
                select(ParkingZone.grace_period_min).where(ParkingZone.id == zone_id)
            )
            grace_minutes = zone_result.scalar_one_or_none() or 10
        else:
            grace_minutes = 10

    grace_window = timedelta(minutes=grace_minutes)
    grace_start = checked_at - grace_window

    stmt = (
        select(ParkingTicket)
        .where(
            ParkingTicket.plate_text == plate_text,
            ParkingTicket.valid_until < checked_at,
            ParkingTicket.valid_until >= grace_start,
            or_(
                ParkingTicket.zone_id == None,  # noqa: E711 – SQLAlchemy IS NULL
                ParkingTicket.zone_id == zone_id,
            ),
        )
        .order_by(ParkingTicket.valid_until.desc())
        .limit(1)
    )

    result = await db.execute(stmt)
    ticket = result.scalars().first()

    if ticket is None:
        return None

    return (ticket.id, ticket.valid_until)


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

async def lookup_ticket(
    plate_text: str,
    zone_id: int | None,
    checked_at: datetime,
    db: AsyncSession,
    detection_id: int | None = None,
) -> tuple[str, datetime | None, int | None, int | None]:
    """Check whether *plate_text* has valid parking at *checked_at*.

    Returns a 4-tuple: (status, expires_at, ticket_id, sub_id)

    Status values
    -------------
    'active'       – valid ticket covering this moment
    'subscription' – active subscription covering this moment
    'grace'        – ticket recently expired, still within grace window
    'none'         – plate is known but has no valid coverage
    'unknown'      – plate not found in the vehicles table at all

    A ``TicketCheck`` audit row is persisted whenever *detection_id* is
    provided (i.e. in production).  The parameter is optional so that
    unit tests that don't have a real ``detections`` row can still call
    this function without FK violations.
    """
    # Step 1 – active ticket
    ticket_id: int | None = None
    sub_id: int | None = None
    expires_at: datetime | None = None
    status: str

    ticket_match = await check_active_ticket(plate_text, zone_id, checked_at, db)
    if ticket_match is not None:
        ticket_id, expires_at = ticket_match
        status = "active"
    else:
        # Step 2 – active subscription
        sub_match = await _check_active_subscription(plate_text, zone_id, checked_at, db)
        if sub_match is not None:
            sub_id, expires_at = sub_match
            status = "subscription"
        else:
            # Step 3 – grace period
            grace_match = await _check_grace_period(plate_text, zone_id, checked_at, db)
            if grace_match is not None:
                ticket_id, expires_at = grace_match
                status = "grace"
            else:
                # No coverage – distinguish known vs unknown plate
                plate_exists = await db.execute(
                    select(Plate.id).where(Plate.normalized_text == plate_text).limit(1)
                )
                if plate_exists.scalar_one_or_none() is not None:
                    status = "none"
                else:
                    status = "unknown"

    # --- Audit row ---------------------------------------------------
    if detection_id is not None:
        audit = TicketCheck(
            detection_id=detection_id,
            plate_text=plate_text,
            checked_at=checked_at,
            result_status=status,
            matched_ticket_id=ticket_id,
            matched_sub_id=sub_id,
            expires_at=expires_at,
        )
        db.add(audit)
        # Flush so the row is visible within the same transaction;
        # the caller is responsible for commit.
        await db.flush()
    # -----------------------------------------------------------------

    return (status, expires_at, ticket_id, sub_id)
