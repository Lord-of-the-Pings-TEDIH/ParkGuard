"""Integration test: after processing detections for 3+ known plates,
ticket_checks contains one audit row per detection with correct statuses.

Mirrors the manual verification:
    SELECT tc.plate_text, tc.result_status, tc.checked_at
      FROM ticket_checks tc
     ORDER BY tc.checked_at DESC LIMIT 10;
"""

from datetime import date, datetime, timedelta, timezone
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy import BigInteger, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.postgresql import JSONB, UUID

# ---------------------------------------------------------------------------
# SQLite shims for PostgreSQL-specific column types
# ---------------------------------------------------------------------------
@compiles(JSONB, "sqlite")
def compile_jsonb(type_, compiler, **kw):
    return "TEXT"

@compiles(UUID, "sqlite")
def compile_uuid(type_, compiler, **kw):
    return "CHAR(32)"

@compiles(BigInteger, "sqlite")
def compile_bigint(type_, compiler, **kw):
    return "INTEGER"

from app.models.base import Base
from app.models.detection import Detection, Frame, Plate, Session
from app.models.parking import (
    ParkingSubscription,
    ParkingTicket,
    ParkingZone,
    TicketCheck,
)
from app.services.ticket_lookup import lookup_ticket

# ---------------------------------------------------------------------------
# In-memory SQLite engine
# ---------------------------------------------------------------------------
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestingSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
)

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture(scope="module", autouse=True)
async def setup_database():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db() -> AsyncGenerator[AsyncSession, None]:
    async with TestingSessionLocal() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def seed_parking(db: AsyncSession):
    """Reproduce the relevant slice of seed.py so plate lookups behave
    the same as they would against the real database."""
    now = datetime.now(UTC)
    today = date.today()

    zone = ParkingZone(
        name="Downtown Zone", code="DT-01", grace_period_min=15, is_active=True
    )
    db.add(zone)
    await db.flush()

    # B11AAA — active ticket (valid now)
    db.add(
        ParkingTicket(
            plate_text="B11AAA",
            zone_id=zone.id,
            valid_from=now - timedelta(hours=1),
            valid_until=now + timedelta(hours=1),
            ticket_ref="TKT-001",
            payment_method="SMS",
        )
    )

    # B77GGG — active subscription (no ticket)
    db.add(
        ParkingSubscription(
            plate_text="B77GGG",
            zone_id=zone.id,
            subscription_type="MONTHLY",
            start_date=today - timedelta(days=5),
            end_date=today + timedelta(days=25),
            holder_name="John Doe",
            ref_number="SUB-001",
            is_active=True,
        )
    )

    # B44DDD — expired ticket (expired 2 h ago, outside 15-min grace)
    db.add(
        ParkingTicket(
            plate_text="B44DDD",
            zone_id=zone.id,
            valid_from=now - timedelta(days=1),
            valid_until=now - timedelta(hours=2),
            ticket_ref="TKT-004",
            payment_method="SMS",
        )
    )

    # B33CCC — grace-period ticket (expired 5 min ago, within 15-min grace)
    db.add(
        ParkingTicket(
            plate_text="B33CCC",
            zone_id=zone.id,
            valid_from=now - timedelta(hours=2),
            valid_until=now - timedelta(minutes=5),
            ticket_ref="TKT-003",
            payment_method="CASH",
        )
    )

    # Create Plate rows so lookup can distinguish "none" from "unknown"
    for text in ("B11AAA", "B77GGG", "B44DDD", "B33CCC"):
        db.add(Plate(normalized_text=text))

    await db.flush()
    return zone


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def _create_detection(
    db: AsyncSession, session_obj: Session, frame_index: int
) -> Detection:
    """Insert a Frame + Detection and return the Detection."""
    frame = Frame(session_id=session_obj.id, frame_index=frame_index)
    db.add(frame)
    await db.flush()

    detection = Detection(
        frame_id=frame.id,
        bbox_x=0,
        bbox_y=0,
        bbox_w=100,
        bbox_h=50,
        detection_confidence=0.92,
    )
    db.add(detection)
    await db.flush()
    return detection


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pipeline_audit_rows(db: AsyncSession, seed_parking):
    """Process detections for 4 known plates.  Verify one audit row per
    detection and that result_status matches the expected seed-data values."""
    zone = seed_parking
    now = datetime.now(UTC)

    session_obj = Session(source_filename="integration_test.mp4")
    db.add(session_obj)
    await db.flush()

    # plate_text -> expected status
    expected = {
        "B11AAA": "active",
        "B77GGG": "subscription",
        "B44DDD": "none",        # expired well beyond grace
        "B33CCC": "grace",       # expired 5 min ago, grace is 15
    }

    detection_ids: dict[str, int] = {}

    for idx, (plate_text, _) in enumerate(expected.items()):
        det = await _create_detection(db, session_obj, frame_index=idx)
        detection_ids[plate_text] = det.id

        await lookup_ticket(
            plate_text=plate_text,
            zone_id=zone.id,
            checked_at=now,
            db=db,
            detection_id=det.id,
        )

    # -- Query ticket_checks (mirrors the manual SQL) -----------------------
    stmt = (
        select(
            TicketCheck.plate_text,
            TicketCheck.result_status,
            TicketCheck.checked_at,
            TicketCheck.matched_ticket_id,
            TicketCheck.matched_sub_id,
            TicketCheck.expires_at,
        )
        .order_by(TicketCheck.checked_at.desc())
        .limit(10)
    )
    rows = (await db.execute(stmt)).all()

    # -- Assertions ---------------------------------------------------------
    # One row per detection (4 detections -> 4 rows)
    assert len(rows) == len(expected)

    # Build a lookup: plate_text -> row
    audit_by_plate = {row.plate_text: row for row in rows}

    for plate_text, expected_status in expected.items():
        assert plate_text in audit_by_plate, (
            f"Missing audit row for {plate_text}"
        )
        row = audit_by_plate[plate_text]
        assert row.result_status == expected_status, (
            f"{plate_text}: expected status '{expected_status}', "
            f"got '{row.result_status}'"
        )

    # --- Deeper field checks -----------------------------------------------

    # B11AAA (active) — matched_ticket_id set, expires_at in the future
    active_row = audit_by_plate["B11AAA"]
    assert active_row.matched_ticket_id is not None
    assert active_row.matched_sub_id is None
    assert active_row.expires_at is not None

    # B77GGG (subscription) — matched_sub_id set, no ticket
    sub_row = audit_by_plate["B77GGG"]
    assert sub_row.matched_ticket_id is None
    assert sub_row.matched_sub_id is not None
    assert sub_row.expires_at is not None

    # B44DDD (none) — no ticket or sub matched
    none_row = audit_by_plate["B44DDD"]
    assert none_row.matched_ticket_id is None
    assert none_row.matched_sub_id is None
    assert none_row.expires_at is None

    # B33CCC (grace) — matched_ticket_id set, expires_at in the past
    grace_row = audit_by_plate["B33CCC"]
    assert grace_row.matched_ticket_id is not None
    assert grace_row.matched_sub_id is None
    assert grace_row.expires_at is not None
