"""Tests that lookup_ticket() writes a TicketCheck audit row."""

from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy import BigInteger
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.postgresql import JSONB, UUID

# SQLite shims for PostgreSQL-specific column types
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
from app.models.detection import Session, Frame, Detection, Plate
from app.models.parking import ParkingTicket, TicketCheck
from app.services.ticket_lookup import lookup_ticket

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestingSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
)

UTC = timezone.utc


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


@pytest.mark.asyncio
async def test_active_ticket_creates_audit_row(db: AsyncSession):
    """lookup_ticket() with detection_id writes one TicketCheck row for an active ticket."""
    now = datetime.now(UTC)

    # -- Seed: Plate + active ParkingTicket ---------------------------------
    plate = Plate(normalized_text="B999ZZZ")
    db.add(plate)
    await db.flush()

    ticket = ParkingTicket(
        plate_text="B999ZZZ",
        valid_from=now - timedelta(hours=1),
        valid_until=now + timedelta(hours=1),
        ticket_ref="TKT-AUDIT-1",
    )
    db.add(ticket)
    await db.flush()

    # -- Seed: Session → Frame → Detection (to satisfy FK) ------------------
    session_obj = Session(source_filename="audit_test.mp4")
    db.add(session_obj)
    await db.flush()

    frame = Frame(session_id=session_obj.id, frame_index=0)
    db.add(frame)
    await db.flush()

    detection = Detection(
        frame_id=frame.id,
        bbox_x=0, bbox_y=0, bbox_w=100, bbox_h=50,
        detection_confidence=0.95,
    )
    db.add(detection)
    await db.flush()

    # -- Act ----------------------------------------------------------------
    status, expires_at, ticket_id, sub_id = await lookup_ticket(
        plate_text="B999ZZZ",
        zone_id=None,
        checked_at=now,
        db=db,
        detection_id=detection.id,
    )

    # -- Assert: lookup return values ---------------------------------------
    assert status == "active"
    assert ticket_id == ticket.id

    # -- Assert: exactly one audit row in ticket_checks ---------------------
    rows = (await db.execute(select(TicketCheck))).scalars().all()
    assert len(rows) == 1

    audit = rows[0]
    assert audit.result_status == "active"
    assert audit.matched_ticket_id == ticket.id

    # Normalise tz-awareness (SQLite may strip tzinfo)
    audit_expires = audit.expires_at
    if audit_expires is not None and audit_expires.tzinfo is None:
        audit_expires = audit_expires.replace(tzinfo=UTC)

    ticket_valid_until = ticket.valid_until
    if ticket_valid_until is not None and ticket_valid_until.tzinfo is None:
        ticket_valid_until = ticket_valid_until.replace(tzinfo=UTC)

    assert audit_expires == ticket_valid_until
