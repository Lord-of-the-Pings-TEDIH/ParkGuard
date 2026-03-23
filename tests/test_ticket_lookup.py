import uuid
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.postgresql import JSONB, UUID

@compiles(JSONB, "sqlite")
def compile_jsonb(type_, compiler, **kw):
    return "TEXT"

@compiles(UUID, "sqlite")
def compile_uuid(type_, compiler, **kw):
    return "CHAR(32)"
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.base import Base
from app.models.parking import ParkingTicket, ParkingSubscription, ParkingZone
from app.models.detection import Plate
from app.services.ticket_lookup import lookup_ticket

# In-memory SQLite for testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestingSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
)

UTC = timezone.utc

@pytest_asyncio.fixture(scope="module", autouse=True)
async def setup_database():
    """Create tables before tests run and drop them after."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest_asyncio.fixture
async def db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a database session, rollback after each test."""
    async with TestingSessionLocal() as session:
        yield session
        await session.rollback()

@pytest_asyncio.fixture
async def seed_data(db: AsyncSession):
    """Insert base data for tests."""
    # Create a plate
    test_plate = Plate(normalized_text="B123ABC")
    db.add(test_plate)
    await db.flush()
    
    # Create an active ticket (valid now)
    now = datetime.now(UTC)
    active_ticket = ParkingTicket(
        plate_text="B123ABC",
        valid_from=now - timedelta(hours=1),
        valid_until=now + timedelta(hours=1),
        ticket_ref="TKT-ACTIVE"
    )
    db.add(active_ticket)
    
    # Create an active subscription (valid today)
    active_sub = ParkingSubscription(
        plate_text="B123ABC",
        subscription_type="monthly",
        start_date=(now - timedelta(days=5)).date(),
        end_date=(now + timedelta(days=5)).date(),
        is_active=True
    )
    db.add(active_sub)
    
    # Create an expired ticket in the grace period (expired 5 mins ago, grace is 10)
    grace_ticket = ParkingTicket(
        plate_text="B123ABC",
        valid_from=now - timedelta(hours=2),
        valid_until=now - timedelta(minutes=5),
        ticket_ref="TKT-GRACE"
    )
    db.add(grace_ticket)

    # Empty plate for 'none' test (has no tickets/subs)
    empty_plate = Plate(normalized_text="CJ99XYZ")
    db.add(empty_plate)
    
    await db.flush()

@pytest.mark.asyncio
async def test_active_ticket(db: AsyncSession, seed_data):
    """Test 1: Returns 'active' when a valid ticket exists."""
    now = datetime.now(UTC)
    status, expires_at, ticket_id, sub_id = await lookup_ticket("B123ABC", None, now, db)
    
    assert status == "active"
    assert ticket_id is not None
    assert sub_id is None
    
    if expires_at and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    assert expires_at > now

@pytest.mark.asyncio
async def test_active_subscription(db: AsyncSession, seed_data):
    """Test 2: Returns 'subscription' when no active ticket but valid sub exists."""
    # Check 1.5 hours from now (ticket expired, but subscription valid)
    future = datetime.now(UTC) + timedelta(hours=1, minutes=30)
    status, expires_at, ticket_id, sub_id = await lookup_ticket("B123ABC", None, future, db)
    
    assert status == "subscription"
    assert ticket_id is None
    assert sub_id is not None
    assert expires_at.date() == (datetime.now(UTC) + timedelta(days=5)).date()

@pytest.mark.asyncio
async def test_grace_period(db: AsyncSession, seed_data):
    """Test 3: Returns 'grace' when ticket expired recently within grace window."""
    now = datetime.now(UTC)
    from sqlalchemy import select
    # Delete the active ticket and sub so we only hit the grace ticket
    result = await db.execute(select(ParkingTicket).where(ParkingTicket.ticket_ref == 'TKT-ACTIVE'))
    tkt = result.scalars().first()
    if tkt:
        await db.delete(tkt)
    result = await db.execute(select(ParkingSubscription))
    subs = result.scalars().all()
    for s in subs:
        await db.delete(s)
    await db.flush()

    status, expires_at, ticket_id, sub_id = await lookup_ticket("B123ABC", None, now, db)
    
    assert status == "grace"
    assert ticket_id is not None
    assert sub_id is None
    
    if expires_at and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    assert expires_at < now

@pytest.mark.asyncio
async def test_none_status(db: AsyncSession, seed_data):
    """Test 4: Returns 'none' when plate is known but no coverage exists."""
    now = datetime.now(UTC)
    status, expires_at, ticket_id, sub_id = await lookup_ticket("CJ99XYZ", None, now, db)
    
    assert status == "none"
    assert expires_at is None
    assert ticket_id is None
    assert sub_id is None

@pytest.mark.asyncio
async def test_unknown_status(db: AsyncSession, seed_data):
    """Test 5: Returns 'unknown' when plate is completely unknown."""
    now = datetime.now(UTC)
    status, expires_at, ticket_id, sub_id = await lookup_ticket("GHOST", None, now, db)
    
    assert status == "unknown"
    assert expires_at is None
    assert ticket_id is None
    assert sub_id is None
