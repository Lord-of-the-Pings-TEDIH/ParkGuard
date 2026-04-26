"""Integration tests for the A3 streaming endpoints on the sessions router.

Covers:
  * ``GET /sessions/{id}/detections?since=<cursor>`` — incremental delta payload
  * ``GET /sessions/{id}/events``                    — Server-Sent Events stream

The route handlers are invoked directly with an in-memory SQLite session, so
the tests don't pay the FastAPI/uvicorn round-trip cost or boot the PaddleOCR
model that ``main.app`` pre-warms on startup.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy import BigInteger
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles


@compiles(JSONB, "sqlite")
def _compile_jsonb(type_, compiler, **kw):
    return "TEXT"


@compiles(UUID, "sqlite")
def _compile_uuid(type_, compiler, **kw):
    return "CHAR(32)"


@compiles(BigInteger, "sqlite")
def _compile_bigint(type_, compiler, **kw):
    return "INTEGER"


from app.core import events
from app.models.base import Base
from app.models.detection import Detection, Frame, Session
from app.routers.sessions import list_detections, session_events


TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
_TestingSessionLocal = async_sessionmaker(
    bind=_engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
)

UTC = timezone.utc


@pytest_asyncio.fixture(scope="module", autouse=True)
async def _setup_database():
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db() -> AsyncGenerator[AsyncSession, None]:
    async with _TestingSessionLocal() as s:
        yield s
        await s.rollback()


@pytest.fixture(autouse=True)
def _isolate_event_listeners():
    """events._listeners is module-level — clear it between cases."""
    events._listeners.clear()
    yield
    events._listeners.clear()


async def _make_session(db: AsyncSession, *, status: str = "pending", **kwargs) -> Session:
    session = Session(source_filename="sample.mp4", status=status, **kwargs)
    db.add(session)
    await db.flush()
    return session


async def _make_detection(
    db: AsyncSession,
    session: Session,
    *,
    created_at: datetime | None = None,
    ocr_text: str = "CJ 05 XYO",
) -> Detection:
    frame = Frame(session_id=session.id, frame_index=0, pts_ms=0)
    db.add(frame)
    await db.flush()

    detection = Detection(
        frame_id=frame.id,
        bbox_x=0.0,
        bbox_y=0.0,
        bbox_w=10.0,
        bbox_h=4.0,
        detection_confidence=0.9,
        ocr_normalized_text=ocr_text,
        is_valid_ro_plate=True,
    )
    if created_at is not None:
        detection.created_at = created_at
    db.add(detection)
    await db.flush()
    return detection


# ---------------------------------------------------------------------------
# /sessions/{id}/detections?since=<cursor>
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_detections_returns_all_when_since_omitted(db: AsyncSession) -> None:
    session = await _make_session(db)
    base = datetime.now(UTC)
    older = await _make_detection(db, session, created_at=base - timedelta(seconds=10))
    newer = await _make_detection(db, session, created_at=base)

    result = await list_detections(session_id=session.id, since=None, db=db)

    returned_ids = {d.id for d in result}
    assert returned_ids == {older.id, newer.id}


@pytest.mark.asyncio
async def test_list_detections_with_since_returns_only_newer_rows(
    db: AsyncSession,
) -> None:
    """``since`` is the whole point of the delta endpoint — older rows must be
    excluded so the frontend's incremental fetch stays cheap as the session
    grows."""
    session = await _make_session(db)
    base = datetime.now(UTC)
    older = await _make_detection(db, session, created_at=base - timedelta(seconds=30))
    cursor = base - timedelta(seconds=10)
    newer = await _make_detection(db, session, created_at=base)

    result = await list_detections(
        session_id=session.id, since=cursor.isoformat(), db=db
    )

    returned_ids = [d.id for d in result]
    assert returned_ids == [newer.id]
    assert older.id not in returned_ids


@pytest.mark.asyncio
async def test_list_detections_accepts_zulu_iso_timestamp(db: AsyncSession) -> None:
    """EventSource clients commonly send ``...Z``-suffixed ISO strings; the
    handler explicitly rewrites that to a tz-aware offset."""
    session = await _make_session(db)
    base = datetime.now(UTC)
    cursor = base - timedelta(seconds=10)
    newer = await _make_detection(db, session, created_at=base)

    # Build a Z-suffixed cursor (drop microseconds for a clean ISO string).
    zulu_cursor = cursor.replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")

    result = await list_detections(session_id=session.id, since=zulu_cursor, db=db)
    assert [d.id for d in result] == [newer.id]


@pytest.mark.asyncio
async def test_list_detections_invalid_since_returns_400(db: AsyncSession) -> None:
    session = await _make_session(db)

    with pytest.raises(HTTPException) as exc_info:
        await list_detections(session_id=session.id, since="not-a-date", db=db)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_list_detections_unknown_session_returns_404(db: AsyncSession) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await list_detections(session_id=uuid.uuid4(), since=None, db=db)
    assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# /sessions/{id}/events  (SSE)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_events_unknown_session_returns_404(db: AsyncSession) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await session_events(session_id=uuid.uuid4(), db=db)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_session_events_returns_text_event_stream_response(
    db: AsyncSession,
) -> None:
    session = await _make_session(db, status="processing")

    response = await session_events(session_id=session.id, db=db)

    # SSE clients (EventSource) require this exact media type and rely on the
    # X-Accel-Buffering hint to disable proxy buffering on nginx deployments.
    assert response.media_type == "text/event-stream"
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"

    # Drain the iterator so the stream coroutine can clean up its subscriber.
    response.body_iterator.aclose()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_session_events_completed_session_emits_one_completed_then_closes(
    db: AsyncSession,
) -> None:
    """A terminal-state session must NOT block on the queue — the stream sends
    a single ``session_completed`` event and ends, otherwise reconnects after
    a finished run would hang for 25 s on the next keepalive."""
    session = await _make_session(
        db, status="completed", frames_processed=42, total_frames=42
    )

    response = await session_events(session_id=session.id, db=db)
    chunks = [chunk async for chunk in response.body_iterator]

    assert len(chunks) == 1
    chunk = _decode(chunks[0])
    event_line, data_line = _split_sse(chunk)
    assert event_line == "event: session_completed"
    assert json.loads(data_line.removeprefix("data: ")) == {
        "frames_processed": 42,
        "total_frames": 42,
    }


@pytest.mark.asyncio
async def test_session_events_failed_session_emits_one_failed_then_closes(
    db: AsyncSession,
) -> None:
    session = await _make_session(
        db, status="failed", error_message="Cancelled by user"
    )

    response = await session_events(session_id=session.id, db=db)
    chunks = [chunk async for chunk in response.body_iterator]

    assert len(chunks) == 1
    chunk = _decode(chunks[0])
    event_line, data_line = _split_sse(chunk)
    assert event_line == "event: session_failed"
    assert json.loads(data_line.removeprefix("data: ")) == {
        "error": "Cancelled by user"
    }


@pytest.mark.asyncio
async def test_session_events_streams_published_events_in_order(
    db: AsyncSession,
) -> None:
    """The pub/sub plumbing must propagate publish() → SSE chunk for the live
    case; this is the whole reason the colleague's A3 change exists."""
    session = await _make_session(db, status="processing")
    sid = str(session.id)

    response = await session_events(session_id=session.id, db=db)
    iterator = response.body_iterator

    # Kick the generator: it runs subscribe() before suspending on q.get(),
    # so by the time this task is parked we're guaranteed to be a registered
    # listener.  A short sleep gives the scheduler a chance to run that prefix.
    first_chunk_task = asyncio.create_task(iterator.__anext__())
    await _wait_for_listener(sid)

    events.publish(sid, "frame_processed", {"frames_processed": 3, "total_frames": 10})
    first_chunk = _decode(await asyncio.wait_for(first_chunk_task, timeout=2.0))
    event_line, data_line = _split_sse(first_chunk)
    assert event_line == "event: frame_processed"
    assert json.loads(data_line.removeprefix("data: ")) == {
        "frames_processed": 3,
        "total_frames": 10,
    }

    # A second event on the same connection: detection_finalized.
    second_task = asyncio.create_task(iterator.__anext__())
    events.publish(sid, "detection_finalized", {"plate": "CJ 05 XYO"})
    second_chunk = _decode(await asyncio.wait_for(second_task, timeout=2.0))
    event_line, data_line = _split_sse(second_chunk)
    assert event_line == "event: detection_finalized"
    assert json.loads(data_line.removeprefix("data: ")) == {"plate": "CJ 05 XYO"}

    # Terminal event closes the loop — the next __anext__ should stop iteration.
    third_task = asyncio.create_task(iterator.__anext__())
    events.publish(sid, "session_completed", {"frames_processed": 10, "total_frames": 10})
    third_chunk = _decode(await asyncio.wait_for(third_task, timeout=2.0))
    assert third_chunk.startswith("event: session_completed")

    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(iterator.__anext__(), timeout=2.0)

    # And the listener registry was cleaned up by the stream's finally block.
    assert sid not in events._listeners


@pytest.mark.asyncio
async def test_session_events_drops_subscriber_when_iterator_closes_early(
    db: AsyncSession,
) -> None:
    """If the client disconnects mid-stream the SSE generator's ``finally``
    must run unsubscribe(), or the listener registry leaks one queue per
    aborted connection."""
    session = await _make_session(db, status="processing")
    sid = str(session.id)

    response = await session_events(session_id=session.id, db=db)
    iterator = response.body_iterator

    # Park the generator on q.get() so subscribe() has run.
    pending = asyncio.create_task(iterator.__anext__())
    await _wait_for_listener(sid)
    assert sid in events._listeners

    pending.cancel()
    with pytest.raises((asyncio.CancelledError, StopAsyncIteration)):
        await pending
    await iterator.aclose()  # type: ignore[attr-defined]

    assert sid not in events._listeners


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode(chunk: bytes | str) -> str:
    return chunk.decode() if isinstance(chunk, (bytes, bytearray)) else chunk


def _split_sse(chunk: str) -> tuple[str, str]:
    """Return (event-line, data-line) from an ``event: …\\ndata: …\\n\\n`` block."""
    body = chunk.rstrip("\n")
    lines = body.split("\n")
    assert len(lines) >= 2, f"unexpected SSE chunk: {chunk!r}"
    return lines[0], lines[1]


async def _wait_for_listener(session_id: str, *, timeout: float = 1.0) -> None:
    """Spin until the SSE generator has registered its subscriber.

    Avoids a hard-coded sleep — the generator runs subscribe() before parking
    on q.get(), so as soon as the listener appears we know we can safely
    publish without racing the registration.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while session_id not in events._listeners:
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError(
                f"SSE generator never subscribed for {session_id!r}"
            )
        await asyncio.sleep(0)
