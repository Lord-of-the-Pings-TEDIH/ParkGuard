"""Unit tests for the asyncio pub/sub used by the SSE endpoint.

Covers the contract that ``app/routers/sessions.py::session_events`` relies on:
events published while a subscriber is listening land in its queue, queues do
not cross session boundaries, full queues drop silently (so a slow client
cannot stall the processing pipeline) and unsubscribe leaves no orphan state
in the module-level listener registry.
"""

from __future__ import annotations

import asyncio

import pytest

from app.core import events


@pytest.fixture(autouse=True)
def _clear_listeners():
    """events._listeners is module-level; reset it between tests."""
    events._listeners.clear()
    yield
    events._listeners.clear()


@pytest.mark.asyncio
async def test_subscribe_returns_queue_and_registers_listener() -> None:
    q = await events.subscribe("session-1")
    assert isinstance(q, asyncio.Queue)
    assert events._listeners["session-1"] == [q]


@pytest.mark.asyncio
async def test_publish_delivers_event_to_subscriber() -> None:
    q = await events.subscribe("session-1")

    events.publish("session-1", "frame_processed", {"frames_processed": 7})

    delivered = await asyncio.wait_for(q.get(), timeout=1.0)
    assert delivered == {
        "type": "frame_processed",
        "data": {"frames_processed": 7},
    }


@pytest.mark.asyncio
async def test_publish_with_no_listeners_is_a_noop() -> None:
    # Should not raise even though nobody is subscribed.
    events.publish("nobody-listening", "frame_processed", {"frames_processed": 1})
    assert "nobody-listening" not in events._listeners


@pytest.mark.asyncio
async def test_publish_skips_full_queue_without_blocking() -> None:
    """A slow consumer must never stall the pipeline that calls publish()."""
    q = await events.subscribe("session-1")
    # Drain capacity by stuffing the queue up to its maxsize (256).
    for _ in range(q.maxsize):
        q.put_nowait({"type": "filler", "data": {}})

    # publish() catches QueueFull internally — must return cleanly.
    events.publish("session-1", "frame_processed", {"frames_processed": 1})

    # The dropped event is silently lost; queue still at capacity.
    assert q.qsize() == q.maxsize


@pytest.mark.asyncio
async def test_unsubscribe_removes_listener_and_cleans_up_session() -> None:
    q = await events.subscribe("session-1")
    events.unsubscribe("session-1", q)
    # Once the last listener leaves, the session entry is removed entirely so
    # the registry doesn't accumulate dead session_ids over time.
    assert "session-1" not in events._listeners


@pytest.mark.asyncio
async def test_unsubscribe_unknown_queue_is_safe() -> None:
    q = await events.subscribe("session-1")
    other_q: asyncio.Queue = asyncio.Queue()

    # Removing a queue that was never registered must be a no-op, not raise.
    events.unsubscribe("session-1", other_q)
    assert events._listeners["session-1"] == [q]

    # Unknown session id is also safe.
    events.unsubscribe("unknown-session", other_q)


@pytest.mark.asyncio
async def test_events_are_isolated_by_session_id() -> None:
    q_a = await events.subscribe("session-A")
    q_b = await events.subscribe("session-B")

    events.publish("session-A", "detection_finalized", {"plate": "CJ 05 XYO"})

    delivered = await asyncio.wait_for(q_a.get(), timeout=1.0)
    assert delivered["data"] == {"plate": "CJ 05 XYO"}
    assert q_b.empty()


@pytest.mark.asyncio
async def test_multiple_subscribers_all_receive_same_event() -> None:
    q1 = await events.subscribe("session-1")
    q2 = await events.subscribe("session-1")

    events.publish("session-1", "session_completed", {"frames_processed": 42})

    msg1 = await asyncio.wait_for(q1.get(), timeout=1.0)
    msg2 = await asyncio.wait_for(q2.get(), timeout=1.0)
    assert msg1 == msg2 == {
        "type": "session_completed",
        "data": {"frames_processed": 42},
    }


@pytest.mark.asyncio
async def test_partial_unsubscribe_leaves_other_listeners_active() -> None:
    q1 = await events.subscribe("session-1")
    q2 = await events.subscribe("session-1")
    events.unsubscribe("session-1", q1)

    events.publish("session-1", "frame_processed", {"frames_processed": 1})

    msg = await asyncio.wait_for(q2.get(), timeout=1.0)
    assert msg["data"] == {"frames_processed": 1}
    assert q1.empty()
    assert events._listeners["session-1"] == [q2]
