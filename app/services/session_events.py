"""In-memory pub/sub for live session updates.

A single FastAPI process can have many SSE clients subscribed to the same
session (e.g. the operator's browser plus a dashboard on a wall).  We
publish events from the pipeline (frame_processed / detection_changed /
status) into one :class:`asyncio.Queue` per subscriber, indexed by the
session id, so every connected client receives the same stream without
the publisher needing to know how many listeners are out there.

Why not Redis / NATS?  ParkGuard runs as a single Uvicorn process today;
an in-memory bus is the simplest correct thing.  When the deployment
moves to multiple workers the same interface can be backed by Redis Pub/Sub
without changing the pipeline or the SSE endpoint.

The bus is intentionally best-effort: if a subscriber's queue fills up we
drop the oldest events rather than blocking the publisher (the pipeline
must keep running even when a slow client falls behind).  The frontend
re-fetches on reconnect, so missed events surface naturally.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Iterable

logger = logging.getLogger(__name__)


# Each subscriber gets a bounded queue so a slow client cannot exhaust
# memory with a runaway pipeline.  When the queue is full we drop the
# *oldest* event rather than the new one — clients care about the latest
# state, and a missed intermediate `frame` event is recoverable from the
# next one.
_QUEUE_MAX_SIZE: int = 256


@dataclass(frozen=True)
class SessionEvent:
    """A single event broadcast for a session.

    ``event`` is the SSE event type (``frame``, ``detections``, ``status``);
    ``data`` is a JSON-serialisable payload.  Keeping events small lets us
    fan out to dozens of subscribers cheaply.
    """

    event: str
    data: dict


class SessionEventBus:
    """Asyncio-only fan-out hub for ``SessionEvent`` objects.

    The bus is process-local and not thread-safe — callers (the pipeline
    coroutine, the SSE endpoint) all run on the same event loop.
    """

    def __init__(self, *, queue_max_size: int = _QUEUE_MAX_SIZE) -> None:
        self._subscribers: dict[uuid.UUID, list[asyncio.Queue[SessionEvent]]] = (
            defaultdict(list)
        )
        self._queue_max_size = queue_max_size

    def publish(self, session_id: uuid.UUID, event: SessionEvent) -> None:
        """Send *event* to every subscriber listening on *session_id*.

        Drops the oldest queued event when a subscriber's queue is full —
        the pipeline must never block on a slow consumer.  Returns
        immediately so callers can keep processing frames.
        """
        queues = self._subscribers.get(session_id)
        if not queues:
            return
        for queue in queues:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drop oldest, push newest.  This keeps the consumer at
                # the most recent state at the cost of a missed intermediate.
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    logger.warning(
                        "SessionEventBus: dropping event for %s — queue still full",
                        session_id,
                    )

    @asynccontextmanager
    async def subscribe(
        self, session_id: uuid.UUID
    ) -> AsyncIterator[asyncio.Queue[SessionEvent]]:
        """Yield a queue that receives every event for *session_id*.

        Use as a context manager so the queue is unregistered automatically
        when the subscriber disconnects, even on exceptions.  Multiple
        subscribers can listen on the same session — each gets its own
        queue and sees every event.
        """
        queue: asyncio.Queue[SessionEvent] = asyncio.Queue(
            maxsize=self._queue_max_size
        )
        self._subscribers[session_id].append(queue)
        try:
            yield queue
        finally:
            try:
                self._subscribers[session_id].remove(queue)
            except ValueError:
                pass
            if not self._subscribers[session_id]:
                self._subscribers.pop(session_id, None)

    def has_subscribers(self, session_id: uuid.UUID) -> bool:
        """True when at least one client is subscribed to *session_id*."""
        return bool(self._subscribers.get(session_id))

    def subscriber_count(self, session_id: uuid.UUID) -> int:
        return len(self._subscribers.get(session_id, ()))

    def reset(self) -> None:
        """Drop every subscriber.  Test helper — never call from production."""
        self._subscribers.clear()


# Module-level singleton.  The pipeline imports this directly so it can
# publish without taking a dependency on a FastAPI request scope.
event_bus = SessionEventBus()


def publish_frame_processed(
    session_id: uuid.UUID,
    *,
    frames_processed: int,
    total_frames: int | None,
) -> None:
    """Convenience wrapper for the most common event.

    Called from the pipeline once per processed frame (or per commit
    batch).  Keeps the payload tiny so the SSE stream stays cheap even at
    high frame rates.
    """
    event_bus.publish(
        session_id,
        SessionEvent(
            event="frame",
            data={
                "frames_processed": int(frames_processed),
                "total_frames": int(total_frames) if total_frames else None,
            },
        ),
    )


def publish_detections_changed(
    session_id: uuid.UUID,
    *,
    detection_ids: Iterable[uuid.UUID],
) -> None:
    """Notify subscribers that a batch of detections was created/updated.

    The payload only carries the touched detection ids — the frontend
    fetches the full rows via the delta endpoint so the SSE message stays
    small regardless of crop size.
    """
    ids = [str(d) for d in detection_ids]
    if not ids:
        return
    event_bus.publish(
        session_id,
        SessionEvent(event="detections", data={"detection_ids": ids}),
    )


def publish_status_changed(
    session_id: uuid.UUID,
    *,
    status: str,
    error_message: str | None = None,
) -> None:
    """Notify subscribers that the session moved to a new status."""
    event_bus.publish(
        session_id,
        SessionEvent(
            event="status",
            data={
                "status": status,
                "error_message": error_message,
            },
        ),
    )
