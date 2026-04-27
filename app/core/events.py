"""
app/core/events.py

Lightweight asyncio pub/sub for per-session pipeline events.

publish() is called from process_session (background asyncio Task).
subscribe()/unsubscribe() are called by the SSE endpoint coroutine.
Both run in the same event loop, so asyncio.Queue is safe.
"""

from __future__ import annotations

import asyncio
from typing import Any

# session_id (str) → list of subscriber queues
_listeners: dict[str, list[asyncio.Queue[dict[str, Any] | None]]] = {}


async def subscribe(session_id: str) -> asyncio.Queue[dict[str, Any] | None]:
    """Register a listener for *session_id* and return its queue."""
    q: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=256)
    _listeners.setdefault(session_id, []).append(q)
    return q


def unsubscribe(session_id: str, q: asyncio.Queue) -> None:
    """Remove *q* from the listener list, cleaning up empty entries."""
    lst = _listeners.get(session_id)
    if lst:
        try:
            lst.remove(q)
        except ValueError:
            pass
        if not lst:
            _listeners.pop(session_id, None)


def publish(session_id: str, event_type: str, data: dict[str, Any]) -> None:
    """Broadcast an event to all listeners of *session_id*.

    Non-blocking: queues that are full are silently skipped so the
    processing pipeline is never stalled by a slow client.
    """
    for q in list(_listeners.get(session_id, [])):
        try:
            q.put_nowait({"type": event_type, "data": data})
        except asyncio.QueueFull:
            pass
