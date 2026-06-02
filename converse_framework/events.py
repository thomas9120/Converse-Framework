"""Event sink API and event envelope for the speech stack."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FrameworkEvent:
    """Canonical event envelope emitted by the framework.

    The wire shape is the v0.1 contract: ``{"type": str, "ts": float,
    "payload": dict}``. ``type`` is a stable dotted string (e.g.
    ``"turn.started"``, ``"tts.audio"``); ``ts`` is a monotonic
    timestamp taken from :func:`time.perf_counter`; ``payload`` is
    the keyword arguments supplied to
    :meth:`EventSink.emit`. The dataclass is mutable so call sites
    can adjust fields before forwarding, but the event flow itself
    treats instances as value objects.
    """

    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.perf_counter)

    def to_json(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "ts": self.ts,
            "payload": self.payload,
        }


class EventSink:
    """Abstract sink for emitting typed events during pipeline execution.

    Implementations forward :meth:`emit` calls to their own delivery
    mechanism (WebSocket, in-memory queue, log, ...). The framework
    depends on this protocol rather than a concrete class so apps
    can wire the pipeline into any transport without changing the
    pipeline code.

    The base class raises :class:`NotImplementedError`; subclasses
    must override :meth:`emit`.
    """

    async def emit(self, event_type: str, **payload: Any) -> None:
        raise NotImplementedError


class QueueEventSink(EventSink):
    """In-memory event sink backed by an :class:`asyncio.Queue`.

    Each call to :meth:`emit` puts a wire-shaped dict onto the queue
    owned by the caller. Tests use this sink to assert on the exact
    event stream the pipeline produces without involving a real
    transport.

    Args:
        queue: The asyncio queue the sink writes into. The caller
            owns the queue and is responsible for draining or
            closing it.
    """

    def __init__(self, queue: asyncio.Queue[dict[str, Any]]):
        self.queue = queue

    async def emit(self, event_type: str, **payload: Any) -> None:
        await self.queue.put(
            {"type": event_type, "ts": time.perf_counter(), "payload": payload}
        )


class TransportEventSink(EventSink):
    """Event sink adapter that forwards emitted events to a transport.

    This is the bridge for consumers that already implemented the
    :class:`converse_framework.transport.Transport` protocol and want
    pipeline / collector events delivered through ``send_event`` without
    writing their own small adapter class.

    Args:
        transport: Object with an async ``send_event(FrameworkEvent)``
            method, typically a ``Transport`` implementation.
    """

    def __init__(self, transport) -> None:
        self.transport = transport

    async def emit(self, event_type: str, **payload: Any) -> None:
        await self.transport.send_event(FrameworkEvent(event_type, payload))
