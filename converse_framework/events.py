"""Event sink API and event envelope for the speech stack."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FrameworkEvent:
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
    """Sink for emitting typed events during pipeline execution."""

    async def emit(self, event_type: str, **payload: Any) -> None:
        raise NotImplementedError


class QueueEventSink(EventSink):
    """Event sink backed by an asyncio.Queue, useful for testing."""

    def __init__(self, queue: asyncio.Queue[dict[str, Any]]):
        self.queue = queue

    async def emit(self, event_type: str, **payload: Any) -> None:
        await self.queue.put(
            {"type": event_type, "ts": time.perf_counter(), "payload": payload}
        )
