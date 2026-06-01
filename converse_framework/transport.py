"""Transport protocol and in-memory queue transport for testing."""

from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

from converse_framework.events import FrameworkEvent


@runtime_checkable
class Transport(Protocol):
    """Protocol for sending and receiving framework events over a transport."""

    async def send_event(self, event: FrameworkEvent) -> None: ...

    async def receive_event(self) -> FrameworkEvent: ...


class QueueTransport(Transport):
    """In-memory dual-queue transport for testing consumers without I/O."""

    def __init__(self) -> None:
        self._send_queue: asyncio.Queue[FrameworkEvent] = asyncio.Queue()
        self._recv_queue: asyncio.Queue[FrameworkEvent] = asyncio.Queue()

    async def send_event(self, event: FrameworkEvent) -> None:
        await self._send_queue.put(event)

    async def receive_event(self) -> FrameworkEvent:
        return await self._recv_queue.get()
