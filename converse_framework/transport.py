"""Transport protocol and in-memory queue transport for testing."""

from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

from converse_framework.events import FrameworkEvent


@runtime_checkable
class Transport(Protocol):
    """Protocol for sending and receiving framework events over a transport.

    A :class:`Transport` is the boundary between the framework and
    whatever wire the host application uses (WebSocket, in-process
    queue, log file, ...). The framework itself never reaches
    across this boundary -- it produces :class:`FrameworkEvent`
    instances and the transport decides how to serialise and
    deliver them.

    Implementations must be :func:`asyncio` compatible. Tests use
    :class:`QueueTransport` to capture events without involving a
    real I/O stack.
    """

    async def send_event(self, event: FrameworkEvent) -> None: ...

    async def receive_event(self) -> FrameworkEvent: ...


class QueueTransport(Transport):
    """In-memory dual-queue transport for testing consumers without I/O.

    Maintains two independent ``asyncio.Queue`` instances: one
    queue collects events the pipeline / sinks push via
    :meth:`send_event` (the "outbound" stream a fake client would
    read from), the other feeds events a fake client pushes back
    into the framework via :meth:`receive_event`.

    The queues are unbounded by default, which matches the
    semantics expected by tests: every emitted event must be
    observable, and the test controls when the consumer drains.
    """

    def __init__(self) -> None:
        self._send_queue: asyncio.Queue[FrameworkEvent] = asyncio.Queue()
        self._recv_queue: asyncio.Queue[FrameworkEvent] = asyncio.Queue()

    async def send_event(self, event: FrameworkEvent) -> None:
        await self._send_queue.put(event)

    async def receive_event(self) -> FrameworkEvent:
        return await self._recv_queue.get()
