"""Tests for Transport protocol and QueueTransport."""

import asyncio

from converse_framework.events import FrameworkEvent
from converse_framework.transport import QueueTransport, Transport


def test_queue_transport_is_transport():
    qt = QueueTransport()
    assert isinstance(qt, Transport)


def test_queue_transport_send_and_receive():
    async def run():
        qt = QueueTransport()
        event = FrameworkEvent(type="test.x", payload={"a": 1})
        await qt.send_event(event)
        # Send queue is internal — verify by accessing directly
        sent = await qt._send_queue.get()
        assert sent is event
        assert sent.type == "test.x"
        assert sent.payload == {"a": 1}

    asyncio.run(run())


def test_queue_transport_receive_blocks_until_event():
    async def run():
        qt = QueueTransport()
        event = FrameworkEvent(type="incoming")
        # Put something on the receive queue from "outside"
        await qt._recv_queue.put(event)
        received = await qt.receive_event()
        assert received is event
        assert received.type == "incoming"

    asyncio.run(run())


def test_queue_transport_separate_queues():
    """Send queue and receive queue are independent."""

    async def run():
        qt = QueueTransport()
        send_evt = FrameworkEvent(type="send")
        recv_evt = FrameworkEvent(type="recv")

        await qt.send_event(send_evt)
        await qt._recv_queue.put(recv_evt)

        sent = await qt._send_queue.get()
        received = await qt.receive_event()

        assert sent.type == "send"
        assert received.type == "recv"

    asyncio.run(run())
