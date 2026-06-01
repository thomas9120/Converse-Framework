"""Tests for event envelope compatibility and QueueEventSink."""

import asyncio

from converse_framework.events import EventSink, FrameworkEvent, QueueEventSink


def test_framework_event_has_type_ts_payload():
    event = FrameworkEvent(type="test.event", payload={"key": "val"})
    d = event.to_json()
    assert d["type"] == "test.event"
    assert "ts" in d
    assert isinstance(d["ts"], float)
    assert d["payload"] == {"key": "val"}


def test_framework_event_defaults():
    event = FrameworkEvent(type="def")
    d = event.to_json()
    assert d["type"] == "def"
    assert d["payload"] == {}


def test_event_sink_is_abstract():
    sink = EventSink()
    try:
        asyncio.run(sink.emit("test"))
        raise AssertionError("should have raised")
    except NotImplementedError:
        pass


def test_queue_event_sink_puts_on_queue():
    async def run():
        queue = asyncio.Queue()
        sink = QueueEventSink(queue)
        await sink.emit("my.event", foo="bar", num=42)
        event = await queue.get()
        return event

    event = asyncio.run(run())
    assert event["type"] == "my.event"
    assert event["payload"] == {"foo": "bar", "num": 42}
    assert isinstance(event["ts"], float)


def test_queue_event_sink_ordering():
    async def run():
        queue = asyncio.Queue()
        sink = QueueEventSink(queue)
        await sink.emit("first")
        await sink.emit("second")
        e1 = await queue.get()
        e2 = await queue.get()
        return e1, e2

    e1, e2 = asyncio.run(run())
    assert e1["type"] == "first"
    assert e2["type"] == "second"


def test_event_timestamps_are_monotonic():
    """Emitted events must have increasing timestamps."""

    async def run():
        queue = asyncio.Queue()
        sink = QueueEventSink(queue)
        await sink.emit("a")
        await sink.emit("b")
        e1 = await queue.get()
        e2 = await queue.get()
        return e1["ts"], e2["ts"]

    ts1, ts2 = asyncio.run(run())
    assert ts2 >= ts1
