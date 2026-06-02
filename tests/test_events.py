"""Tests for event envelope compatibility and QueueEventSink."""

import asyncio

from converse_framework.events import (
    EventSink,
    FrameworkEvent,
    QueueEventSink,
    TransportEventSink,
)


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


def test_transport_event_sink_forwards_framework_event():
    class FakeTransport:
        def __init__(self):
            self.events = []

        async def send_event(self, event):
            self.events.append(event)

    async def run():
        transport = FakeTransport()
        sink = TransportEventSink(transport)
        await sink.emit("bridge.event", ok=True)
        return transport.events

    events = asyncio.run(run())
    assert len(events) == 1
    assert isinstance(events[0], FrameworkEvent)
    assert events[0].type == "bridge.event"
    assert events[0].payload == {"ok": True}


# ---------------------------------------------------------------------------
# provider_events helper functions
# ---------------------------------------------------------------------------


def test_provider_loading_event_shape():
    from converse_framework.provider_events import provider_loading_event

    result = provider_loading_event(
        kind="asr",
        provider="faster-whisper",
        message="Loading model...",
    )
    assert result["event_type"] == "provider.loading"
    assert result["kind"] == "asr"
    assert result["provider"] == "faster-whisper"
    assert result["loaded"] is False
    assert "message" in result
    assert result["message"] == "Loading model..."


def test_provider_loaded_event_shape():
    from converse_framework.provider_events import provider_loaded_event

    result = provider_loaded_event(
        kind="tts",
        provider="pocket-tts",
        latency_ms=1500,
    )
    assert result["event_type"] == "provider.loaded"
    assert result["kind"] == "tts"
    assert result["provider"] == "pocket-tts"
    assert result["loaded"] is True
    assert result["latency_ms"] == 1500


def test_provider_error_event_shape():
    from converse_framework.provider_events import provider_error_event

    result = provider_error_event(
        kind="asr",
        provider="faster-whisper",
        message="Load timed out",
        error_type="TimeoutError",
        loaded=False,
    )
    assert result["event_type"] == "provider.error"
    assert result["kind"] == "asr"
    assert result["provider"] == "faster-whisper"
    assert result["error_type"] == "TimeoutError"
    assert result["message"] == "Load timed out"
    assert result["loaded"] is False
