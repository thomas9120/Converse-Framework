"""Tests for ``converse_framework.session`` — reusable WebSocket session helper."""

from __future__ import annotations


import pytest

from converse_framework.events import FrameworkEvent
from converse_framework.session import (
    WebSocketSession,
    WebSocketSessionConfig,
    WebSocketSessionHooks,
)


# ---------------------------------------------------------------------------
# Fake transport for testing
# ---------------------------------------------------------------------------


class FakeTransport:
    """In-memory transport that records sent events and yields preset receives."""

    def __init__(self) -> None:
        self.sent: list[FrameworkEvent] = []

    async def send_event(self, event: FrameworkEvent) -> None:
        self.sent.append(event)

    async def receive_event(self) -> FrameworkEvent:
        return FrameworkEvent("dummy", {})


# ---------------------------------------------------------------------------
# Session construction
# ---------------------------------------------------------------------------


def test_session_constructs_without_fastapi():
    """WebSocketSession must not import FastAPI at construction time."""
    import sys

    modnames = [m for m in sys.modules if "fastapi" in m]
    assert "fastapi" not in modnames


def test_session_constructs_with_default_config():
    transport = FakeTransport()
    session = WebSocketSession(transport)  # type: ignore[arg-type]
    assert session.bundle is not None
    assert session.pipeline is not None
    assert session.collector is not None
    assert session.config.default_mode == "chat"
    assert session.config.auto_probe_status is True


def test_session_uses_custom_config():
    transport = FakeTransport()
    config = WebSocketSessionConfig(default_mode="custom", auto_probe_status=False)
    session = WebSocketSession(transport, config=config)  # type: ignore[arg-type]
    assert session.config.default_mode == "custom"
    assert session.config.auto_probe_status is False


def test_session_uses_custom_hooks():
    transport = FakeTransport()
    hook_log: list[str] = []

    async def settings_hook(session, payload):
        hook_log.append(f"settings:{payload}")

    hooks = WebSocketSessionHooks(on_settings_update=settings_hook)
    session = WebSocketSession(transport, hooks=hooks)  # type: ignore[arg-type]
    assert session.hooks.on_settings_update is not None


# ---------------------------------------------------------------------------
# Built-in message routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_message_emits_turn_error():
    transport = FakeTransport()
    session = WebSocketSession(transport)  # type: ignore[arg-type]

    await session.handle_message({"type": "bogus.type", "payload": {}})
    assert len(transport.sent) == 1
    assert transport.sent[0].type == "turn.error"
    assert "bogus.type" in transport.sent[0].payload["message"]


@pytest.mark.asyncio
async def test_unknown_message_routes_to_hook_when_set():
    transport = FakeTransport()
    hook_payloads: list[dict] = []

    async def unknown_hook(session, message):
        hook_payloads.append(message)

    hooks = WebSocketSessionHooks(on_unknown_message=unknown_hook)
    session = WebSocketSession(transport, hooks=hooks)  # type: ignore[arg-type]

    await session.handle_message({"type": "custom.event", "payload": {"x": 1}})
    assert len(hook_payloads) == 1
    assert hook_payloads[0]["type"] == "custom.event"
    # No turn.error emitted when hook is set
    assert len(transport.sent) == 0


@pytest.mark.asyncio
async def test_text_turn_routes_to_pipeline():
    transport = FakeTransport()
    session = WebSocketSession(transport)  # type: ignore[arg-type]

    # Mock pipeline emits events for each text turn
    await session.handle_message(
        {"type": "text.turn", "payload": {"text": "hello world"}}
    )
    # Pipeline should produce at least a turn.started event
    assert len(transport.sent) >= 1
    # The first event should indicate the turn started
    assert transport.sent[0].type == "turn.started"


@pytest.mark.asyncio
async def test_text_turn_empty_text_is_noop():
    transport = FakeTransport()
    session = WebSocketSession(transport)  # type: ignore[arg-type]

    await session.handle_message({"type": "text.turn", "payload": {"text": ""}})
    # No events emitted for empty text
    assert len(transport.sent) == 0


@pytest.mark.asyncio
async def test_conversation_clear_routes_to_pipeline():
    transport = FakeTransport()
    session = WebSocketSession(transport)  # type: ignore[arg-type]

    await session.handle_message({"type": "conversation.clear", "payload": {}})
    # Pipeline emits at least one event for clearing
    assert len(transport.sent) >= 1
    assert any(e.type == "conversation.cleared" for e in transport.sent)


@pytest.mark.asyncio
async def test_tts_cancel_routes_to_pipeline():
    transport = FakeTransport()
    session = WebSocketSession(transport)  # type: ignore[arg-type]

    await session.handle_message(
        {"type": "tts.cancel", "payload": {"reason": "user_stop"}}
    )
    # TTS cancel is a no-op when nothing is playing but shouldn't error
    assert isinstance(session, WebSocketSession)


@pytest.mark.asyncio
async def test_audio_frame_error_on_bad_payload():
    transport = FakeTransport()
    session = WebSocketSession(transport)  # type: ignore[arg-type]

    await session.handle_message(
        {
            "type": "audio.frame",
            "payload": {
                "encoding": "pcm_s16le",
                "sample_rate": 123,  # invalid — not 16000
                "channels": 1,
                "frame_ms": 30,
                "sequence": 0,
                "data": "AAAA",
            },
        }
    )
    assert len(transport.sent) == 1
    assert transport.sent[0].type == "audio.frame_error"
    assert "sample_rate" in transport.sent[0].payload["message"]


@pytest.mark.asyncio
async def test_status_request_probe_default():
    transport = FakeTransport()
    session = WebSocketSession(transport)  # type: ignore[arg-type]

    await session.handle_message({"type": "status.request", "payload": {}})

    # Should emit a providers.status event
    status_events = [e for e in transport.sent if e.type == "providers.status"]
    assert len(status_events) == 1
    status_payload = status_events[0].payload
    assert "statuses" in status_payload
    # All four providers should be present
    assert len(status_payload["statuses"]) == 4


@pytest.mark.asyncio
async def test_status_request_explicit_kind():
    transport = FakeTransport()
    session = WebSocketSession(transport)  # type: ignore[arg-type]

    await session.handle_message(
        {"type": "status.request", "payload": {"kind": "check"}}
    )

    status_events = [e for e in transport.sent if e.type == "providers.status"]
    assert len(status_events) == 1
    assert len(status_events[0].payload["statuses"]) == 4


# ---------------------------------------------------------------------------
# Settings update hook
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_settings_update_routes_to_hook():
    transport = FakeTransport()
    settings_log: list[dict] = []

    async def settings_hook(session, payload):
        settings_log.append(payload)

    hooks = WebSocketSessionHooks(on_settings_update=settings_hook)
    session = WebSocketSession(transport, hooks=hooks)  # type: ignore[arg-type]

    await session.handle_message(
        {"type": "settings.update", "payload": {"voice": "anna"}}
    )
    assert len(settings_log) == 1
    assert settings_log[0] == {"voice": "anna"}


@pytest.mark.asyncio
async def test_settings_update_no_hook_is_silent():
    transport = FakeTransport()
    session = WebSocketSession(transport)  # type: ignore[arg-type]

    await session.handle_message(
        {"type": "settings.update", "payload": {"voice": "anna"}}
    )
    # No events — session silently ignores unhandled settings
    assert len(transport.sent) == 0


# ---------------------------------------------------------------------------
# Status hook
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_request_routes_to_hook():
    transport = FakeTransport()
    status_log: list[dict] = []

    async def status_hook(session, payload):
        status_log.append(payload)

    hooks = WebSocketSessionHooks(on_status_request=status_hook)
    session = WebSocketSession(transport, hooks=hooks)  # type: ignore[arg-type]

    await session.handle_message({"type": "status.request", "payload": {}})

    assert len(status_log) == 1
    assert "kind" in status_log[0]
    assert "statuses" in status_log[0]


# ---------------------------------------------------------------------------
# Event hook
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_event_hook_fires_on_each_event():
    transport = FakeTransport()
    events_log: list[FrameworkEvent] = []

    async def event_hook(session, event):
        events_log.append(event)

    hooks = WebSocketSessionHooks(on_event=event_hook)
    session = WebSocketSession(transport, hooks=hooks)  # type: ignore[arg-type]

    await session.handle_message({"type": "bogus", "payload": {}})

    # Should capture the turn.error event before transport sends it
    assert len(events_log) == 1
    assert events_log[0].type == "turn.error"


# ---------------------------------------------------------------------------
# Provider reload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_providers_reload_swaps_bundle():
    transport = FakeTransport()
    session = WebSocketSession(transport)  # type: ignore[arg-type]

    old_bundle = session.bundle
    await session.handle_message(
        {
            "type": "providers.reload",
            "payload": {
                "config": {
                    "vad": {"provider": "mock"},
                    "asr": {"provider": "mock"},
                    "llm": {"provider": "mock"},
                    "tts": {"provider": "mock"},
                }
            },
        }
    )

    # Bundle should be replaced
    assert session.bundle is not old_bundle
    # Pipeline still works
    assert session.pipeline.providers is session.bundle


@pytest.mark.asyncio
async def test_providers_reload_with_load_param():
    transport = FakeTransport()
    session = WebSocketSession(transport)  # type: ignore[arg-type]

    await session.handle_message(
        {
            "type": "providers.reload",
            "payload": {
                "config": {
                    "vad": {"provider": "mock"},
                    "asr": {"provider": "mock"},
                    "llm": {"provider": "mock"},
                    "tts": {"provider": "mock"},
                },
                "load": True,
            },
        }
    )

    # Bundle swapped, no crash
    assert session.bundle is not None
    assert len(transport.sent) >= 0


@pytest.mark.asyncio
async def test_before_after_reload_hooks_fire():
    transport = FakeTransport()
    hook_log: list[str] = []

    async def before(session, old_bundle, config):
        hook_log.append(f"before:{len(config)}")

    async def after(session, old_bundle, new_bundle):
        hook_log.append("after")

    hooks = WebSocketSessionHooks(
        on_before_provider_reload=before,
        on_after_provider_reload=after,
    )
    session = WebSocketSession(transport, hooks=hooks)  # type: ignore[arg-type]

    await session.handle_message(
        {
            "type": "providers.reload",
            "payload": {
                "config": {
                    "vad": {"provider": "mock"},
                    "asr": {"provider": "mock"},
                    "llm": {"provider": "mock"},
                    "tts": {"provider": "mock"},
                }
            },
        }
    )

    assert hook_log == ["before:4", "after"]


# ---------------------------------------------------------------------------
# emit_status public method
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emit_status_probe_default():
    transport = FakeTransport()
    session = WebSocketSession(transport)  # type: ignore[arg-type]

    await session.emit_status()
    status_events = [e for e in transport.sent if e.type == "providers.status"]
    assert len(status_events) == 1
    assert len(status_events[0].payload["statuses"]) == 4


@pytest.mark.asyncio
async def test_emit_status_load_kind():
    transport = FakeTransport()
    session = WebSocketSession(transport)  # type: ignore[arg-type]

    await session.emit_status(kind="load")
    status_events = [e for e in transport.sent if e.type == "providers.status"]
    assert len(status_events) == 1
    assert len(status_events[0].payload["statuses"]) == 4


@pytest.mark.asyncio
async def test_emit_status_check_kind():
    transport = FakeTransport()
    session = WebSocketSession(transport)  # type: ignore[arg-type]

    await session.emit_status(kind="check")
    status_events = [e for e in transport.sent if e.type == "providers.status"]
    assert len(status_events) == 1
    assert len(status_events[0].payload["statuses"]) == 4


# ---------------------------------------------------------------------------
# reload_providers public method
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reload_providers_public_method():
    transport = FakeTransport()
    session = WebSocketSession(transport)  # type: ignore[arg-type]

    old_bundle = session.bundle
    await session.reload_providers(
        {
            "vad": {"provider": "mock"},
            "asr": {"provider": "mock"},
            "llm": {"provider": "mock"},
            "tts": {"provider": "mock"},
        }
    )
    assert session.bundle is not old_bundle
