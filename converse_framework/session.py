"""Reusable WebSocket session helper built on the framework transport protocol.

Provides the runtime state machine and message routing that the
``websocket_voice_chat`` recipe previously owned, packaged as a
framework-level component so apps do not need to reimplement the
message-dispatch loop.

The session module depends only on framework protocols and dataclasses.
It does **not** import FastAPI or any HTTP/Wire server library.
Apps serve the actual WebSocket endpoint and pass events to
:meth:`WebSocketSession.handle_message`.

Example usage in a FastAPI endpoint::

    import json

    from fastapi import FastAPI, WebSocket
    from converse_framework.transport import Transport
    from converse_framework.session import WebSocketSession, WebSocketSessionConfig

    app = FastAPI()

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        transport = _as_transport(websocket)
        session = WebSocketSession(transport)
        while True:
            packet = await websocket.receive()
            if packet.get("type") == "websocket.disconnect":
                break
            if packet.get("bytes") is not None:
                await session.handle_message(packet["bytes"])
            elif packet.get("text") is not None:
                await session.handle_message(json.loads(packet["text"]))
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from converse_framework.audio_utils import (
    AudioFrameStats,
    parse_audio_frame,
    parse_binary_audio_frame,
)
from converse_framework.events import FrameworkEvent, TransportEventSink
from converse_framework.pipeline import PipelineConfig, SpeechPipeline
from converse_framework.registry import ProviderBundle, build_provider_bundle
from converse_framework.transport import Transport
from converse_framework.utterance_collector import (
    AudioUtteranceCollector,
    UtteranceCollectorConfig,
)

# Forward-reference so hook signatures can use the class name.
HookFn = Callable[["WebSocketSession", dict[str, Any]], Awaitable[None]]  # noqa: F821


@dataclass
class WebSocketSessionConfig:
    """Configuration for building a :class:`WebSocketSession`.

    Attributes:
        provider_config: Provider configuration dict passed to
            :func:`build_provider_bundle`. Defaults to mock-only.
        collector_config: VAD frame collector configuration.
        pipeline_config: Pipeline configuration.
        default_mode: Default conversation mode when the client
            does not specify one (``"chat"``, ``"custom"``, ...).
        auto_probe_status: If True, run ``probe_statuses()`` on
            each ``status.request`` message. If False, run the
            heavier ``check_statuses()`` call instead.
    """

    provider_config: dict[str, dict[str, Any]] | None = None
    collector_config: UtteranceCollectorConfig | None = None
    pipeline_config: PipelineConfig | None = None
    default_mode: str = "chat"
    auto_probe_status: bool = True


@dataclass
class WebSocketSessionHooks:
    """Optional hooks injected into :class:`WebSocketSession`.

    Each hook is an async callable ``(session, payload) -> None``.
    If a hook is not provided, the session falls back to default
    behaviour (emit ``turn.error`` for unknown messages, ignore
    settings updates, etc.).

    Attributes:
        on_unknown_message: Called when no built-in handler matches
            the message type. Payload is the full message dict.
        on_settings_update: Called for ``settings.update`` messages.
        on_status_request: Called after status request is handled.
            Payload includes ``kind`` and ``statuses``.
        on_before_provider_reload: Called before a provider reload
            with the old bundle and new provider config.
        on_after_provider_reload: Called after a provider reload
            with the old and new bundles.
        on_event: Called for every :class:`FrameworkEvent` the
            session emits, before the transport sends it. Apps can
            use this for logging or filtering.
    """

    on_unknown_message: HookFn | None = None
    on_settings_update: HookFn | None = None
    on_status_request: HookFn | None = None
    on_before_provider_reload: Callable[..., Awaitable[None]] | None = None
    on_after_provider_reload: Callable[..., Awaitable[None]] | None = None
    on_event: Callable[..., Awaitable[None]] | None = None


class WebSocketSession:
    """Reusable WebSocket voice-chat session.

    Owns the full runtime (bundle, pipeline, collector, transport,
    sink) and exposes :meth:`handle_message` for each inbound
    WebSocket event.

    Example::

        session = WebSocketSession(transport)

        while True:
            packet = await websocket.receive()
            if packet.get("type") == "websocket.disconnect":
                break
            if packet.get("bytes") is not None:
                await session.handle_message(packet["bytes"])
            elif packet.get("text") is not None:
                await session.handle_message(json.loads(packet["text"]))
    """

    def __init__(
        self,
        transport: Transport,
        *,
        config: WebSocketSessionConfig | None = None,
        hooks: WebSocketSessionHooks | None = None,
    ) -> None:
        self.config = config or WebSocketSessionConfig()
        self.hooks = hooks or WebSocketSessionHooks()

        self.transport = transport
        self.sink = TransportEventSink(transport)

        provider_config = self.config.provider_config or {
            "vad": {"provider": "mock"},
            "asr": {"provider": "mock"},
            "llm": {"provider": "mock"},
            "tts": {"provider": "mock"},
        }
        collector_config = self.config.collector_config or UtteranceCollectorConfig()
        self.bundle: ProviderBundle = build_provider_bundle(provider_config)
        self.pipeline = SpeechPipeline(
            self.bundle,
            self.sink,
            self.config.pipeline_config or PipelineConfig(),
        )

        async def utterance_callback(pcm: bytes, sample_rate: int, mode: str) -> None:
            await self.pipeline.handle_audio_turn(pcm, sample_rate, mode=mode)

        async def cancel_callback(reason: str) -> None:
            await self.pipeline.cancel_tts(reason)

        self.collector = AudioUtteranceCollector(
            vad_provider=self.bundle.vad,
            event_sink=self.sink,
            utterance_callback=utterance_callback,
            config=collector_config,
            cancel_callback=cancel_callback,
        )
        self.frame_stats = AudioFrameStats(
            expected_sample_rate=collector_config.sample_rate,
            expected_channels=collector_config.channels,
            expected_frame_ms=collector_config.frame_ms,
        )

    # ------------------------------------------------------------------
    # Message routing
    # ------------------------------------------------------------------

    async def handle_message(
        self, message: dict[str, Any] | bytes | bytearray | memoryview
    ) -> None:
        """Route one inbound message to the appropriate handler.

        Built-in message types:

        * Binary v1 packet — raw PCM audio data from the client mic.
        * ``audio.frame`` — legacy JSON/base64 mic audio data.
        * ``text.turn`` — text input (non-audio path).
        * ``conversation.clear`` — reset conversation history.
        * ``tts.cancel`` — interrupt active TTS playback.
        * ``status.request`` — request current provider statuses.
        * ``settings.update`` — routed to ``on_settings_update`` hook.
        * ``providers.reload`` — reload providers from ``payload.config``.
        """
        if isinstance(message, (bytes, bytearray, memoryview)):
            await self._handle_binary_audio_frame(message)
            return

        message_type = str(message.get("type", ""))
        payload: dict[str, Any] = dict(message.get("payload", {}) or {})
        mode = str(payload.pop("mode", self.config.default_mode))

        if message_type == "audio.frame":
            await self._handle_audio_frame(payload, mode=mode)
            return

        if message_type == "text.turn":
            text = str(payload.get("text", ""))
            if text:
                await self.pipeline.handle_text_turn(text, mode=mode)
            return

        if message_type == "conversation.clear":
            await self.pipeline.clear_conversation(mode=mode)
            return

        if message_type == "tts.cancel":
            reason = str(payload.get("reason", "client_cancelled"))
            await self.pipeline.cancel_tts(reason)
            return

        if message_type == "status.request":
            await self._handle_status_request(payload)
            return

        if message_type == "settings.update":
            if self.hooks.on_settings_update:
                await self.hooks.on_settings_update(self, payload)
            return

        if message_type == "providers.reload":
            new_config = dict(payload.get("config", {}))
            load = bool(payload.get("load", False))
            await self._reload_providers(new_config, load=load)
            return

        # Unknown message type — try hook, then emit error.
        if self.hooks.on_unknown_message:
            await self.hooks.on_unknown_message(self, message)
        else:
            await self._send_event(
                "turn.error", message=f"unknown message type: {message_type}"
            )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def emit_status(self, kind: str = "probe") -> None:
        """Probe and emit current provider statuses."""
        await self._handle_status_request({"kind": kind})

    async def _handle_status_request(self, payload: dict[str, Any]) -> None:
        kind = str(
            payload.get("kind", "probe" if self.config.auto_probe_status else "check")
        )

        if kind == "probe":
            statuses = await self.bundle.probe_statuses()
        elif kind == "check":
            statuses = await self.bundle.check_statuses()
        elif kind == "load":
            statuses = await self.bundle.load_statuses()
        else:
            statuses = await self.bundle.probe_statuses()

        event = FrameworkEvent("providers.status", {"statuses": statuses})

        if self.hooks.on_status_request:
            await self.hooks.on_status_request(
                self, {"kind": kind, "statuses": statuses}
            )

        await self._send_event(event.type, **event.payload)

    # ------------------------------------------------------------------
    # Provider reload
    # ------------------------------------------------------------------

    async def reload_providers(
        self, config: dict[str, dict[str, Any]], *, load: bool = False
    ) -> None:
        """Rebuild provider bundle and swap into pipeline/collector.

        Args:
            config: Provider configuration dict for
                :func:`build_provider_bundle`.
            load: If True, run ``load_statuses()`` on the new bundle
                after swapping (loads heavy models).
        """
        await self._reload_providers(config, load=load)

    async def _reload_providers(
        self, config: dict[str, dict[str, Any]], *, load: bool = False
    ) -> None:
        old_bundle = self.bundle

        if self.hooks.on_before_provider_reload:
            await self.hooks.on_before_provider_reload(self, old_bundle, config)

        new_bundle = build_provider_bundle(config)
        await self.pipeline.update_providers(new_bundle, reason="provider_reload")
        self.bundle = new_bundle

        # Swap the VAD provider in the collector
        self.collector.update_vad_provider(new_bundle.vad)

        if load:
            await new_bundle.load_statuses()

        if self.hooks.on_after_provider_reload:
            await self.hooks.on_after_provider_reload(self, old_bundle, new_bundle)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _handle_audio_frame(self, payload: dict[str, Any], mode: str) -> None:
        try:
            frame = parse_audio_frame(payload, self.frame_stats)
        except ValueError as exc:
            await self._send_event("audio.frame_error", message=str(exc))
            return
        await self.collector.ingest_frame(frame, mode=mode)

    async def _handle_binary_audio_frame(
        self, packet: bytes | bytearray | memoryview
    ) -> None:
        try:
            frame, mode = parse_binary_audio_frame(packet, self.frame_stats)
        except ValueError as exc:
            await self._send_event("audio.frame_error", message=str(exc))
            return
        await self.collector.ingest_frame(
            frame, mode=mode or self.config.default_mode
        )

    async def _send_event(self, event_type: str, **payload: Any) -> None:
        """Emit a framework event, routing through ``on_event`` hook if set."""
        event = FrameworkEvent(event_type, payload)
        if self.hooks.on_event:
            await self.hooks.on_event(self, event)
        await self.transport.send_event(event)
