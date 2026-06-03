"""FastAPI WebSocket voice-chat recipe.

This example shows the browser-oriented wire shape: clients send JSON
messages containing ``audio.frame`` payloads, the framework validates
them with :func:`parse_audio_frame`, and all framework events are sent
back over the same WebSocket.

FastAPI is imported only by :func:`create_app`, so importing this module
does not add a dependency to the base framework package. To run it::

    pip install fastapi uvicorn
    uvicorn converse_framework.examples.websocket_voice_chat:create_app --factory

The mock providers are used by default. Pass a provider config to
:func:`build_websocket_voice_runtime` when embedding this recipe in a
real app.

.. seealso::

   :class:`converse_framework.session.WebSocketSession` provides a
   reusable message-dispatch loop that replaces the per-endpoint
   routing in this recipe.  See the WebSocket Session Helper section
   in the README.

.. note::

   Mobile browser microphone access (``getUserMedia``) requires a
   **secure context** — HTTPS, ``localhost``, or ``127.0.0.1``.
   Over a plain ``http://<lan-ip>`` page, ``getUserMedia`` will be
   rejected on mobile browsers.  See the "Mobile Browser Microphone
   Testing" section in the README for tunnel and HTTPS recipes.
   The WebSocket URL for tunneled setups changes from
   ``ws://<host>/ws`` to ``wss://<tunnel-host>/ws``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from converse_framework.audio_utils import AudioFrameStats, parse_audio_frame
from converse_framework.events import FrameworkEvent, TransportEventSink
from converse_framework.pipeline import PipelineConfig, SpeechPipeline
from converse_framework.registry import build_provider_bundle
from converse_framework.transport import Transport
from converse_framework.utterance_collector import (
    AudioUtteranceCollector,
    UtteranceCollectorConfig,
)


@dataclass
class WebSocketVoiceRuntime:
    """Objects needed to drive voice frames from a WebSocket."""

    pipeline: SpeechPipeline
    collector: AudioUtteranceCollector
    frame_stats: AudioFrameStats


class WebSocketTransport(Transport):
    """Minimal transport adapter for FastAPI-compatible WebSockets."""

    def __init__(self, websocket) -> None:
        self.websocket = websocket

    async def send_event(self, event: FrameworkEvent) -> None:
        await self.websocket.send_json(event.to_json())

    async def receive_event(self) -> FrameworkEvent:
        message = await self.websocket.receive_json()
        return FrameworkEvent(
            type=str(message.get("type", "")),
            payload=dict(message.get("payload", {}) or {}),
            ts=float(message.get("ts", 0.0)),
        )


def build_websocket_voice_runtime(
    transport: Transport,
    *,
    provider_config: dict[str, dict[str, Any]] | None = None,
    collector_config: UtteranceCollectorConfig | None = None,
    pipeline_config: PipelineConfig | None = None,
) -> WebSocketVoiceRuntime:
    """Build the pipeline and collector used by the WebSocket handler."""
    provider_config = provider_config or {
        "vad": {"provider": "mock"},
        "asr": {"provider": "mock"},
        "llm": {"provider": "mock"},
        "tts": {"provider": "mock"},
    }
    collector_config = collector_config or UtteranceCollectorConfig()
    sink = TransportEventSink(transport)
    bundle = build_provider_bundle(provider_config)
    pipeline = SpeechPipeline(bundle, sink, pipeline_config or PipelineConfig())

    async def utterance_callback(pcm: bytes, sample_rate: int, mode: str) -> None:
        await pipeline.handle_audio_turn(pcm, sample_rate, mode=mode)

    async def cancel_callback(reason: str) -> None:
        await pipeline.cancel_tts(reason)

    collector = AudioUtteranceCollector(
        vad_provider=bundle.vad,
        event_sink=sink,
        utterance_callback=utterance_callback,
        config=collector_config,
        cancel_callback=cancel_callback,
    )
    frame_stats = AudioFrameStats(
        expected_sample_rate=collector_config.sample_rate,
        expected_channels=collector_config.channels,
        expected_frame_ms=collector_config.frame_ms,
    )
    return WebSocketVoiceRuntime(pipeline, collector, frame_stats)


async def handle_websocket_message(
    runtime: WebSocketVoiceRuntime,
    transport: Transport,
    message: dict[str, Any],
) -> None:
    """Handle one client message from the WebSocket recipe."""
    message_type = str(message.get("type", ""))
    payload = dict(message.get("payload", {}) or {})
    mode = str(payload.pop("mode", "chat"))

    if message_type == "audio.frame":
        try:
            frame = parse_audio_frame(payload, runtime.frame_stats)
        except ValueError as exc:
            await transport.send_event(
                FrameworkEvent("audio.frame_error", {"message": str(exc)})
            )
            return
        await runtime.collector.ingest_frame(frame, mode=mode)
        return

    if message_type == "text.turn":
        text = str(payload.get("text", ""))
        if text:
            await runtime.pipeline.handle_text_turn(text, mode=mode)
        return

    if message_type == "conversation.clear":
        await runtime.pipeline.clear_conversation(mode=mode)
        return

    await transport.send_event(
        FrameworkEvent(
            "turn.error", {"message": f"unknown message type: {message_type}"}
        )
    )


def create_app():
    """Create a tiny FastAPI app exposing ``/ws`` for voice chat."""
    from fastapi import FastAPI, WebSocket

    app = FastAPI()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        transport = WebSocketTransport(websocket)
        runtime = build_websocket_voice_runtime(transport)
        while True:
            message = await websocket.receive_json()
            await handle_websocket_message(runtime, transport, message)
            await asyncio.sleep(0)

    return app
