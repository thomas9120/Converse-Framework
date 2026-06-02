"""Text-only chat example showing the framework as a second consumer.

This example runs a real conversation against :class:`SpeechPipeline`
using only the framework's public API. No harness modules, FastAPI,
WebSocket, profile files, or browser UI are involved. It is intended
as a quick smoke test that the extracted package can drive a complete
turn loop on its own.

The example exposes two surfaces:

* :func:`run_text_chat` — async driver that runs a list of scripted
  inputs through the pipeline. Used by the test suite.
* ``__main__`` — a small CLI that reads lines from stdin, prints LLM
  tokens as they stream, and summarizes each turn's events.

The CLI uses mock providers by default. To try a real provider, pass
``--asr``, ``--llm``, ``--tts`` with a registered provider name and
install the matching extra. If the extra is missing, the framework
falls back to :class:`UnavailableProvider` and the status message
will tell the user which extra to install.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from converse_framework.events import EventSink, QueueEventSink
from converse_framework.pipeline import PipelineConfig, SpeechPipeline
from converse_framework.registry import build_provider_bundle


@dataclass
class TextChatExampleConfig:
    """User-facing configuration for the text chat example."""

    providers: dict[str, str] = field(
        default_factory=lambda: {
            "vad": "mock",
            "asr": "mock",
            "llm": "mock",
            "tts": "mock",
        }
    )
    tts_chunk_chars: int = 80
    min_tts_chars: int = 0
    mode: str = "chat"
    system_prompt: str = "You are a helpful assistant. Be concise."


def build_example_bundle(config: TextChatExampleConfig) -> Any:
    """Build a provider bundle from the example config.

    Each section is forwarded to the framework registry. Real provider
    names are honored when the matching optional dependency is
    installed; otherwise the registry returns an unavailable provider
    whose status message tells the user which extra to install.
    """
    nested: dict[str, dict[str, Any]] = {}
    for kind, name in config.providers.items():
        nested[kind] = {"provider": name}
    return build_provider_bundle(nested)


def build_example_pipeline(
    config: TextChatExampleConfig | None = None,
    *,
    sink: EventSink | None = None,
) -> tuple[SpeechPipeline, Any]:
    """Construct a :class:`SpeechPipeline` ready for the example.

    Returns the pipeline and the provider bundle it was built with so
    callers can inspect ``bundle.statuses()`` if they want to.
    """
    config = config or TextChatExampleConfig()
    bundle = build_example_bundle(config)
    sink = sink or QueueEventSink(asyncio.Queue())
    pipeline = SpeechPipeline(
        providers=bundle,
        sink=sink,
        config=PipelineConfig(
            tts_chunk_chars=config.tts_chunk_chars,
            min_tts_chars=config.min_tts_chars,
            default_mode=config.mode,
        ),
    )
    if config.system_prompt:
        pipeline.set_system_prompt(config.system_prompt, mode=config.mode)
    return pipeline, bundle


async def run_text_chat(
    inputs: Iterable[str],
    config: TextChatExampleConfig | None = None,
) -> dict[str, Any]:
    """Drive the example end-to-end and return a structured summary.

    The summary contains the per-turn event types, the LLM text per
    turn, and the number of TTS audio chunks produced per turn. It is
    designed to be assertion-friendly for the framework test suite.
    """
    config = config or TextChatExampleConfig()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    pipeline, _ = build_example_pipeline(config, sink=QueueEventSink(queue))

    turns: list[dict[str, Any]] = []
    for text in inputs:
        await pipeline.handle_text_turn(text, mode=config.mode)
        # Drain events emitted during the turn before recording them so
        # concurrent TTS tasks finish flushing.
        await _drain_in_flight_tts(pipeline)
        events = await _drain_queue(queue)
        turns.append(
            {
                "input": text,
                "events": [event["type"] for event in events],
                "llm_text": _joined_llm_text(events),
                "tts_audio_chunks": sum(
                    1 for event in events if event["type"] == "tts.audio"
                ),
            }
        )
    return {
        "mode": config.mode,
        "providers": config.providers,
        "turns": turns,
        "messages": pipeline.messages_for_mode(config.mode),
    }


async def _drain_in_flight_tts(pipeline: SpeechPipeline) -> None:
    """Wait for any pending TTS tasks to finish so events are settled."""
    active = list(pipeline.state.active_tts_tasks)
    if not active:
        return
    await asyncio.gather(*active, return_exceptions=True)


async def _drain_queue(queue: asyncio.Queue[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


def _joined_llm_text(events: list[dict[str, Any]]) -> str:
    pieces: list[str] = []
    for event in events:
        if event["type"] == "llm.token":
            payload = event.get("payload", {})
            token = payload.get("text")
            if isinstance(token, str):
                pieces.append(token)
    return "".join(pieces).strip()


def _format_audio_summary(audio_b64: str, mime_type: str) -> str:
    """Best-effort one-line summary for a TTS audio payload."""
    try:
        decoded = base64.b64decode(audio_b64.encode("ascii"))
    except Exception:
        return f"[tts.audio] {mime_type} (decode-error)"
    return f"[tts.audio] {mime_type} {len(decoded)} bytes"


def _format_event_for_cli(event: dict[str, Any]) -> str:
    event_type = event.get("type", "")
    payload = event.get("payload", {}) or {}
    if event_type == "llm.token":
        return f"  token: {payload.get('text', '')!r}"
    if event_type == "asr.transcript":
        marker = "final" if payload.get("final") else "partial"
        return f"  asr({marker}): {payload.get('text', '')!r}"
    if event_type == "tts.audio":
        return "  " + _format_audio_summary(
            str(payload.get("data", "")), str(payload.get("mime_type", "audio/wav"))
        )
    if event_type == "turn.finished":
        return "  turn.finished"
    if event_type == "turn.error":
        return f"  turn.error: {payload.get('message', '')}"
    return f"  {event_type}"


_DEFAULT_PROVIDER_NAMES: dict[str, str] = {
    "vad": "mock",
    "asr": "mock",
    "llm": "mock",
    "tts": "mock",
}


def _parse_provider_args(values: Iterable[str]) -> dict[str, str]:
    """Parse ``--provider KIND=NAME`` style CLI arguments.

    Returns a complete provider name map (vad/asr/llm/tts) with each
    kind defaulting to ``mock`` when no override is supplied. The
    text-only example does not exercise voice activity detection, so
    callers typically leave the VAD default alone.
    """
    parsed: dict[str, str] = dict(_DEFAULT_PROVIDER_NAMES)
    for entry in values:
        if "=" not in entry:
            raise SystemExit(
                f"Expected --provider-style argument of the form KIND=NAME, got {entry!r}"
            )
        kind, name = entry.split("=", 1)
        kind = kind.strip()
        name = name.strip()
        if kind not in {"vad", "asr", "llm", "tts"}:
            raise SystemExit(f"Unknown provider kind: {kind}")
        if not name:
            raise SystemExit(f"Provider name is empty for kind {kind}")
        parsed[kind] = name
    return parsed


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m converse_framework.examples.text_chat",
        description=(
            "Run a text conversation against Converse Framework with mock "
            "providers by default. Use KIND=NAME overrides to select a real "
            "provider when the matching extra is installed."
        ),
    )
    parser.add_argument(
        "--provider",
        action="append",
        default=[],
        metavar="KIND=NAME",
        help="Override a provider, e.g. --provider asr=faster-whisper",
    )
    parser.add_argument(
        "--tts-chunk-chars",
        type=int,
        default=80,
        help="Flush TTS once the LLM has produced this many characters.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful assistant. Be concise.",
        help="Initial system prompt.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="chat",
        help="Conversation mode key (default: chat).",
    )
    return parser


async def _run_cli(args: argparse.Namespace) -> int:
    config = TextChatExampleConfig(
        providers=_parse_provider_args(args.provider),
        tts_chunk_chars=args.tts_chunk_chars,
        min_tts_chars=0,
        mode=args.mode,
        system_prompt=args.system_prompt,
    )
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    pipeline, bundle = build_example_pipeline(config, sink=QueueEventSink(queue))

    print(f"providers: {config.providers}")
    for status in bundle.statuses():
        print(f"  - {status['kind']}/{status['name']}: {status['message']}")
    print("type 'quit' to exit, 'clear' to reset history")
    print("-" * 60)

    loop = asyncio.get_running_loop()
    while True:
        try:
            line = await loop.run_in_executor(None, lambda: input("you> "))
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        text = line.strip()
        if not text:
            continue
        if text in {"quit", "exit"}:
            return 0
        if text == "clear":
            await pipeline.clear_conversation(mode=config.mode)
            print("(conversation cleared)")
            continue
        await pipeline.handle_text_turn(text, mode=config.mode)
        await _drain_in_flight_tts(pipeline)
        events = await _drain_queue(queue)
        for event in events:
            print(_format_event_for_cli(event))
        print("-" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_run_cli(args))


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI smoke test
    sys.exit(main())
