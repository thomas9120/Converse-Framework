"""Voice chat example (manual).

This example shows the framework's voice flow: a :class:`AudioUtteranceCollector`
feeds PCM frames into the :class:`SpeechPipeline` after detecting speech
end. It is the recommended starting point for any consumer that wants
to build a voice assistant on top of the framework.

The example is **manual** by design — it requires either a working
microphone or a WAV file to read from, and it is not exercised by the
automated test suite. The text-only example in
:mod:`converse_framework.examples.text_chat` is the one covered by
tests.

Usage
-----

Run the voice example from the repository root after installing the
optional ``silero`` and ``faster-whisper`` extras::

    python -m converse_framework.examples.voice_chat

The example will:

1. Build a provider bundle (``silero`` VAD, ``faster-whisper`` ASR,
   ``llamacpp`` LLM, ``kokoro`` TTS by default).
2. Open a microphone stream at 16 kHz mono (or read from a file when
   ``--input path/to/file.wav`` is passed).
3. Feed 30 ms PCM frames into the utterance collector.
4. For each completed utterance, hand the PCM bytes to
   :meth:`SpeechPipeline.handle_audio_turn`.
5. Stream ``tts.audio`` events to the default audio output.

Implementation notes
--------------------

* The collector's :attr:`cancel_callback` is wired to
  :meth:`SpeechPipeline.cancel_tts` so VAD-driven speech starts cancel
  any in-flight TTS (barge-in).
* The collector's ``pre_speech_start_hook`` is a no-op here, but it
  is where a real consumer would update a per-frame system prompt
  before the utterance is finalized.
* ``--mock`` swaps the VAD/ASR providers for the in-process mock
  providers so the example can run without any heavy model
  dependencies.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from converse_framework.audio_utils import AudioFrame, AudioFrameStats, parse_audio_frame
from converse_framework.events import QueueEventSink
from converse_framework.pipeline import PipelineConfig, SpeechPipeline
from converse_framework.registry import build_provider_bundle
from converse_framework.utterance_collector import (
    AudioUtteranceCollector,
    UtteranceCollectorConfig,
)


def _parse_provider_args(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {"vad": "silero", "asr": "faster-whisper", "llm": "llamacpp", "tts": "kokoro"}
    for entry in values:
        if "=" not in entry:
            raise SystemExit(f"Expected KIND=NAME, got {entry!r}")
        kind, name = entry.split("=", 1)
        if kind not in parsed:
            raise SystemExit(f"Unknown provider kind: {kind}")
        parsed[kind] = name.strip()
    return parsed


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m converse_framework.examples.voice_chat",
        description=(
            "Manual voice example. Streams microphone (or WAV file) frames "
            "through the framework's utterance collector and pipeline. "
            "Run --mock to avoid heavy provider dependencies."
        ),
    )
    parser.add_argument(
        "--provider",
        action="append",
        default=[],
        metavar="KIND=NAME",
        help="Override a provider, e.g. --provider tts=pocket-tts",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock VAD/ASR providers so the example runs without extras.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Read PCM frames from a 16 kHz mono WAV file instead of the mic.",
    )
    parser.add_argument(
        "--frame-ms",
        type=int,
        default=30,
        help="Frame size in milliseconds (default 30).",
    )
    parser.add_argument(
        "--pre-speech-ms",
        type=int,
        default=450,
        help="Pre-speech buffer size in milliseconds (default 450).",
    )
    return parser


async def _drive_collector(
    collector: AudioUtteranceCollector,
    pipeline: SpeechPipeline,
    config: UtteranceCollectorConfig,
    *,
    input_path: str | None = None,
) -> None:
    """Read PCM frames from the mic or a file and feed the collector.

    Real deployments should plug a microphone capture coroutine in
    here. The framework only cares that ``parse_audio_frame`` succeeds
    and that the resulting :class:`AudioFrame` is passed to
    :meth:`AudioUtteranceCollector.ingest_frame`.
    """
    if input_path is None:
        raise SystemExit(
            "No --input file provided. Pass a 16 kHz mono WAV file to drive the "
            "collector. (Live microphone capture is intentionally out of scope "
            "for the example; the framework is platform-agnostic.)"
        )

    stats = AudioFrameStats(
        expected_sample_rate=config.sample_rate,
        expected_channels=config.channels,
        expected_frame_ms=config.frame_ms,
    )
    expected_frame_bytes = config.bytes_per_ms * config.frame_ms

    with open(input_path, "rb") as handle:
        # Skip the 44-byte WAV header. The example assumes a bare
        # PCM_s16le body; production code should use ``wave`` instead.
        handle.read(44)
        while True:
            chunk = handle.read(expected_frame_bytes)
            if not chunk:
                break
            frame = parse_audio_frame(
                {
                    "data": chunk.hex(),
                    "sample_rate": config.sample_rate,
                    "channels": config.channels,
                    "frame_ms": config.frame_ms,
                },
                stats,
            )
            await collector.ingest_frame(frame)


async def _main_async(args: argparse.Namespace) -> int:
    providers = _parse_provider_args(args.provider)
    if args.mock:
        providers = {"vad": "mock", "asr": "mock", "llm": "mock", "tts": "mock"}

    queue: asyncio.Queue = asyncio.Queue()
    sink = QueueEventSink(queue)
    bundle = build_provider_bundle({kind: {"provider": name} for kind, name in providers.items()})
    pipeline = SpeechPipeline(
        providers=bundle,
        sink=sink,
        config=PipelineConfig(tts_chunk_chars=80),
    )
    collector_config = UtteranceCollectorConfig(
        sample_rate=16000,
        channels=1,
        frame_ms=args.frame_ms,
        pre_speech_ms=args.pre_speech_ms,
    )

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

    print(f"providers: {providers}")
    print(f"frame_ms={args.frame_ms}  pre_speech_ms={args.pre_speech_ms}")
    print("-" * 60)

    await _drive_collector(collector, pipeline, collector_config, input_path=args.input)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_main_async(args))


if __name__ == "__main__":  # pragma: no cover - manual example
    sys.exit(main())
