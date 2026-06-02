"""Performance benchmark for the Converse Framework turn-orchestration path.

This script measures four latency metrics on the framework's pipeline
plumbing using only the framework's own abstractions:

1. **First token latency** -- ``turn.started`` -> ``llm.first_token``
2. **First TTS chunk latency** -- ``llm.first_token`` -> ``tts.first_chunk``
3. **Speech start to ASR start** -- ``vad.speech_start`` -> first
   ``asr.transcript``
4. **Barge-in cancellation latency** -- ``vad.speech_start`` (barge-in)
   -> ``tts.cancelled``

The benchmark is **mock-only**. It uses
``converse_framework.providers.mock`` for VAD, ASR, LLM, and TTS, plus
the framework's own ``QueueEventSink`` and ``QueueTransport`` to avoid
any real I/O. The numbers it reports measure framework-level plumbing
(event ordering, scheduling, task management) -- not provider-level
work. Real providers will show different absolute numbers but the
framework overhead should stay comparable.

Why mock-only for v0.1: the framework's performance story is about the
pipeline plumbing, not the providers. Mock providers give stable,
reproducible numbers; real providers will vary by machine. The
harness-vs-framework comparison is left for a future real-provider
run.

Extending to real providers
---------------------------

To benchmark the same four metrics against real providers, swap the
provider names in ``MOCK_PROVIDER_CONFIG_FAST`` /
``MOCK_PROVIDER_CONFIG_BARGE_IN`` (e.g. ``"asr": "faster-whisper"``,
``"llm": "llamacpp"``, ``"tts": "kokoro"``) and install the matching
optional extra via ``pip install "converse-framework[<extra>]"``. The
metric definitions and the event-watching pattern do not need to
change. Real providers will produce substantially larger numbers for
metrics 1-3 (first-token / first-TTS-chunk / speech-to-ASR); metric 4
(barge-in cancellation) will still be dominated by asyncio scheduling
overhead unless a real TTS provider does significant synchronous
cleanup work.

Usage::

    python benchmarks/perf_compare.py --iterations 50
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from typing import Any

from converse_framework import (
    AudioFrame,
    AudioUtteranceCollector,
    FrameworkEvent,
    PipelineConfig,
    ProviderBundle,
    QueueEventSink,
    QueueTransport,
    SpeechPipeline,
    VADEvent,
    build_provider_bundle,
)
from converse_framework.protocols import ProviderCapabilities, ProviderStatus


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------
#
# Two configurations are used:
# - ``MOCK_PROVIDER_CONFIG_FAST`` for metrics 1-3. TTS delays are short
#   so a turn completes quickly and the iteration stays cheap.
# - ``MOCK_PROVIDER_CONFIG_BARGE_IN`` for metric 4. TTS delays are
#   stretched out so a TTS task is still alive when the VAD fires
#   ``vad.speech_start`` and the barge-in can actually land.

MOCK_PROVIDER_CONFIG_FAST: dict[str, dict[str, Any]] = {
    "vad": {"provider": "mock"},
    "asr": {"provider": "mock"},
    "llm": {"provider": "mock", "first_token_delay_ms": 5, "token_delay_ms": 1},
    "tts": {"provider": "mock", "first_chunk_delay_ms": 5, "chunk_delay_ms": 1},
}

MOCK_PROVIDER_CONFIG_BARGE_IN: dict[str, dict[str, Any]] = {
    "vad": {"provider": "mock"},
    "asr": {"provider": "mock"},
    "llm": {"provider": "mock", "first_token_delay_ms": 5, "token_delay_ms": 1},
    "tts": {"provider": "mock", "first_chunk_delay_ms": 50, "chunk_delay_ms": 500},
}

# User input. Long enough to flush at least one TTS chunk; short enough
# to keep the per-iteration cost low.
TURN_INPUT = (
    "Hello framework, please share a brief thoughtful reply with two sentences."
)


METRIC_LABELS: dict[str, str] = {
    "first_token_latency_ms": (
        "First token (turn.started -> llm.first_token)"
    ),
    "first_tts_chunk_latency_ms": (
        "First TTS chunk (llm.first_token -> tts.first_chunk)"
    ),
    "speech_to_asr_latency_ms": (
        "Speech start -> ASR start (vad.speech_start -> asr.transcript)"
    ),
    "barge_in_cancel_latency_ms": (
        "Barge-in cancel (vad.speech_start -> tts.cancelled)"
    ),
}


# ---------------------------------------------------------------------------
# Event recording plumbing
# ---------------------------------------------------------------------------
#
# ``EventRecorder`` wires the pipeline's ``QueueEventSink`` to a
# ``QueueTransport`` via a background "wire" task, and a "watcher" task
# pulls from ``transport.receive_event()`` to record when each event
# was first observed. This mirrors the production pattern of the
# framework: the pipeline emits to a sink, a transport forwards events
# to a client, and the client measures latency.
#
# The two background tasks each yield to the event loop
# (``await asyncio.sleep(0)``) at well-defined points so the loop has
# a chance to schedule. This keeps the recorded timestamps consistent
# with the framework's own latency fields (which also rely on the
# loop scheduler).


class EventRecorder:
    """Capture pipeline events through a (QueueEventSink -> QueueTransport) chain."""

    def __init__(self) -> None:
        self._source_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.sink = QueueEventSink(self._source_queue)
        self.transport = QueueTransport()
        self._log: list[tuple[str, int, dict[str, Any]]] = []
        self._wire_stop = asyncio.Event()
        self._watcher_stop = asyncio.Event()
        self._wire_task: asyncio.Task[None] | None = None
        self._watcher_task: asyncio.Task[None] | None = None

    async def _wire(self) -> None:
        """Forward events from the sink's queue onto the transport's recv queue."""
        while not self._wire_stop.is_set():
            try:
                event = await asyncio.wait_for(
                    self._source_queue.get(), timeout=0.05
                )
            except asyncio.TimeoutError:
                continue
            await asyncio.sleep(0)  # let the loop schedule other tasks
            await self.transport._recv_queue.put(
                FrameworkEvent(
                    type=event["type"],
                    payload=event.get("payload", {}),
                    ts=event.get("ts", 0.0),
                )
            )
        # Drain any remaining source events so the watcher can record them.
        while True:
            try:
                event = self._source_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            await self.transport._recv_queue.put(
                FrameworkEvent(
                    type=event["type"],
                    payload=event.get("payload", {}),
                    ts=event.get("ts", 0.0),
                )
            )

    async def _watcher(self) -> None:
        """Read events from the transport and record observed timestamps."""
        while not self._watcher_stop.is_set():
            try:
                event = await asyncio.wait_for(
                    self.transport.receive_event(), timeout=0.05
                )
            except asyncio.TimeoutError:
                continue
            await asyncio.sleep(0)  # let the loop schedule other tasks
            self._log.append((event.type, time.perf_counter_ns(), event.payload))
        # Drain any remaining transport events so nothing is lost.
        while True:
            try:
                event = self.transport._recv_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._log.append((event.type, time.perf_counter_ns(), event.payload))

    def start(self) -> None:
        """Spawn the wire + watcher background tasks."""
        self._wire_task = asyncio.create_task(self._wire())
        self._watcher_task = asyncio.create_task(self._watcher())

    async def stop(self) -> None:
        """Stop the background tasks and flush any remaining events."""
        self._wire_stop.set()
        self._watcher_stop.set()
        if self._wire_task is not None:
            await self._wire_task
        if self._watcher_task is not None:
            await self._watcher_task

    def first_event(self, event_type: str) -> int | None:
        """Return the first observed timestamp (perf_counter_ns) for ``event_type``."""
        for ev_type, ts, _ in self._log:
            if ev_type == event_type:
                return ts
        return None


# ---------------------------------------------------------------------------
# VAD test double for barge-in
# ---------------------------------------------------------------------------


class _OnDemandVAD:
    """VAD that returns scripted events on demand; nothing for other frames."""

    def __init__(self) -> None:
        self._scripted: list[list[VADEvent]] = []

    def arm(self, *events: VADEvent) -> None:
        """Queue a list of VAD events to be returned on the next process_frame call."""
        self._scripted.append(list(events))

    @property
    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name="bench-vad",
            kind="vad",
            ready=True,
            message="Benchmark VAD fires scripted events on demand.",
            capabilities=ProviderCapabilities(supports_barge_in=True),
        )

    async def check_status(self) -> ProviderStatus:
        return self.status

    async def process_frame(self, frame: AudioFrame) -> list[VADEvent]:
        if self._scripted:
            await asyncio.sleep(0)  # let the loop schedule other tasks
            return self._scripted.pop(0)
        return []

    async def unload(self) -> ProviderStatus:
        return self.status


def _make_audio_frame(sequence: int = 0, samples: int = 480) -> AudioFrame:
    """Create a tiny silent ``AudioFrame`` (30 ms mono at 16 kHz)."""
    return AudioFrame(
        data=b"\x00\x00" * samples,
        sequence=sequence,
        sample_rate=16000,
        channels=1,
        frame_ms=30,
        encoding="pcm_s16le",
    )


# ---------------------------------------------------------------------------
# Per-iteration measurement
# ---------------------------------------------------------------------------


async def _wait_for_inactive_tts(
    pipeline: SpeechPipeline, max_iterations: int = 20
) -> None:
    """Yield until the pipeline's active TTS task set is empty (or the budget runs out)."""
    for _ in range(max_iterations):
        if not pipeline.state.active_tts_tasks:
            return
        await asyncio.sleep(0.01)
    # Final yield in case there is one straggler
    await asyncio.sleep(0)


async def measure_text_turn_metrics() -> dict[str, float]:
    """Run one text turn and record latencies for the first three metrics."""
    recorder = EventRecorder()
    bundle: ProviderBundle = build_provider_bundle(MOCK_PROVIDER_CONFIG_FAST)
    pipeline = SpeechPipeline(
        bundle, recorder.sink, PipelineConfig(tts_chunk_chars=60)
    )
    recorder.start()
    try:
        await pipeline.handle_text_turn(TURN_INPUT)
        await _wait_for_inactive_tts(pipeline)
        await asyncio.sleep(0)
    finally:
        await recorder.stop()

    turn_started = recorder.first_event("turn.started")
    vad_speech_start = recorder.first_event("vad.speech_start")
    asr_transcript = recorder.first_event("asr.transcript")
    llm_first_token = recorder.first_event("llm.first_token")
    tts_first_chunk = recorder.first_event("tts.first_chunk")

    metrics: dict[str, float] = {}
    if turn_started is not None and llm_first_token is not None:
        metrics["first_token_latency_ms"] = (
            llm_first_token - turn_started
        ) / 1e6
    if llm_first_token is not None and tts_first_chunk is not None:
        metrics["first_tts_chunk_latency_ms"] = (
            tts_first_chunk - llm_first_token
        ) / 1e6
    if vad_speech_start is not None and asr_transcript is not None:
        metrics["speech_to_asr_latency_ms"] = (
            asr_transcript - vad_speech_start
        ) / 1e6
    return metrics


async def measure_barge_in_metric() -> float:
    """Measure cancellation latency: barge-in ``vad.speech_start`` -> ``tts.cancelled``.

    The barge-in test uses a separate pipeline with stretched-out TTS
    delays so a TTS task is still alive when the VAD fires
    ``vad.speech_start``. The collector's ``cancel_callback`` invokes
    ``pipeline.cancel_tts``, which cancels the active TTS task and
    emits ``tts.cancelled``.
    """
    recorder = EventRecorder()
    bundle: ProviderBundle = build_provider_bundle(MOCK_PROVIDER_CONFIG_BARGE_IN)
    pipeline = SpeechPipeline(
        bundle, recorder.sink, PipelineConfig(tts_chunk_chars=60)
    )
    vad = _OnDemandVAD()

    async def _noop_utterance(pcm: bytes, sample_rate: int, mode: str) -> None:
        return None

    collector = AudioUtteranceCollector(
        vad_provider=vad,
        event_sink=recorder.sink,
        utterance_callback=_noop_utterance,
        cancel_callback=pipeline.cancel_tts,
    )
    recorder.start()
    try:
        # Start a turn that produces TTS tasks.
        await pipeline.handle_text_turn(TURN_INPUT)

        # Wait for the first TTS chunk to be observed (means a TTS task is alive).
        for _ in range(50):
            if recorder.first_event("tts.first_chunk") is not None:
                break
            await asyncio.sleep(0.02)
        if recorder.first_event("tts.first_chunk") is None:
            return float("nan")

        # Arm the VAD to fire ``vad.speech_start`` on the next ingest.
        vad.arm(VADEvent("vad.speech_start", 0.9, 30))

        # Capture start time just before triggering the barge-in. The
        # start of the collector's ``ingest_frame`` call is the moment
        # the new utterance begins to be processed by the framework.
        start_ns = time.perf_counter_ns()
        await collector.ingest_frame(_make_audio_frame(sequence=0))

        # Wait for ``tts.cancelled`` to be observed by the watcher.
        cancel_ts: int | None = None
        for _ in range(50):
            cancel_ts = recorder.first_event("tts.cancelled")
            if cancel_ts is not None:
                break
            await asyncio.sleep(0.02)

        if cancel_ts is None:
            return float("nan")
        return (cancel_ts - start_ns) / 1e6
    finally:
        await recorder.stop()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _percentile(sorted_values: list[float], target_percentile: float) -> float:
    """Compute ``target_percentile`` (0-100) using linear interpolation."""
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    if target_percentile <= 0:
        return sorted_values[0]
    if target_percentile >= 100:
        return sorted_values[-1]
    rank = (target_percentile / 100.0) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = rank - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def _print_table(samples: dict[str, list[float]]) -> None:
    """Print a small fixed-width table of the benchmark results."""
    header = ("Metric", "median (ms)", "p95 (ms)", "min (ms)", "max (ms)", "N")
    rows: list[tuple[str, str, str, str, str, str]] = []
    for key, label in METRIC_LABELS.items():
        values = sorted(samples[key])
        if not values:
            rows.append((label, "-", "-", "-", "-", "0"))
            continue
        median = statistics.median(values)
        p95 = _percentile(values, 95)
        rows.append(
            (
                label,
                f"{median:.1f}",
                f"{p95:.1f}",
                f"{min(values):.1f}",
                f"{max(values):.1f}",
                str(len(values)),
            )
        )

    widths = [
        max(len(str(row[i])) for row in [header, *rows])
        for i in range(len(header))
    ]
    line = "  ".join
    print(line(str(cell).ljust(widths[i]) for i, cell in enumerate(header)))
    print(line("-" * widths[i] for i in range(len(header))))
    for row in rows:
        print(line(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def _run_benchmark(iterations: int) -> dict[str, list[float]]:
    samples: dict[str, list[float]] = {key: [] for key in METRIC_LABELS}
    progress_step = max(1, iterations // 10)
    for i in range(iterations):
        text_metrics = await measure_text_turn_metrics()
        for key, value in text_metrics.items():
            samples[key].append(value)
        barge_in_ms = await measure_barge_in_metric()
        samples["barge_in_cancel_latency_ms"].append(barge_in_ms)
        if (i + 1) % progress_step == 0 or i == 0:
            print(f"  iteration {i + 1}/{iterations} complete")
            await asyncio.sleep(0)  # let stdout flush / loop schedule
    return samples


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python benchmarks/perf_compare.py",
        description=(
            "Run a deterministic performance benchmark for the Converse "
            "Framework pipeline (mock providers, no real I/O)."
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of iterations for each metric (default: 50).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    iterations = max(1, args.iterations)
    print(f"Converse Framework performance benchmark -- N={iterations}")
    print(
        "  using mock providers + QueueEventSink + QueueTransport (no real I/O)"
    )
    print()
    samples = asyncio.run(_run_benchmark(iterations))
    print()
    _print_table(samples)
    return 0


if __name__ == "__main__":
    sys.exit(main())
