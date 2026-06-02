"""Tests for the VAD-driven audio utterance collector."""

from __future__ import annotations

import asyncio
import base64
import struct
from typing import Any

import pytest

from converse_framework.audio_utils import AudioFrame
from converse_framework.events import QueueEventSink
from converse_framework.protocols import (
    ProviderCapabilities,
    ProviderStatus,
    VADEvent,
)
from converse_framework.utterance_collector import (
    AudioUtteranceCollector,
    UtteranceCollectorConfig,
)


# ---------------------------------------------------------------------------
# Test doubles and helpers
# ---------------------------------------------------------------------------


SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_MS = 30
EXPECTED_FRAME_BYTES = SAMPLE_RATE * CHANNELS * 2 * FRAME_MS // 1000


def make_frame(
    sequence: int = 0,
    samples: list[int] | None = None,
    *,
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
    frame_ms: int = FRAME_MS,
) -> AudioFrame:
    if samples is None:
        samples = [0] * (sample_rate * frame_ms // 1000)
    return AudioFrame(
        data=struct.pack(f"<{len(samples)}h", *samples),
        sequence=sequence,
        sample_rate=sample_rate,
        channels=channels,
        frame_ms=frame_ms,
        encoding="pcm_s16le",
    )


def make_payload(
    sequence: int = 0,
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
    frame_ms: int = FRAME_MS,
    samples: list[int] | None = None,
) -> dict[str, Any]:
    frame = make_frame(
        sequence=sequence,
        samples=samples,
        sample_rate=sample_rate,
        channels=channels,
        frame_ms=frame_ms,
    )
    return {
        "encoding": frame.encoding,
        "sample_rate": frame.sample_rate,
        "channels": frame.channels,
        "frame_ms": frame.frame_ms,
        "sequence": frame.sequence,
        "data": base64.b64encode(frame.data).decode("ascii"),
    }


class FakeVADProvider:
    """VAD provider that returns a scripted list of VADEvent lists.

    Each call to :meth:`process_frame` pops the next list from
    ``scripted``. When ``raise_value_error`` is set the next call raises
    a ``ValueError`` instead.
    """

    def __init__(self, scripted: list[list[VADEvent]] | None = None):
        self.scripted: list[list[VADEvent]] = list(scripted or [])
        self.calls = 0
        self.raise_value_error = False
        self.unloaded = False

    @property
    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name="fake-vad",
            kind="vad",
            ready=True,
            message="Fake VAD for collector tests.",
            capabilities=ProviderCapabilities(supports_barge_in=True),
        )

    async def check_status(self) -> ProviderStatus:
        return self.status

    async def process_frame(self, frame: AudioFrame) -> list[VADEvent]:
        self.calls += 1
        if self.raise_value_error:
            self.raise_value_error = False
            raise ValueError("simulated VAD failure")
        if not self.scripted:
            return []
        return list(self.scripted.pop(0))

    async def unload(self) -> ProviderStatus:
        self.unloaded = True
        return self.status


def drain(queue: asyncio.Queue[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


def event_types(events: list[dict[str, Any]]) -> list[str]:
    return [event["type"] for event in events]


def by_type(events: list[dict[str, Any]], type_name: str) -> list[dict[str, Any]]:
    return [event for event in events if event["type"] == type_name]


def make_collector(
    vad: FakeVADProvider | None = None,
    *,
    config: UtteranceCollectorConfig | None = None,
    cancel_callback=None,
    pre_speech_start_hook=None,
):
    sink_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    utterances: list[tuple[bytes, int, str]] = []
    cancel_calls: list[str] = []

    async def utterance_callback(pcm: bytes, sample_rate: int, mode: str) -> None:
        utterances.append((pcm, sample_rate, mode))

    if cancel_callback is not None:
        cancel_cb = cancel_callback
    else:

        async def cancel_cb(reason: str) -> None:
            cancel_calls.append(reason)

    vad = vad or FakeVADProvider()
    collector = AudioUtteranceCollector(
        vad_provider=vad,
        event_sink=QueueEventSink(sink_queue),
        utterance_callback=utterance_callback,
        config=config,
        cancel_callback=cancel_cb,
        pre_speech_start_hook=pre_speech_start_hook,
    )
    return collector, sink_queue, utterances, cancel_calls


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_derived_values_for_mono():
    config = UtteranceCollectorConfig(
        sample_rate=16000,
        channels=1,
        frame_ms=30,
        pre_speech_ms=450,
        max_utterance_ms=30000,
    )
    assert config.pre_speech_frames == 15
    assert config.max_utterance_frames == 1000
    assert config.bytes_per_ms == 32
    assert config.expected_frame_bytes == 960


def test_config_derived_values_for_stereo():
    config = UtteranceCollectorConfig(
        sample_rate=16000,
        channels=2,
        frame_ms=30,
    )
    assert config.bytes_per_ms == 64
    assert config.expected_frame_bytes == 1920


def test_config_rejects_non_positive_frame_ms():
    with pytest.raises(ValueError, match="frame_ms"):
        UtteranceCollectorConfig(frame_ms=0)


def test_config_rejects_non_positive_channels():
    with pytest.raises(ValueError, match="channels"):
        UtteranceCollectorConfig(channels=0)


def test_config_rejects_non_positive_sample_rate():
    with pytest.raises(ValueError, match="sample_rate"):
        UtteranceCollectorConfig(sample_rate=0)


def test_config_to_dict_excludes_derived_values():
    config = UtteranceCollectorConfig(min_speech_duration_ms=123)
    serialized = config.to_dict()
    assert serialized["min_speech_duration_ms"] == 123
    assert "expected_frame_bytes" not in serialized
    assert "pre_speech_frames" not in serialized


# ---------------------------------------------------------------------------
# Frame ingestion
# ---------------------------------------------------------------------------


def test_ingest_frame_emits_input_level_with_metrics():
    async def run():
        vad = FakeVADProvider()
        config = UtteranceCollectorConfig(
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
            frame_ms=FRAME_MS,
        )
        collector, queue, _, _ = make_collector(vad=vad, config=config)
        collector._audio_stats.last_emit_ts = 0
        await collector.ingest_frame(make_frame(sequence=0))
        return drain(queue)

    events = asyncio.run(run())
    levels = by_type(events, "audio.input_level")
    assert len(levels) == 1
    payload = levels[0]["payload"]
    assert payload["sequence"] == 0
    assert payload["received_frames"] == 1
    assert payload["dropped_frames"] == 0
    assert payload["sample_rate"] == SAMPLE_RATE
    assert payload["channels"] == CHANNELS
    assert payload["frame_ms"] == FRAME_MS


def test_speech_start_drains_pre_buffer_into_utterance():
    async def run():
        vad = FakeVADProvider(
            [
                [VADEvent("vad.probability", 0.3, 30)],
                [VADEvent("vad.probability", 0.3, 60)],
                [
                    VADEvent("vad.probability", 0.9, 90),
                    VADEvent("vad.speech_start", 0.9, 90),
                ],
                [VADEvent("vad.probability", 0.9, 120)],
                [
                    VADEvent("vad.probability", 0.1, 150),
                    VADEvent("vad.speech_end", 0.1, 150),
                ],
            ]
        )
        config = UtteranceCollectorConfig(
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
            frame_ms=FRAME_MS,
            pre_speech_ms=60,
            min_speech_duration_ms=0,
            reject_low_energy_rms=0,
            reject_utterance_rms=0,
            trim_silence_rms=0,
        )
        collector, queue, utterances, _ = make_collector(vad=vad, config=config)
        for seq in range(5):
            await collector.ingest_frame(make_frame(sequence=seq, samples=[2000] * 480))
        return drain(queue), utterances

    events, utterances = asyncio.run(run())
    types = event_types(events)
    assert types.count("vad.speech_start") == 1
    assert types.count("vad.speech_end") == 1
    assert len(utterances) == 1
    pcm, sample_rate, mode = utterances[0]
    # pre_speech_frames = 2, so the pre-buffer holds the most recent 2
    # frames at speech_start. The other 2 frames come from frames 3 and 4
    # received after speech_start while still recording.
    assert len(pcm) == 4 * EXPECTED_FRAME_BYTES
    assert sample_rate == SAMPLE_RATE
    assert mode == "chat"


def test_speech_end_dispatches_utterance_to_callback():
    async def run():
        vad = FakeVADProvider(
            [
                [VADEvent("vad.speech_start", 0.9, 30)],
                [VADEvent("vad.speech_end", 0.1, 60)],
            ]
        )
        config = UtteranceCollectorConfig(
            min_speech_duration_ms=0,
            reject_low_energy_rms=0,
            reject_utterance_rms=0,
            trim_silence_rms=0,
        )
        collector, queue, utterances, _ = make_collector(vad=vad, config=config)
        for seq in range(2):
            await collector.ingest_frame(make_frame(sequence=seq, samples=[1500] * 480))
        return drain(queue), utterances

    events, utterances = asyncio.run(run())
    assert len(utterances) == 1
    pcm, sample_rate, mode = utterances[0]
    expected_frame = struct.pack("<480h", *([1500] * 480))
    assert pcm == expected_frame * 2
    assert sample_rate == SAMPLE_RATE
    assert mode == "chat"


# ---------------------------------------------------------------------------
# Rejection gates
# ---------------------------------------------------------------------------


def test_min_duration_rejection_drops_short_utterance():
    async def run():
        vad = FakeVADProvider(
            [
                [VADEvent("vad.speech_start", 0.9, 30)],
                [VADEvent("vad.speech_end", 0.1, 60)],
            ]
        )
        config = UtteranceCollectorConfig(min_speech_duration_ms=500)
        collector, queue, utterances, _ = make_collector(vad=vad, config=config)
        for seq in range(2):
            await collector.ingest_frame(make_frame(sequence=seq, samples=[2000] * 480))
        return drain(queue), utterances

    events, utterances = asyncio.run(run())
    rejected = by_type(events, "vad.speech_rejected")
    assert len(rejected) == 1
    payload = rejected[0]["payload"]
    assert payload["min_duration_ms"] == 500
    assert payload["duration_ms"] < 500
    assert "reason" not in payload
    assert utterances == []


def test_short_low_energy_rejection_emits_low_energy_reason():
    async def run():
        vad = FakeVADProvider(
            [
                [VADEvent("vad.speech_start", 0.9, 30)],
                [VADEvent("vad.speech_end", 0.1, 60)],
            ]
        )
        config = UtteranceCollectorConfig(
            min_speech_duration_ms=0,
            reject_low_energy_rms=0.01,
            reject_low_energy_max_duration_ms=900,
            trim_silence_rms=0,
        )
        collector, queue, utterances, _ = make_collector(vad=vad, config=config)
        for seq in range(2):
            await collector.ingest_frame(make_frame(sequence=seq, samples=[10] * 480))
        return drain(queue), utterances

    events, utterances = asyncio.run(run())
    rejected = by_type(events, "vad.speech_rejected")
    assert len(rejected) == 1
    payload = rejected[0]["payload"]
    assert payload["reason"] == "low_energy"
    assert payload["rms"] < payload["min_rms"]
    assert utterances == []


def test_utterance_rms_rejection_emits_utterance_low_energy_reason():
    async def run():
        vad = FakeVADProvider(
            [
                [VADEvent("vad.speech_start", 0.9, 30)],
                [VADEvent("vad.speech_end", 0.1, 60)],
            ]
        )
        config = UtteranceCollectorConfig(
            min_speech_duration_ms=0,
            reject_low_energy_rms=0,
            reject_utterance_rms=0.05,
            trim_silence_rms=0,
        )
        collector, queue, utterances, _ = make_collector(vad=vad, config=config)
        for seq in range(2):
            await collector.ingest_frame(make_frame(sequence=seq, samples=[200] * 480))
        return drain(queue), utterances

    events, utterances = asyncio.run(run())
    rejected = by_type(events, "vad.speech_rejected")
    assert len(rejected) == 1
    payload = rejected[0]["payload"]
    assert payload["reason"] == "utterance_low_energy"
    assert payload["rms"] < payload["min_rms"]
    assert utterances == []


def test_silence_trimming_emits_asr_audio_trimmed_event():
    async def run():
        vad = FakeVADProvider(
            [
                [VADEvent("vad.speech_start", 0.9, 30)],
                [VADEvent("vad.speech_end", 0.1, 60)],
            ]
        )
        quiet = [0] * 480
        loud = [2000] * 480
        config = UtteranceCollectorConfig(
            min_speech_duration_ms=50,
            reject_low_energy_rms=0,
            reject_utterance_rms=0,
            trim_silence_rms=0.01,
            trim_silence_frame_ms=30,
        )
        collector, queue, utterances, _ = make_collector(vad=vad, config=config)
        await collector.ingest_frame(make_frame(sequence=0, samples=quiet))
        await collector.ingest_frame(make_frame(sequence=1, samples=loud))
        return drain(queue), utterances

    events, utterances = asyncio.run(run())
    trimmed = by_type(events, "asr.audio_trimmed")
    assert len(trimmed) == 1
    payload = trimmed[0]["payload"]
    assert payload["original_duration_ms"] > payload["trimmed_duration_ms"]
    assert len(utterances) == 1
    pcm = utterances[0][0]
    expected_speech = struct.pack("<480h", *([2000] * 480))
    assert pcm == expected_speech


def test_max_utterance_emits_buffer_warning_and_stops_recording():
    async def run():
        vad = FakeVADProvider(
            [
                [VADEvent("vad.speech_start", 0.9, 30)],
                [VADEvent("vad.speech_start", 0.9, 60)],
            ]
        )
        config = UtteranceCollectorConfig(
            min_speech_duration_ms=0,
            reject_low_energy_rms=0,
            reject_utterance_rms=0,
            trim_silence_rms=0,
            max_utterance_ms=60,
        )
        collector, queue, _, _ = make_collector(vad=vad, config=config)
        for seq in range(3):
            await collector.ingest_frame(make_frame(sequence=seq, samples=[1000] * 480))
        return drain(queue), collector.is_recording

    events, is_recording = asyncio.run(run())
    warnings = by_type(events, "asr.buffer_warning")
    assert len(warnings) == 1
    assert is_recording is False


# ---------------------------------------------------------------------------
# VAD event forwarding
# ---------------------------------------------------------------------------


def test_vad_probability_forwarded_with_per_frame_mode():
    async def run():
        vad = FakeVADProvider([[VADEvent("vad.probability", 0.42, 30)]])
        collector, queue, _, _ = make_collector(vad=vad)
        await collector.ingest_frame(make_frame(sequence=0), mode="custom")
        return drain(queue)

    events = asyncio.run(run())
    probs = by_type(events, "vad.probability")
    assert len(probs) == 1
    payload = probs[0]["payload"]
    assert payload["mode"] == "custom"
    assert payload["probability"] == 0.42


def test_vad_probability_uses_per_frame_mode_not_recording_mode():
    async def run():
        vad = FakeVADProvider(
            [
                [VADEvent("vad.speech_start", 0.9, 30)],
                [VADEvent("vad.probability", 0.5, 60)],
            ]
        )
        collector, queue, _, _ = make_collector(vad=vad)
        await collector.ingest_frame(make_frame(sequence=0), mode="chat")
        await collector.ingest_frame(make_frame(sequence=1), mode="custom")
        return drain(queue)

    events = asyncio.run(run())
    probs = by_type(events, "vad.probability")
    assert len(probs) == 1
    assert probs[0]["payload"]["mode"] == "custom"


def test_vad_value_error_emits_vad_error_and_skips_remaining_events():
    async def run():
        vad = FakeVADProvider([[VADEvent("vad.probability", 0.4, 30)]])
        vad.raise_value_error = True
        collector, queue, _, _ = make_collector(vad=vad)
        await collector.ingest_frame(make_frame(sequence=0))
        return drain(queue)

    events = asyncio.run(run())
    types = event_types(events)
    assert types == ["vad.error"]
    assert events[0]["payload"]["message"] == "simulated VAD failure"


# ---------------------------------------------------------------------------
# Callbacks and hooks
# ---------------------------------------------------------------------------


def test_barge_in_cancel_callback_fires_on_speech_start():
    async def run():
        vad = FakeVADProvider([[VADEvent("vad.speech_start", 0.9, 30)]])
        collector, _, _, cancel_calls = make_collector(vad=vad)
        await collector.ingest_frame(make_frame(sequence=0))
        return cancel_calls

    cancel_calls = asyncio.run(run())
    assert cancel_calls == ["vad_barge_in"]


def test_instance_pre_speech_start_hook_fires_with_frame_and_mode():
    async def run():
        hook_calls: list[tuple[AudioFrame, str]] = []

        async def hook(frame: AudioFrame, mode: str) -> None:
            hook_calls.append((frame, mode))

        vad = FakeVADProvider([[VADEvent("vad.speech_start", 0.9, 30)]])
        collector, _, _, _ = make_collector(vad=vad, pre_speech_start_hook=hook)
        await collector.ingest_frame(make_frame(sequence=0), mode="custom")
        return hook_calls

    hook_calls = asyncio.run(run())
    assert len(hook_calls) == 1
    frame, mode = hook_calls[0]
    assert isinstance(frame, AudioFrame)
    assert frame.sequence == 0
    assert mode == "custom"


def test_per_frame_pre_speech_start_hook_overrides_instance():
    async def run():
        instance_calls: list[str] = []
        frame_calls: list[str] = []

        async def instance_hook(frame: AudioFrame, mode: str) -> None:
            instance_calls.append(mode)

        async def frame_hook(frame: AudioFrame, mode: str) -> None:
            frame_calls.append(mode)

        vad = FakeVADProvider(
            [
                [VADEvent("vad.speech_start", 0.9, 30)],
                [VADEvent("vad.speech_start", 0.9, 60)],
            ]
        )
        collector, _, _, _ = make_collector(
            vad=vad, pre_speech_start_hook=instance_hook
        )
        await collector.ingest_frame(
            make_frame(sequence=0),
            mode="first",
            pre_speech_start_hook=frame_hook,
        )
        await collector.ingest_frame(make_frame(sequence=1), mode="second")
        return instance_calls, frame_calls

    instance_calls, frame_calls = asyncio.run(run())
    assert frame_calls == ["first"]
    assert instance_calls == ["second"]


def test_pre_speech_start_hook_runs_before_cancel_callback():
    async def run():
        order: list[str] = []

        async def hook(frame: AudioFrame, mode: str) -> None:
            order.append("hook")

        async def cancel(reason: str) -> None:
            order.append("cancel")

        vad = FakeVADProvider([[VADEvent("vad.speech_start", 0.9, 30)]])
        collector, _, _, _ = make_collector(
            vad=vad,
            cancel_callback=cancel,
            pre_speech_start_hook=hook,
        )
        await collector.ingest_frame(make_frame(sequence=0))
        return order

    order = asyncio.run(run())
    assert order == ["hook", "cancel"]


def test_cancel_active_turn_invokes_cancel_callback():
    async def run():
        cancel_reasons: list[str] = []

        async def cancel(reason: str) -> None:
            cancel_reasons.append(reason)

        vad = FakeVADProvider()
        collector, _, _, _ = make_collector(vad=vad, cancel_callback=cancel)
        await collector.cancel_active_turn("manual")
        await collector.cancel_active_turn("barge_in")
        return cancel_reasons

    reasons = asyncio.run(run())
    assert reasons == ["manual", "barge_in"]


# ---------------------------------------------------------------------------
# State properties
# ---------------------------------------------------------------------------


def test_is_recording_and_current_mode_track_state():
    async def run():
        vad = FakeVADProvider(
            [
                [VADEvent("vad.speech_start", 0.9, 30)],
                [VADEvent("vad.speech_end", 0.1, 60)],
            ]
        )
        collector, _, _, _ = make_collector(vad=vad)
        start_state = (collector.is_recording, collector.current_mode)
        await collector.ingest_frame(make_frame(sequence=0), mode="chat")
        mid_state = (collector.is_recording, collector.current_mode)
        await collector.ingest_frame(make_frame(sequence=1, samples=[1000] * 480))
        return (
            start_state,
            mid_state,
            (
                collector.is_recording,
                collector.current_mode,
            ),
        )

    start, mid, end = asyncio.run(run())
    assert start == (False, "chat")
    assert mid == (True, "chat")
    assert end == (False, "chat")


def test_collector_owns_independent_audio_stats():
    async def run():
        a, _, _, _ = make_collector()
        b, _, _, _ = make_collector()
        a._audio_stats.last_emit_ts = 0
        await a.ingest_frame(make_frame(sequence=0))
        return a._audio_stats.received_frames, b._audio_stats.received_frames

    a_count, b_count = asyncio.run(run())
    assert a_count == 1
    assert b_count == 0


def test_collector_uses_config_sample_rate_for_callback():
    async def run():
        vad = FakeVADProvider(
            [
                [VADEvent("vad.speech_start", 0.9, 30)],
                [VADEvent("vad.speech_end", 0.1, 60)],
            ]
        )
        config = UtteranceCollectorConfig(
            sample_rate=8000,
            channels=1,
            frame_ms=30,
            min_speech_duration_ms=0,
            reject_low_energy_rms=0,
            reject_utterance_rms=0,
            trim_silence_rms=0,
        )
        collector, _, utterances, _ = make_collector(vad=vad, config=config)
        for seq in range(2):
            await collector.ingest_frame(make_frame(sequence=seq, sample_rate=8000))
        return utterances

    utterances = asyncio.run(run())
    assert len(utterances) == 1
    assert utterances[0][1] == 8000


def test_collector_update_config_rebuilds_derived_state():
    collector, _, _, _ = make_collector()
    updated = collector.update_config(
        sample_rate=8000,
        frame_ms=20,
        min_speech_duration_ms=150,
    )
    assert updated.sample_rate == 8000
    assert updated.frame_ms == 20
    assert updated.expected_frame_bytes == 320
    assert collector.serialize_config()["min_speech_duration_ms"] == 150
    assert collector._audio_stats.expected_sample_rate == 8000
    assert collector._audio_stats.expected_frame_ms == 20
    assert collector._pre_buffer.maxlen == updated.pre_speech_frames


def test_collector_update_config_rejects_unknown_field():
    collector, _, _, _ = make_collector()
    with pytest.raises(ValueError, match="unknown"):
        collector.update_config(nope=1)


def test_collector_update_config_rejects_while_recording():
    async def run():
        vad = FakeVADProvider([[VADEvent("vad.speech_start", 0.9, 30)]])
        collector, _, _, _ = make_collector(vad=vad)
        await collector.ingest_frame(make_frame(sequence=0))
        return collector

    collector = asyncio.run(run())
    with pytest.raises(RuntimeError, match="recording"):
        collector.update_config(min_speech_duration_ms=150)


# ---------------------------------------------------------------------------
# Disabled gates (zero thresholds) do not block dispatch
# ---------------------------------------------------------------------------


def test_zero_thresholds_pass_through():
    async def run():
        vad = FakeVADProvider(
            [
                [VADEvent("vad.speech_start", 0.9, 30)],
                [VADEvent("vad.speech_end", 0.1, 60)],
            ]
        )
        config = UtteranceCollectorConfig(
            min_speech_duration_ms=0,
            reject_low_energy_rms=0,
            reject_utterance_rms=0,
            trim_silence_rms=0,
        )
        collector, queue, utterances, _ = make_collector(vad=vad, config=config)
        for seq in range(2):
            await collector.ingest_frame(make_frame(sequence=seq, samples=[0] * 480))
        return drain(queue), utterances

    events, utterances = asyncio.run(run())
    assert by_type(events, "vad.speech_rejected") == []
    assert by_type(events, "asr.audio_trimmed") == []
    assert len(utterances) == 1
