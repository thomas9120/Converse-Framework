"""VAD-driven audio utterance collector.

Encapsulates the state machine that turns a stream of parsed
:class:`AudioFrame` objects into complete utterance byte buffers ready
for ASR, while emitting compatible input-level, VAD, and rejection
events through an :class:`EventSink`.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, fields, replace

from converse_framework.audio_utils import (
    AudioFrame,
    AudioFrameStats,
    compute_pcm16_level,
    trim_pcm16_silence,
)
from converse_framework.events import EventSink
from converse_framework.protocols import VADProvider

logger = logging.getLogger(__name__)


PreSpeechStartHook = Callable[[AudioFrame, str], Awaitable[None]]
UtteranceCallback = Callable[[bytes, int, str], Awaitable[None]]
CancelCallback = Callable[[str], Awaitable[None]]


@dataclass(frozen=True)
class UtteranceCollectorConfig:
    """Tuning knobs for the VAD utterance collector.

    Derived frame counts and byte sizes are computed in
    :meth:`__post_init__` so callers can read them after construction.
    """

    sample_rate: int = 16000
    channels: int = 1
    frame_ms: int = 30
    pre_speech_ms: int = 450
    max_utterance_ms: int = 30000
    min_speech_duration_ms: int = 300
    reject_low_energy_rms: float = 0.003
    reject_low_energy_max_duration_ms: int = 900
    reject_utterance_rms: float = 0.002
    trim_silence_rms: float = 0.003
    trim_silence_frame_ms: int = 30

    pre_speech_frames: int = field(init=False)
    max_utterance_frames: int = field(init=False)
    bytes_per_ms: int = field(init=False)
    expected_frame_bytes: int = field(init=False)

    def __post_init__(self) -> None:
        if self.frame_ms <= 0:
            raise ValueError("frame_ms must be > 0")
        if self.channels <= 0:
            raise ValueError("channels must be > 0")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        object.__setattr__(
            self, "pre_speech_frames", max(1, self.pre_speech_ms // self.frame_ms)
        )
        object.__setattr__(
            self,
            "max_utterance_frames",
            max(1, self.max_utterance_ms // self.frame_ms),
        )
        object.__setattr__(
            self,
            "bytes_per_ms",
            self.sample_rate * self.channels * 2 // 1000,
        )
        object.__setattr__(
            self, "expected_frame_bytes", self.bytes_per_ms * self.frame_ms
        )

    def to_dict(self) -> dict[str, int | float]:
        """Return only caller-configurable fields for persistence."""
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.init
        }


class AudioUtteranceCollector:
    """VAD-driven utterance collector.

    Accepts parsed :class:`AudioFrame` objects, runs the configured VAD
    provider, applies the rejection gates, and dispatches the resulting
    utterance bytes to a caller callback. Input-level, VAD, and
    rejection events are emitted through the configured
    :class:`EventSink` so transport layers can forward them to clients
    without coupling to the collector itself.
    """

    def __init__(
        self,
        vad_provider: VADProvider,
        event_sink: EventSink,
        utterance_callback: UtteranceCallback,
        config: UtteranceCollectorConfig | None = None,
        cancel_callback: CancelCallback | None = None,
        pre_speech_start_hook: PreSpeechStartHook | None = None,
    ) -> None:
        self._vad = vad_provider
        self._sink = event_sink
        self._utterance_callback = utterance_callback
        self._cancel_callback = cancel_callback
        self._pre_speech_start_hook = pre_speech_start_hook
        self.config = config or UtteranceCollectorConfig()
        self._audio_stats = AudioFrameStats(
            expected_sample_rate=self.config.sample_rate,
            expected_channels=self.config.channels,
            expected_frame_ms=self.config.frame_ms,
        )
        self._pre_buffer: deque[bytes] = deque(
            maxlen=self.config.pre_speech_frames
        )
        self._utterance_buffer = bytearray()
        self._recording = False
        self._recording_mode = "chat"

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def current_mode(self) -> str:
        return self._recording_mode

    def serialize_config(self) -> dict[str, int | float]:
        """Return the current collector configuration for app persistence."""
        return self.config.to_dict()

    def update_config(self, **overrides: int | float) -> UtteranceCollectorConfig:
        """Update collector tuning knobs and rebuild derived state.

        The update is rejected while recording so an in-flight utterance
        cannot be interpreted with mixed frame sizes, sample rates, or
        rejection gates.
        """
        if self._recording:
            raise RuntimeError("cannot update collector config while recording")
        allowed = set(self.config.to_dict())
        unknown = set(overrides) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown collector config field(s): {names}")
        self.config = replace(self.config, **overrides)
        self._audio_stats = AudioFrameStats(
            expected_sample_rate=self.config.sample_rate,
            expected_channels=self.config.channels,
            expected_frame_ms=self.config.frame_ms,
        )
        self._pre_buffer = deque(maxlen=self.config.pre_speech_frames)
        self._utterance_buffer.clear()
        return self.config

    async def cancel_active_turn(self, reason: str) -> None:
        if self._cancel_callback is not None:
            await self._cancel_callback(reason)

    async def ingest_frame(
        self,
        frame: AudioFrame,
        *,
        mode: str = "chat",
        pre_speech_start_hook: PreSpeechStartHook | None = None,
    ) -> None:
        config = self.config
        self._pre_buffer.append(frame.data)

        if self._recording:
            self._utterance_buffer.extend(frame.data)
            max_bytes = config.max_utterance_frames * config.expected_frame_bytes
            if len(self._utterance_buffer) > max_bytes:
                await self._sink.emit(
                    "asr.buffer_warning",
                    message=(
                        "Maximum utterance length reached; closing current "
                        "utterance."
                    ),
                )
                self._recording = False

        metrics = self._audio_stats.update(frame)
        if metrics is not None:
            await self._sink.emit("audio.input_level", **metrics)

        try:
            vad_events = await self._vad.process_frame(frame)
        except ValueError as exc:
            await self._sink.emit("vad.error", message=str(exc))
            return

        hook = (
            pre_speech_start_hook
            if pre_speech_start_hook is not None
            else self._pre_speech_start_hook
        )

        for vad_event in vad_events:
            if vad_event.type == "vad.speech_start":
                self._recording_mode = mode
                if hook is not None:
                    await hook(frame, self._recording_mode)
                if self._cancel_callback is not None:
                    await self._cancel_callback("vad_barge_in")
                self._utterance_buffer.clear()
                for buffered in self._pre_buffer:
                    self._utterance_buffer.extend(buffered)
                self._recording = True
                await self._sink.emit(
                    "vad.speech_start",
                    mode=self._recording_mode,
                    probability=vad_event.probability,
                    audio_ms=vad_event.audio_ms,
                )
            elif vad_event.type == "vad.speech_end":
                self._recording = False
                pcm = bytes(self._utterance_buffer)
                self._utterance_buffer.clear()
                await self._sink.emit(
                    "vad.speech_end",
                    mode=self._recording_mode,
                    probability=vad_event.probability,
                    audio_ms=vad_event.audio_ms,
                )
                pcm = await self._apply_rejection_gates(
                    pcm, mode=self._recording_mode
                )
                if pcm:
                    await self._utterance_callback(
                        pcm, config.sample_rate, self._recording_mode
                    )
            elif vad_event.type == "vad.probability":
                await self._sink.emit(
                    "vad.probability",
                    mode=mode,
                    probability=vad_event.probability,
                    audio_ms=vad_event.audio_ms,
                )

    async def _apply_rejection_gates(self, pcm: bytes, *, mode: str) -> bytes:
        config = self.config
        bytes_per_ms = max(config.bytes_per_ms, 1)

        if pcm and config.min_speech_duration_ms > 0:
            duration_ms = len(pcm) // bytes_per_ms
            if duration_ms < config.min_speech_duration_ms:
                await self._sink.emit(
                    "vad.speech_rejected",
                    mode=mode,
                    duration_ms=duration_ms,
                    min_duration_ms=config.min_speech_duration_ms,
                )
                return b""

        if (
            pcm
            and config.reject_low_energy_rms > 0
            and config.reject_low_energy_max_duration_ms > 0
        ):
            duration_ms = len(pcm) // bytes_per_ms
            level = compute_pcm16_level(pcm)
            if (
                duration_ms <= config.reject_low_energy_max_duration_ms
                and level["rms"] < config.reject_low_energy_rms
            ):
                await self._sink.emit(
                    "vad.speech_rejected",
                    mode=mode,
                    duration_ms=duration_ms,
                    rms=level["rms"],
                    min_rms=config.reject_low_energy_rms,
                    reason="low_energy",
                )
                return b""

        if pcm and config.reject_utterance_rms > 0:
            duration_ms = len(pcm) // bytes_per_ms
            level = compute_pcm16_level(pcm)
            if level["rms"] < config.reject_utterance_rms:
                await self._sink.emit(
                    "vad.speech_rejected",
                    mode=mode,
                    duration_ms=duration_ms,
                    rms=level["rms"],
                    min_rms=config.reject_utterance_rms,
                    reason="utterance_low_energy",
                )
                return b""

        if pcm and config.trim_silence_rms > 0:
            original_duration_ms = len(pcm) // bytes_per_ms
            trimmed = trim_pcm16_silence(
                pcm,
                frame_ms=config.trim_silence_frame_ms,
                sample_rate=config.sample_rate,
                rms_threshold=config.trim_silence_rms,
            )
            trimmed_duration_ms = len(trimmed) // bytes_per_ms
            if trimmed_duration_ms != original_duration_ms:
                await self._sink.emit(
                    "asr.audio_trimmed",
                    mode=mode,
                    original_duration_ms=original_duration_ms,
                    trimmed_duration_ms=trimmed_duration_ms,
                    rms_threshold=config.trim_silence_rms,
                )
            pcm = trimmed

        return pcm


__all__ = [
    "AudioUtteranceCollector",
    "PreSpeechStartHook",
    "UtteranceCallback",
    "UtteranceCollectorConfig",
]
