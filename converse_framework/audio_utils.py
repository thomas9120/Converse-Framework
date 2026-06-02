"""Audio frame parsing, PCM conversion, metering, and silence trimming."""

from __future__ import annotations

import base64
import math
import struct
import time
import wave
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import numpy as np


SUPPORTED_ENCODING = "pcm_s16le"


@dataclass(frozen=True)
class AudioFrame:
    """A single parsed frame of mono PCM audio carried over the wire.

    Instances are produced by :func:`parse_audio_frame` from a JSON
    payload the transport received from a client. ``data`` is the
    raw 16-bit signed little-endian PCM bytes for this frame;
    ``sequence`` is the frame's monotonically increasing index from
    the sender, used by the stats tracker to detect drops; the
    remaining fields are the audio shape the sender promises
    (which the parser has already validated against the
    :class:`AudioFrameStats` expectation).

    Attributes:
        data: Raw 16-bit signed LE PCM bytes for this frame.
        sequence: Sender-assigned monotonically increasing frame
            index, starting at zero.
        sample_rate: Samples per second of the decoded audio.
        channels: Channel count (the framework's wire format
            currently only uses mono).
        frame_ms: Duration of one frame in milliseconds.
        encoding: Encoding name; always ``"pcm_s16le"`` for v0.1.
    """

    data: bytes
    sequence: int
    sample_rate: int
    channels: int
    frame_ms: int
    encoding: str


@dataclass
class AudioFrameStats:
    """Running frame statistics and throttled level emitter.

    The utterance collector keeps one of these per pipeline; the
    :meth:`update` method folds a fresh :class:`AudioFrame` into
    the running counters and, at most every 100 ms, returns a
    level / drop summary suitable for forwarding as an
    ``audio.input_level`` event. ``None`` means "throttled -- no
    event this frame".

    The class is mutable; call :meth:`update` once per frame
    received.

    Attributes:
        expected_sample_rate: Sample rate the parser will accept.
            Frames that disagree are rejected upstream.
        expected_channels: Channel count the parser will accept.
        expected_frame_ms: Frame duration the parser will accept.
        received_frames: Number of frames folded in so far.
        dropped_frames: Cumulative gap between the last seen
            sequence and the current one, i.e. the count of frames
            the sender skipped.
        last_sequence: Sequence number of the most recent frame,
            or ``None`` before the first frame.
        last_emit_ts: Wall-clock timestamp of the last emitted
            level summary, used for the 100 ms throttle.
    """

    expected_sample_rate: int
    expected_channels: int
    expected_frame_ms: int
    received_frames: int = 0
    dropped_frames: int = 0
    last_sequence: int | None = None
    last_emit_ts: float = field(default_factory=time.perf_counter)

    def update(self, frame: AudioFrame) -> dict[str, Any] | None:
        if self.last_sequence is not None and frame.sequence > self.last_sequence + 1:
            self.dropped_frames += frame.sequence - self.last_sequence - 1
        self.last_sequence = frame.sequence
        self.received_frames += 1

        now = time.perf_counter()
        if now - self.last_emit_ts < 0.1:
            return None
        self.last_emit_ts = now
        level = compute_pcm16_level(frame.data)
        return {
            "sequence": frame.sequence,
            "received_frames": self.received_frames,
            "dropped_frames": self.dropped_frames,
            "rms": level["rms"],
            "peak": level["peak"],
            "sample_rate": frame.sample_rate,
            "channels": frame.channels,
            "frame_ms": frame.frame_ms,
        }


# ---------------------------------------------------------------------------
# PCM conversion utilities
# ---------------------------------------------------------------------------


def pcm_s16le_to_float32(pcm_s16le: bytes) -> np.ndarray:
    """Decode signed-16-bit little-endian PCM bytes into a float32 array.

    Values are normalised to ``[-1.0, 1.0]`` using 32768 as the
    negative full-scale divisor (and 32767 for the positive side),
    matching the convention used by the rest of the framework and
    most speech model training pipelines.

    Args:
        pcm_s16le: Raw PCM bytes. An empty input returns an empty
            float32 array rather than raising.

    Returns:
        A 1-D ``np.float32`` array of decoded samples. ``dtype`` is
        always ``float32`` so downstream code can rely on a single
        numeric type.
    """
    if not pcm_s16le:
        return np.array([], dtype=np.float32)
    audio = np.frombuffer(pcm_s16le, dtype="<i2")
    return audio.astype(np.float32) / 32768.0


def _tensor_or_array_to_numpy(audio) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio)


def float_audio_to_wav_bytes(audio, sample_rate: int) -> bytes:
    """Encode a float audio buffer as a mono 16-bit PCM WAV byte string.

    Accepts a ``numpy`` array, a list, or a torch tensor. Values
    outside ``[-1.0, 1.0]`` are clipped to the valid PCM range; the
    output is always mono, 16-bit signed little-endian, at the
    requested sample rate. Empty input returns ``b""`` rather than
    a valid (silent) WAV.

    Args:
        audio: Array-like of float samples in ``[-1.0, 1.0]``.
        sample_rate: Sample rate to write into the WAV header.

    Returns:
        Complete WAV file as ``bytes`` (header + data), ready to
        stream to a transport or write to disk.
    """
    array = _tensor_or_array_to_numpy(audio)
    if array.size == 0:
        return b""
    array = np.asarray(array, dtype=np.float32).reshape(-1)
    clipped = np.clip(array, -1.0, 1.0)
    pcm = np.where(clipped < 0, clipped * 32768, clipped * 32767).astype("<i2")
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


def float_audio_to_pcm_s16le_bytes(audio) -> bytes:
    """Encode a float audio buffer as raw 16-bit signed LE PCM bytes.

    Equivalent to :func:`float_audio_to_wav_bytes` without the WAV
    header. Useful for sending audio to providers that take raw
    PCM over the wire (e.g. faster-whisper) or for in-memory
    concatenation before a final encoding step.

    Args:
        audio: Array-like of float samples in ``[-1.0, 1.0]``;
            torch tensors are accepted and detached automatically.

    Returns:
        Raw little-endian 16-bit PCM bytes. Empty input yields
        ``b""``.
    """
    array = _tensor_or_array_to_numpy(audio)
    if array.size == 0:
        return b""
    array = np.asarray(array, dtype=np.float32).reshape(-1)
    clipped = np.clip(array, -1.0, 1.0)
    pcm = np.where(clipped < 0, clipped * 32768, clipped * 32767).astype("<i2")
    return pcm.tobytes()


def make_tone_wav(
    duration_s: float = 0.18, frequency: float = 440.0, sample_rate: int = 16000
) -> bytes:
    """Generate a tiny mono PCM WAV tone.

    The mock TTS provider uses this to emit a deterministic,
    dependency-free stand-in for real speech so smoke tests can
    exercise the audio path end-to-end.

    Args:
        duration_s: Length of the tone in seconds. The function
            rounds up to the nearest sample, so the actual duration
            is ``ceil(duration_s * sample_rate) / sample_rate``.
        frequency: Sine frequency in Hertz.
        sample_rate: Sample rate of the generated WAV. The mock
            tests use 16 kHz to match the default ASR expectation.

    Returns:
        Complete 16-bit mono WAV as ``bytes``. The amplitude is
        hard-coded to 0.18 of full scale so the tone cannot clip
        the int16 range.
    """
    samples = max(1, int(duration_s * sample_rate))
    pcm = bytearray()
    amplitude = 0.18
    for index in range(samples):
        value = int(
            32767 * amplitude * math.sin(2 * math.pi * frequency * index / sample_rate)
        )
        pcm.extend(struct.pack("<h", value))

    data_size = len(pcm)
    byte_rate = sample_rate * 2
    header = b"".join(
        [
            b"RIFF",
            struct.pack("<I", 36 + data_size),
            b"WAVEfmt ",
            struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, byte_rate, 2, 16),
            b"data",
            struct.pack("<I", data_size),
        ]
    )
    return header + bytes(pcm)


# ---------------------------------------------------------------------------
# Metering and silence trimming
# ---------------------------------------------------------------------------


def compute_pcm16_level(data: bytes) -> dict[str, float]:
    """Compute RMS and peak level of a PCM-16 buffer.

    Both metrics are returned as normalised float values in
    ``[0.0, 1.0]`` (RMS is a level, not a power, so a sine at full
    scale reads ~0.707). Results are rounded to four decimals to
    keep wire payloads small.

    Args:
        data: Raw 16-bit signed LE PCM bytes. Empty input returns
            ``{"rms": 0.0, "peak": 0.0}`` rather than raising.

    Returns:
        ``{"rms": float, "peak": float}`` -- both in the ``[0.0,
        1.0]`` range.
    """
    if not data:
        return {"rms": 0.0, "peak": 0.0}
    arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if len(arr) == 0:
        return {"rms": 0.0, "peak": 0.0}
    peak = float(np.abs(arr).max())
    rms = float(np.sqrt(np.mean(arr**2)))
    return {"rms": round(rms, 4), "peak": round(peak, 4)}


def trim_pcm16_silence(
    data: bytes, *, frame_ms: int, sample_rate: int, rms_threshold: float
) -> bytes:
    """Strip leading and trailing silence from a PCM-16 byte buffer.

    The buffer is split into ``frame_ms``-sized frames and any
    frame whose RMS level is below ``rms_threshold`` is dropped
    from the start or end of the buffer. Interior low-level
    frames are kept.

    Args:
        data: Raw 16-bit signed LE PCM bytes. Empty input is
            returned unchanged.
        frame_ms: Frame size in milliseconds; matches the
            collector's ``frame_ms`` so the slice boundaries line
            up.
        sample_rate: Sample rate of the audio; combined with
            ``frame_ms`` to derive the frame byte count.
        rms_threshold: RMS level below which a frame is considered
            silent. ``<= 0`` disables trimming and returns
            ``data`` unchanged.

    Returns:
        The trimmed PCM byte buffer. If every frame is below
        threshold the function returns ``b""`` (the entire buffer
        was silence).
    """
    if not data or rms_threshold <= 0:
        return data
    bytes_per_frame = max(1, sample_rate * frame_ms // 1000 * 2)
    frames = [
        data[index : index + bytes_per_frame]
        for index in range(0, len(data), bytes_per_frame)
        if len(data[index : index + bytes_per_frame]) == bytes_per_frame
    ]
    if not frames:
        return data

    start = 0
    while (
        start < len(frames)
        and compute_pcm16_level(frames[start])["rms"] < rms_threshold
    ):
        start += 1

    end = len(frames) - 1
    while end >= start and compute_pcm16_level(frames[end])["rms"] < rms_threshold:
        end -= 1

    if start > end:
        return b""
    return b"".join(frames[start : end + 1])


# ---------------------------------------------------------------------------
# Audio frame parsing
# ---------------------------------------------------------------------------


def parse_audio_frame(payload: dict[str, Any], expected: AudioFrameStats) -> AudioFrame:
    """Validate and decode a wire-format audio-frame payload.

    The transport delivers a dict with ``sample_rate``,
    ``channels``, ``frame_ms``, ``sequence``, ``encoding`` and a
    base64-encoded ``data`` field. This function enforces the
    expected audio shape (matching the
    :class:`AudioFrameStats` the collector was constructed with),
    rejects malformed payloads, and returns a ready-to-use
    :class:`AudioFrame`.

    Args:
        payload: Decoded JSON message from the client. Missing or
            wrong-typed fields surface as :class:`ValueError`.
        expected: Frame-shape expectations. The payload must match
            these on ``sample_rate``, ``channels`` and ``frame_ms``.

    Returns:
        A parsed :class:`AudioFrame` whose ``data`` is the decoded
        PCM bytes (already validated to be exactly
        ``expected_bytes`` long).

    Raises:
        ValueError: If the payload has an unexpected sample rate,
            channel count, frame duration, encoding, sequence
            number, base64 ``data`` field, or decoded byte length.
            WebSocket consumers should catch this and usually forward
            an ``audio.frame_error`` event containing the exception
            message so clients can drop the bad frame and continue.
    """
    sample_rate = int(payload.get("sample_rate", 0))
    channels = int(payload.get("channels", 0))
    frame_ms = int(payload.get("frame_ms", 0))
    sequence = int(payload.get("sequence", -1))
    encoding = str(payload.get("encoding", ""))
    encoded_data = payload.get("data")

    if sample_rate != expected.expected_sample_rate:
        raise ValueError(
            f"expected sample_rate {expected.expected_sample_rate}, got {sample_rate}"
        )
    if channels != expected.expected_channels:
        raise ValueError(
            f"expected channels {expected.expected_channels}, got {channels}"
        )
    if frame_ms != expected.expected_frame_ms:
        raise ValueError(
            f"expected frame_ms {expected.expected_frame_ms}, got {frame_ms}"
        )
    if encoding != SUPPORTED_ENCODING:
        raise ValueError(f"expected encoding {SUPPORTED_ENCODING}, got {encoding}")
    if sequence < 0:
        raise ValueError("sequence must be a non-negative integer")
    if not isinstance(encoded_data, str) or not encoded_data:
        raise ValueError("data must be a non-empty base64 string")

    try:
        data = base64.b64decode(encoded_data, validate=True)
    except Exception as exc:
        raise ValueError("data must be valid base64") from exc

    expected_samples = sample_rate * frame_ms // 1000
    expected_bytes = expected_samples * channels * 2
    if len(data) != expected_bytes:
        raise ValueError(f"expected {expected_bytes} audio bytes, got {len(data)}")

    return AudioFrame(
        data=data,
        sequence=sequence,
        sample_rate=sample_rate,
        channels=channels,
        frame_ms=frame_ms,
        encoding=encoding,
    )
