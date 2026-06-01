"""Tests for audio utilities: conversion, frames, metering, trimming."""

import base64
import struct

import pytest

from converse_framework.audio_utils import (
    AudioFrameStats,
    compute_pcm16_level,
    float_audio_to_pcm_s16le_bytes,
    float_audio_to_wav_bytes,
    make_tone_wav,
    parse_audio_frame,
    pcm_s16le_to_float32,
    trim_pcm16_silence,
)


# ---------------------------------------------------------------------------
# make_tone_wav
# ---------------------------------------------------------------------------


def test_make_tone_wav_has_wav_header():
    data = make_tone_wav(duration_s=0.01)
    assert data[:4] == b"RIFF"
    assert data[8:12] == b"WAVE"
    assert b"data" in data[:44]


def test_make_tone_wav_different_frequencies():
    tone1 = make_tone_wav(frequency=440)
    tone2 = make_tone_wav(frequency=880)
    assert tone1 != tone2


# ---------------------------------------------------------------------------
# PCM conversion
# ---------------------------------------------------------------------------


def test_pcm_s16le_to_float32_empty():
    result = pcm_s16le_to_float32(b"")
    assert len(result) == 0


def test_pcm_s16le_to_float32_silence():
    data = struct.pack("<4h", 0, 0, 0, 0)
    result = pcm_s16le_to_float32(data)
    assert len(result) == 4
    assert all(v == 0.0 for v in result)


def test_pcm_s16le_to_float32_max():
    data = struct.pack("<2h", 32767, -32768)
    result = pcm_s16le_to_float32(data)
    assert result[0] == pytest.approx(1.0, abs=0.001)
    assert result[1] == pytest.approx(-1.0, abs=0.001)


def test_float_audio_to_pcm_s16le_bytes_roundtrip():
    import numpy as np

    original = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    pcm = float_audio_to_pcm_s16le_bytes(original)
    back = pcm_s16le_to_float32(pcm)
    assert len(back) == 5
    for a, b in zip(original, back, strict=True):
        assert a == pytest.approx(b, abs=0.001)


def test_float_audio_to_pcm_s16le_bytes_empty():
    import numpy as np

    assert float_audio_to_pcm_s16le_bytes(np.array([], dtype=np.float32)) == b""


def test_float_audio_to_wav_bytes_has_header():
    import numpy as np

    audio = np.zeros(160, dtype=np.float32)
    wav = float_audio_to_wav_bytes(audio, 16000)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"


def test_float_audio_to_wav_bytes_empty():
    import numpy as np

    assert float_audio_to_wav_bytes(np.array([], dtype=np.float32), 16000) == b""


# ---------------------------------------------------------------------------
# compute_pcm16_level
# ---------------------------------------------------------------------------


def test_compute_pcm16_level_reports_peak_and_rms():
    data = struct.pack("<4h", 0, 32767, -32768, 0)
    level = compute_pcm16_level(data)
    assert level["peak"] == 1.0
    assert 0.70 < level["rms"] < 0.72


def test_compute_pcm16_level_empty():
    level = compute_pcm16_level(b"")
    assert level["rms"] == 0.0
    assert level["peak"] == 0.0


# ---------------------------------------------------------------------------
# trim_pcm16_silence
# ---------------------------------------------------------------------------


def test_trim_pcm16_silence_removes_quiet_edges():
    quiet = [0] * 480
    speech = [1200] * 480
    data = struct.pack(f"<{len(quiet + speech + quiet)}h", *(quiet + speech + quiet))
    trimmed = trim_pcm16_silence(
        data, frame_ms=30, sample_rate=16000, rms_threshold=0.003
    )
    assert trimmed == struct.pack(f"<{len(speech)}h", *speech)


def test_trim_pcm16_silence_returns_empty_for_all_quiet_audio():
    data = struct.pack("<480h", *([0] * 480))
    trimmed = trim_pcm16_silence(
        data, frame_ms=30, sample_rate=16000, rms_threshold=0.003
    )
    assert trimmed == b""


def test_trim_pcm16_silence_no_threshold():
    data = struct.pack("<4h", 0, 0, 0, 0)
    assert (
        trim_pcm16_silence(data, frame_ms=30, sample_rate=16000, rms_threshold=0)
        == data
    )


def test_trim_pcm16_silence_empty_input():
    assert (
        trim_pcm16_silence(b"", frame_ms=30, sample_rate=16000, rms_threshold=0.003)
        == b""
    )


# ---------------------------------------------------------------------------
# parse_audio_frame
# ---------------------------------------------------------------------------


def make_payload(sequence=0, sample_rate=16000, channels=1, frame_ms=30, samples=None):
    if samples is None:
        samples = [0] * (sample_rate * frame_ms // 1000)
    data = struct.pack(f"<{len(samples)}h", *samples)
    return {
        "encoding": "pcm_s16le",
        "sample_rate": sample_rate,
        "channels": channels,
        "frame_ms": frame_ms,
        "sequence": sequence,
        "data": base64.b64encode(data).decode("ascii"),
    }


def test_parse_audio_frame_accepts_expected_pcm():
    stats = AudioFrameStats(
        expected_sample_rate=16000, expected_channels=1, expected_frame_ms=30
    )
    frame = parse_audio_frame(make_payload(sequence=3), stats)
    assert frame.sequence == 3
    assert len(frame.data) == 960


def test_parse_audio_frame_rejects_wrong_sample_rate():
    stats = AudioFrameStats(
        expected_sample_rate=16000, expected_channels=1, expected_frame_ms=30
    )
    with pytest.raises(ValueError, match="sample_rate"):
        parse_audio_frame(make_payload(sample_rate=48000), stats)


def test_parse_audio_frame_rejects_wrong_channels():
    stats = AudioFrameStats(
        expected_sample_rate=16000, expected_channels=1, expected_frame_ms=30
    )
    with pytest.raises(ValueError, match="channels"):
        parse_audio_frame(make_payload(channels=2), stats)


def test_parse_audio_frame_rejects_wrong_encoding():
    stats = AudioFrameStats(
        expected_sample_rate=16000, expected_channels=1, expected_frame_ms=30
    )
    payload = make_payload()
    payload["encoding"] = "float32"
    with pytest.raises(ValueError, match="encoding"):
        parse_audio_frame(payload, stats)


def test_parse_audio_frame_rejects_negative_sequence():
    stats = AudioFrameStats(
        expected_sample_rate=16000, expected_channels=1, expected_frame_ms=30
    )
    with pytest.raises(ValueError, match="sequence"):
        parse_audio_frame(make_payload(sequence=-1), stats)


def test_parse_audio_frame_rejects_empty_data():
    stats = AudioFrameStats(
        expected_sample_rate=16000, expected_channels=1, expected_frame_ms=30
    )
    payload = make_payload()
    payload["data"] = ""
    with pytest.raises(ValueError, match="data"):
        parse_audio_frame(payload, stats)


def test_parse_audio_frame_rejects_wrong_byte_count():
    stats = AudioFrameStats(
        expected_sample_rate=16000, expected_channels=1, expected_frame_ms=30
    )
    payload = make_payload(samples=[0] * 100)  # 200 bytes, expected 960
    with pytest.raises(ValueError, match="audio bytes"):
        parse_audio_frame(payload, stats)


# ---------------------------------------------------------------------------
# AudioFrameStats
# ---------------------------------------------------------------------------


def test_audio_frame_stats_tracks_dropped_frames():
    stats = AudioFrameStats(
        expected_sample_rate=16000, expected_channels=1, expected_frame_ms=30
    )
    first = parse_audio_frame(make_payload(sequence=0), stats)
    second = parse_audio_frame(make_payload(sequence=3), stats)
    stats.update(first)
    stats.last_emit_ts = 0
    metrics = stats.update(second)
    assert metrics is not None
    assert metrics["dropped_frames"] == 2


def test_audio_frame_stats_rate_limits_emits():
    stats = AudioFrameStats(
        expected_sample_rate=16000, expected_channels=1, expected_frame_ms=30
    )
    stats.last_emit_ts = 0  # ensure first update passes rate limit
    frame = parse_audio_frame(make_payload(sequence=0), stats)
    m1 = stats.update(frame)
    m2 = stats.update(frame)
    assert m1 is not None
    assert m2 is None  # rate-limited (within 100ms)


def test_audio_frame_stats_initial_metrics():
    stats = AudioFrameStats(
        expected_sample_rate=16000, expected_channels=1, expected_frame_ms=30
    )
    stats.last_emit_ts = 0  # ensure first update passes rate limit
    frame = parse_audio_frame(make_payload(sequence=0), stats)
    metrics = stats.update(frame)
    assert metrics is not None
    assert metrics["received_frames"] == 1
    assert metrics["dropped_frames"] == 0
    assert "rms" in metrics
    assert "peak" in metrics
