"""Tests for the audio.cpp HTTP providers (TTS + ASR).

These mirror the whisper.cpp provider contract:

* The provider module imports cleanly when ``httpx`` is installed.
* :func:`build_provider` returns the right concrete class for the
  ``audio-cpp`` ASR and TTS names.
* TTS :meth:`stream_audio` POSTs to ``/v1/audio/speech`` and yields the
  decoded WAV as a single PCM :class:`AudioChunk`.
* ASR :meth:`transcribe_audio` writes a temp WAV, POSTs its server-local
  path to ``/v1/audio/transcriptions``, and parses the ``text`` field.
* Missing ``httpx`` surfaces the ``converse-framework[audio-cpp]`` hint.
* :meth:`check_status` reports ``ready=True`` on a healthy ``/health``
  plus a matching ``/v1/models`` entry, and ``ready=False`` otherwise.
"""

from __future__ import annotations

import asyncio
import builtins
import importlib
import os
from typing import cast

import pytest

from converse_framework.audio_utils import make_tone_wav
from converse_framework.protocols import ASRProvider, ProviderStatus, TTSProvider
from converse_framework.registry import build_provider
from converse_framework.providers.unavailable import extra_hint_for

from converse_framework.providers.audio_cpp import (  # noqa: F401
    AudioCppASRProvider,
    AudioCppTTSProvider,
)

pytestmark = pytest.mark.httpx  # marker so the suite can opt-out via -m


def _tts(cfg: dict | None = None) -> AudioCppTTSProvider:
    return cast(
        AudioCppTTSProvider, build_provider("tts", "audio-cpp", cfg or {})
    )


def _asr(cfg: dict | None = None) -> AudioCppASRProvider:
    return cast(
        AudioCppASRProvider, build_provider("asr", "audio-cpp", cfg or {})
    )


async def _collect(aiter):
    out = []
    async for item in aiter:
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# Import + construction
# ---------------------------------------------------------------------------


def test_module_imports_when_httpx_available():
    """``converse_framework.providers.audio_cpp`` must import cleanly when
    ``httpx`` is present and expose both provider classes."""
    httpx = pytest.importorskip("httpx")
    module = importlib.import_module("converse_framework.providers.audio_cpp")
    assert hasattr(module, "AudioCppTTSProvider")
    assert hasattr(module, "AudioCppASRProvider")
    assert hasattr(httpx, "AsyncClient")


def test_module_imports_without_httpx():
    """The provider module must import even when httpx is absent (lazy deps)."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "httpx" or name.startswith("httpx."):
            raise ImportError("simulated missing httpx")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = fake_import
    try:
        module = importlib.import_module("converse_framework.providers.audio_cpp")
        importlib.reload(module)
    finally:
        builtins.__import__ = real_import
    assert hasattr(module, "AudioCppTTSProvider")


def test_build_provider_returns_tts_instance():
    pytest.importorskip("httpx")
    provider = _tts({"base_url": "http://example.invalid:8080", "model": "pocket-tts"})
    assert isinstance(provider, TTSProvider)
    assert provider.status.provider_id == "audio-cpp"
    assert provider.status.kind == "tts"
    assert provider.status.managed_externally is True


def test_build_provider_returns_asr_instance():
    pytest.importorskip("httpx")
    provider = _asr({"base_url": "http://example.invalid:8080", "model": "qwen3-asr"})
    assert isinstance(provider, ASRProvider)
    assert provider.status.provider_id == "audio-cpp"
    assert provider.status.kind == "asr"


def test_tts_default_config_values():
    pytest.importorskip("httpx")
    provider = _tts({})
    assert provider.base_url == "http://127.0.0.1:8080"
    assert provider.model == ""
    assert provider.timeout_s == 120.0


def test_asr_default_config_values():
    pytest.importorskip("httpx")
    provider = _asr({"model": "qwen3-asr"})
    assert provider.base_url == "http://127.0.0.1:8080"
    assert provider.model == "qwen3-asr"
    assert provider.cleanup_files is True
    assert provider.timeout_s == 120.0


# ---------------------------------------------------------------------------
# TTS stream_audio
# ---------------------------------------------------------------------------


def _fake_async_client_for_speech(captured: dict, wav_bytes: bytes):
    """Build a fake ``httpx.AsyncClient`` whose POST returns WAV bytes."""

    class _FakeResponse:
        def __init__(self) -> None:
            self.content = wav_bytes

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {}

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self) -> "_FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url, *args, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            return _FakeResponse()

    return _FakeAsyncClient


def test_tts_stream_audio_yields_decoded_pcm_chunk(monkeypatch):
    pytest.importorskip("httpx")
    provider = _tts(
        {"base_url": "http://server.test:8080", "model": "pocket-tts", "voice": "alba"}
    )
    wav_bytes = make_tone_wav(duration_s=0.05, sample_rate=16000)
    captured: dict = {}
    monkeypatch.setattr(
        "httpx.AsyncClient",
        _fake_async_client_for_speech(captured, wav_bytes),
        raising=False,
    )

    async def run():
        return [chunk async for chunk in provider.stream_audio("hello")]

    chunks = asyncio.run(run())
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.encoding == "pcm_s16le"
    assert chunk.mime_type is None
    assert chunk.final is True
    assert chunk.sample_rate == 16000
    assert chunk.duration_ms == 50
    assert len(chunk.data) > 0
    assert not chunk.data.startswith(b"RIFF")
    # PCM s16le is 2 bytes/sample; mono tone => samples*2 bytes.
    assert len(chunk.data) % 2 == 0
    # Request shape.
    assert captured["url"] == "http://server.test:8080/v1/audio/speech"
    payload = captured["kwargs"]["json"]
    assert payload["model"] == "pocket-tts"
    assert payload["input"] == "hello"
    assert payload["voice"] == "alba"


def test_tts_stream_audio_skips_whitespace_only():
    pytest.importorskip("httpx")
    provider = _tts({"model": "pocket-tts"})

    async def run():
        return [c async for c in provider.stream_audio("   ")]

    assert asyncio.run(run()) == []


def test_tts_stream_audio_requires_model(monkeypatch):
    pytest.importorskip("httpx")
    provider = _tts({})  # no model configured

    async def run():
        return [c async for c in provider.stream_audio("hi")]

    with pytest.raises(RuntimeError, match="requires a 'model'"):
        asyncio.run(run())


def test_tts_stream_audio_surfaces_extra_hint_when_httpx_missing(monkeypatch):
    """Missing httpx must surface the install hint, not a bare ImportError."""
    provider_module = importlib.import_module(
        "converse_framework.providers.audio_cpp"
    )
    provider = provider_module.AudioCppTTSProvider({"model": "pocket-tts"})
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "httpx" or name.startswith("httpx."):
            raise ImportError("simulated missing httpx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    async def run():
        return [c async for c in provider.stream_audio("hi")]

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(run())
    assert "converse-framework[audio-cpp]" in str(excinfo.value)
    assert "Install the required extra" in str(excinfo.value)


# ---------------------------------------------------------------------------
# ASR transcribe
# ---------------------------------------------------------------------------


def test_asr_transcribe_text_input_passthrough():
    pytest.importorskip("httpx")
    provider = _asr({"model": "qwen3-asr"})

    events = asyncio.run(_collect(provider.transcribe_text_input("hello world")))
    assert len(events) == 1
    assert events[0].final is True
    assert events[0].text == "hello world"


def test_asr_transcribe_text_input_skips_whitespace():
    pytest.importorskip("httpx")
    provider = _asr({"model": "qwen3-asr"})

    async def run():
        return [e async for e in provider.transcribe_text_input("   ")]

    assert asyncio.run(run()) == []


def test_asr_transcribe_audio_posts_path_and_parses_text(monkeypatch, tmp_path):
    """``transcribe_audio`` writes a temp WAV, POSTs its server-local path
    to ``/v1/audio/transcriptions``, parses the ``text`` field, and removes
    the temp file."""
    pytest.importorskip("httpx")
    provider = _asr(
        {
            "base_url": "http://server.test:8080",
            "model": "qwen3-asr",
            "language": "en",
            "shared_dir": str(tmp_path),
        }
    )

    captured: dict = {}

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"text": "hello from audio.cpp", "timing": {"wall_ms": 12.3}}

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self) -> "_FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url, *args, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            return _FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", _FakeAsyncClient, raising=False)

    async def run():
        pcm = b"\x00\x00" * 1600  # 0.1s of silence at 16 kHz mono
        return [e async for e in provider.transcribe_audio(pcm, 16000)]

    events = asyncio.run(run())
    assert len(events) == 1
    assert events[0].text == "hello from audio.cpp"
    assert events[0].final is True
    assert captured["url"] == "http://server.test:8080/v1/audio/transcriptions"
    payload = captured["kwargs"]["json"]
    assert payload["model"] == "qwen3-asr"
    assert payload["language"] == "en"
    # The audio field is a path written under the shared dir.
    audio_path = payload["audio"]
    assert os.path.dirname(audio_path) == str(tmp_path).rstrip("\\/")
    assert audio_path.endswith(".wav")
    # Temp file cleaned up after the request.
    assert not os.path.exists(audio_path)
    # The temp file was a real WAV before cleanup.
    assert captured.get("wrote_wav", True)


def test_asr_transcribe_audio_requires_model():
    pytest.importorskip("httpx")
    provider = _asr({})  # no model

    async def run():
        pcm = b"\x00\x00" * 16
        return [e async for e in provider.transcribe_audio(pcm, 16000)]

    with pytest.raises(RuntimeError, match="requires a 'model'"):
        asyncio.run(run())


def test_asr_transcribe_audio_rejects_invalid_sample_rate():
    pytest.importorskip("httpx")
    provider = _asr({"model": "qwen3-asr"})

    async def run():
        return [e async for e in provider.transcribe_audio(b"\x00\x00", 0)]

    with pytest.raises(ValueError, match="positive sample_rate"):
        asyncio.run(run())


def test_asr_transcribe_audio_yields_nothing_for_empty_pcm():
    pytest.importorskip("httpx")
    provider = _asr({"model": "qwen3-asr"})

    async def run():
        return [e async for e in provider.transcribe_audio(b"", 16000)]

    assert asyncio.run(run()) == []


def test_asr_transcribe_audio_surfaces_extra_hint_when_httpx_missing(monkeypatch):
    provider_module = importlib.import_module(
        "converse_framework.providers.audio_cpp"
    )
    provider = provider_module.AudioCppASRProvider({"model": "qwen3-asr"})
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "httpx" or name.startswith("httpx."):
            raise ImportError("simulated missing httpx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    async def run():
        pcm = b"\x00\x00" * 16
        return [e async for e in provider.transcribe_audio(pcm, 16000)]

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(run())
    assert "converse-framework[audio-cpp]" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Missing-dep hint table
# ---------------------------------------------------------------------------


def test_extra_hint_table_includes_audio_cpp():
    from converse_framework.providers.unavailable import EXTRA_HINTS

    assert ("asr", "audio-cpp") in EXTRA_HINTS
    assert ("tts", "audio-cpp") in EXTRA_HINTS
    assert extra_hint_for("asr", "audio-cpp") == "converse-framework[audio-cpp]"
    assert extra_hint_for("tts", "audio-cpp") == "converse-framework[audio-cpp]"


def test_unavailable_provider_for_audio_cpp_includes_hint():
    from converse_framework.providers.unavailable import UnavailableProvider

    for kind in ("asr", "tts"):
        p = UnavailableProvider(kind, "audio-cpp")
        assert "converse-framework[audio-cpp]" in p.status.message
        assert p.status.ready is False
        assert p.status.missing_extra == "audio-cpp"


# ---------------------------------------------------------------------------
# check_status
# ---------------------------------------------------------------------------


def _fake_async_client_for_status(
    base_url: str, model_ids: list[str], health_ok: bool = True, error: bool = False
):
    """Fake client answering ``/health`` and ``/v1/models``."""

    class _HealthResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"status": "ok" if health_ok else "bad", "models": len(model_ids)}

    class _ModelsResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "object": "list",
                "data": [{"id": mid, "object": "model"} for mid in model_ids],
            }

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self) -> "_FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url, *args, **kwargs):
            if error:
                raise ConnectionRefusedError("nope")
            if url.endswith("/health"):
                return _HealthResponse()
            if url.endswith("/v1/models"):
                return _ModelsResponse()
            raise AssertionError(f"unexpected GET {url}")

    return _FakeAsyncClient


def test_tts_check_status_requires_model_without_http(monkeypatch):
    pytest.importorskip("httpx")
    provider = _tts({"base_url": "http://server.test:8080"})

    class _Boom:
        def __init__(self, *args, **kwargs):
            raise AssertionError("missing model should not open an HTTP client")

    monkeypatch.setattr("httpx.AsyncClient", _Boom, raising=False)

    status = asyncio.run(provider.check_status())
    assert status.ready is False
    assert status.status_level == "error"
    assert "requires a 'model'" in status.message


def test_tts_check_status_ready_on_health_and_model_match(monkeypatch):
    pytest.importorskip("httpx")
    provider = _tts({"base_url": "http://server.test:8080", "model": "pocket-tts"})
    monkeypatch.setattr(
        "httpx.AsyncClient",
        _fake_async_client_for_status("http://server.test:8080", ["pocket-tts"]),
        raising=False,
    )

    status = asyncio.run(provider.check_status())
    assert status.ready is True
    assert status.status_level == "ready"
    assert "pocket-tts" in status.message


def test_tts_check_status_flags_unknown_model(monkeypatch):
    pytest.importorskip("httpx")
    provider = _tts({"base_url": "http://server.test:8080", "model": "missing"})
    monkeypatch.setattr(
        "httpx.AsyncClient",
        _fake_async_client_for_status("http://server.test:8080", ["pocket-tts"]),
        raising=False,
    )

    status = asyncio.run(provider.check_status())
    assert status.ready is False
    assert "not registered" in status.message


def test_tts_check_status_not_ready_on_connection_error(monkeypatch):
    pytest.importorskip("httpx")
    provider = _tts({"base_url": "http://nope.invalid:1", "model": "pocket-tts"})
    monkeypatch.setattr(
        "httpx.AsyncClient",
        _fake_async_client_for_status("http://nope.invalid:1", [], error=True),
        raising=False,
    )

    status = asyncio.run(provider.check_status())
    assert status.ready is False
    assert "Cannot reach audiocpp_server" in status.message


def test_asr_check_status_ready_on_health_and_model_match(monkeypatch):
    pytest.importorskip("httpx")
    provider = _asr({"base_url": "http://server.test:8080", "model": "qwen3-asr"})
    monkeypatch.setattr(
        "httpx.AsyncClient",
        _fake_async_client_for_status("http://server.test:8080", ["qwen3-asr"]),
        raising=False,
    )

    status = asyncio.run(provider.check_status())
    assert status.ready is True


def test_asr_check_status_requires_model_without_http(monkeypatch):
    pytest.importorskip("httpx")
    provider = _asr({"base_url": "http://server.test:8080"})

    class _Boom:
        def __init__(self, *args, **kwargs):
            raise AssertionError("missing model should not open an HTTP client")

    monkeypatch.setattr("httpx.AsyncClient", _Boom, raising=False)

    status = asyncio.run(provider.check_status())
    assert status.ready is False
    assert status.status_level == "error"
    assert "requires a 'model'" in status.message


def test_tts_probe_status_returns_cached_without_http(monkeypatch):
    """probe_status must not make an HTTP call; it only checks httpx import."""
    pytest.importorskip("httpx")
    provider = _tts({"model": "pocket-tts"})

    class _Boom:
        def __init__(self, *a, **k):
            raise AssertionError("probe_status must not open a client")

    monkeypatch.setattr("httpx.AsyncClient", _Boom, raising=False)
    status = asyncio.run(provider.probe_status())
    assert isinstance(status, ProviderStatus)
    # Cached "configured" state, not an error.
    assert status.status_level == "configured"
