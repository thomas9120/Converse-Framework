"""Tests for the generic OpenAI-compatible providers (LLM, ASR, TTS).

Contract:

* ``build_provider(kind, "openai-compatible", ...)`` returns the right
  concrete class for each of ``llm`` / ``asr`` / ``tts``.
* ``api_key`` config produces an ``Authorization: Bearer`` header on
  every HTTP client; no ``api_key`` means no auth header.
* LLM ``check_status`` goes straight to ``/v1/models`` (no llama.cpp
  ``/health`` probe), while :class:`LlamaCppProvider` keeps probing
  ``/health`` first. ASR/TTS ``check_status`` also uses ``/v1/models``.
* ASR ``transcribe_audio`` uploads a multipart WAV to
  ``/v1/audio/transcriptions`` and parses the ``text`` field.
* TTS ``stream_audio`` POSTs to ``/v1/audio/speech`` with
  ``response_format: "wav"`` and yields the decoded PCM as a single
  final :class:`AudioChunk`.
* The ``converse-framework[openai-compat]`` extra hint is registered
  for all three kinds.
"""

from __future__ import annotations

import asyncio
from typing import cast

import pytest

from converse_framework.audio_utils import make_tone_wav
from converse_framework.providers.llamacpp import LlamaCppProvider
from converse_framework.providers.openai_compat import (
    OpenAICompatASRProvider,
    OpenAICompatLLMProvider,
    OpenAICompatTTSProvider,
)
from converse_framework.providers.unavailable import extra_hint_for
from converse_framework.registry import build_provider, is_provider_available

pytestmark = pytest.mark.httpx


def _provider(cfg: dict | None = None) -> OpenAICompatLLMProvider:
    return cast(
        OpenAICompatLLMProvider,
        build_provider("llm", "openai-compatible", cfg or {}),
    )


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


def _fake_async_client(requested_urls: list[str], captured_kwargs: dict):
    """Build a fake httpx.AsyncClient class that records GET urls."""

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            captured_kwargs.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url: str):
            requested_urls.append(url)
            if url.endswith("/health"):
                return _FakeResponse({"status": "ok"})
            if url.endswith("/v1/models"):
                return _FakeResponse({"data": [{"id": "test-model"}]})
            raise AssertionError(f"unexpected GET {url}")

    return _FakeAsyncClient


# ---------------------------------------------------------------------------
# Registration + construction
# ---------------------------------------------------------------------------


def test_build_provider_returns_openai_compat_instance():
    pytest.importorskip("httpx")
    provider = _provider({"base_url": "http://x:1"})
    assert isinstance(provider, OpenAICompatLLMProvider)
    assert isinstance(provider, LlamaCppProvider)
    assert provider.status.provider_id == "openai-compatible"
    assert provider.status.name == "openai-compatible"
    assert provider.status.kind == "llm"


def test_provider_is_registered_and_available():
    pytest.importorskip("httpx")
    assert is_provider_available("llm", "openai-compatible") is True


def test_extra_hint_registered():
    assert (
        extra_hint_for("llm", "openai-compatible")
        == "converse-framework[openai-compat]"
    )


def test_base_url_trailing_slash_stripped():
    pytest.importorskip("httpx")
    provider = _provider({"base_url": "https://api.openai.com/"})
    assert provider.base_url == "https://api.openai.com"


# ---------------------------------------------------------------------------
# api_key -> Authorization header
# ---------------------------------------------------------------------------


def test_api_key_sets_bearer_header():
    provider = _provider({"api_key": "sk-test"})
    assert provider._headers() == {"Authorization": "Bearer sk-test"}


def test_no_api_key_means_no_auth_header():
    provider = _provider({})
    assert provider._headers() == {}


def test_stream_client_carries_auth_header():
    pytest.importorskip("httpx")
    provider = _provider({"api_key": "sk-test"})
    client = provider._ensure_stream_client()
    try:
        assert client.headers.get("Authorization") == "Bearer sk-test"
    finally:
        asyncio.run(provider.unload())


def test_llamacpp_also_accepts_api_key():
    """llama.cpp server supports --api-key, so the base provider takes one."""
    provider = cast(
        LlamaCppProvider,
        build_provider("llm", "llamacpp", {"api_key": "local-secret"}),
    )
    assert provider._headers() == {"Authorization": "Bearer local-secret"}


# ---------------------------------------------------------------------------
# check_status endpoint selection
# ---------------------------------------------------------------------------


def test_check_status_skips_health_endpoint(monkeypatch):
    httpx = pytest.importorskip("httpx")
    requested: list[str] = []
    captured: dict = {}
    monkeypatch.setattr(
        httpx, "AsyncClient", _fake_async_client(requested, captured)
    )

    provider = _provider({"base_url": "http://x:1", "api_key": "sk-test"})
    status = asyncio.run(provider.check_status())

    assert status.ready is True
    assert "test-model" in status.message
    assert requested == ["http://x:1/v1/models"]
    assert captured["headers"] == {"Authorization": "Bearer sk-test"}


def test_llamacpp_check_status_still_probes_health(monkeypatch):
    httpx = pytest.importorskip("httpx")
    requested: list[str] = []
    captured: dict = {}
    monkeypatch.setattr(
        httpx, "AsyncClient", _fake_async_client(requested, captured)
    )

    provider = cast(
        LlamaCppProvider, build_provider("llm", "llamacpp", {"base_url": "http://x:1"})
    )
    status = asyncio.run(provider.check_status())

    assert status.ready is True
    assert requested == ["http://x:1/health", "http://x:1/v1/models"]


def test_check_status_reports_unreachable_models(monkeypatch):
    httpx = pytest.importorskip("httpx")

    class _FailingClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url: str):
            raise ConnectionError("refused")

    monkeypatch.setattr(httpx, "AsyncClient", _FailingClient)

    provider = _provider({"base_url": "http://x:1"})
    status = asyncio.run(provider.check_status())

    assert status.ready is False
    assert "/v1/models" in status.message


# ---------------------------------------------------------------------------
# Sampler contract inherited from LlamaCppProvider
# ---------------------------------------------------------------------------


def test_sampler_overrides_merge_over_defaults():
    provider = _provider({"temperature": 0.5, "max_tokens": 128})
    provider.set_sampler_provider(lambda: {"top_p": 0.9})
    assert provider._build_sampler() == {
        "temperature": 0.5,
        "max_tokens": 128,
        "top_p": 0.9,
    }


# ---------------------------------------------------------------------------
# ASR + TTS: registration and construction
# ---------------------------------------------------------------------------


def _asr(cfg: dict | None = None) -> OpenAICompatASRProvider:
    return cast(
        OpenAICompatASRProvider,
        build_provider("asr", "openai-compatible", cfg or {}),
    )


def _tts(cfg: dict | None = None) -> OpenAICompatTTSProvider:
    return cast(
        OpenAICompatTTSProvider,
        build_provider("tts", "openai-compatible", cfg or {}),
    )


async def _collect(aiter):
    out = []
    async for item in aiter:
        out.append(item)
    return out


def _fake_post_client(captured: dict, response):
    """Fake httpx.AsyncClient whose post() records kwargs and returns *response*."""

    class _FakePostClient:
        def __init__(self, *args, **kwargs):
            captured["client_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, **kwargs):
            captured["url"] = url
            captured["post_kwargs"] = kwargs
            return response

        async def get(self, url):
            raise AssertionError(f"unexpected GET {url}")

    return _FakePostClient


def test_build_provider_returns_asr_and_tts_instances():
    pytest.importorskip("httpx")
    asr = _asr({"base_url": "http://x:1", "model": "whisper-1"})
    tts = _tts({"base_url": "http://x:1", "model": "tts-1", "voice": "alloy"})
    assert isinstance(asr, OpenAICompatASRProvider)
    assert isinstance(tts, OpenAICompatTTSProvider)
    assert asr.status.provider_id == "openai-compatible"
    assert asr.status.kind == "asr"
    assert asr.status.managed_externally is True
    assert tts.status.provider_id == "openai-compatible"
    assert tts.status.kind == "tts"
    assert tts.status.active_voice == "alloy"


def test_asr_and_tts_registered_and_available():
    pytest.importorskip("httpx")
    assert is_provider_available("asr", "openai-compatible") is True
    assert is_provider_available("tts", "openai-compatible") is True


def test_asr_and_tts_extra_hints_registered():
    assert (
        extra_hint_for("asr", "openai-compatible")
        == "converse-framework[openai-compat]"
    )
    assert (
        extra_hint_for("tts", "openai-compatible")
        == "converse-framework[openai-compat]"
    )


# ---------------------------------------------------------------------------
# ASR + TTS: check_status via /v1/models
# ---------------------------------------------------------------------------


def test_asr_check_status_uses_models_endpoint(monkeypatch):
    httpx = pytest.importorskip("httpx")
    requested: list[str] = []
    captured: dict = {}
    monkeypatch.setattr(
        httpx, "AsyncClient", _fake_async_client(requested, captured)
    )

    provider = _asr({"base_url": "http://x:1", "model": "test-model", "api_key": "k"})
    status = asyncio.run(provider.check_status())

    assert status.ready is True
    assert requested == ["http://x:1/v1/models"]
    assert captured["headers"] == {"Authorization": "Bearer k"}


def test_tts_check_status_rejects_unknown_model(monkeypatch):
    httpx = pytest.importorskip("httpx")
    requested: list[str] = []
    captured: dict = {}
    monkeypatch.setattr(
        httpx, "AsyncClient", _fake_async_client(requested, captured)
    )

    provider = _tts({"base_url": "http://x:1", "model": "not-registered"})
    status = asyncio.run(provider.check_status())

    assert status.ready is False
    assert "not-registered" in status.message
    assert "test-model" in status.message


# ---------------------------------------------------------------------------
# ASR: transcribe_audio multipart upload
# ---------------------------------------------------------------------------


class _FakeJSONResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


def test_asr_transcribe_audio_uploads_multipart_wav(monkeypatch):
    httpx = pytest.importorskip("httpx")
    captured: dict = {}
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        _fake_post_client(captured, _FakeJSONResponse({"text": " hello world "})),
    )

    provider = _asr(
        {
            "base_url": "http://x:1",
            "model": "whisper-1",
            "language": "en",
            "api_key": "sk-test",
        }
    )
    pcm = b"\x00\x01" * 1600  # 0.1 s of 16 kHz mono s16le
    events = asyncio.run(_collect(provider.transcribe_audio(pcm, 16000)))

    assert captured["url"] == "http://x:1/v1/audio/transcriptions"
    assert captured["client_kwargs"]["headers"] == {"Authorization": "Bearer sk-test"}
    form = captured["post_kwargs"]["data"]
    assert form["model"] == "whisper-1"
    assert form["language"] == "en"
    assert form["response_format"] == "json"
    filename, wav_bytes, content_type = captured["post_kwargs"]["files"]["file"]
    assert filename == "audio.wav"
    assert wav_bytes[:4] == b"RIFF"
    assert content_type == "audio/wav"
    assert len(events) == 1
    assert events[0].final is True
    assert events[0].text == "hello world"


def test_asr_transcribe_audio_empty_pcm_yields_nothing():
    pytest.importorskip("httpx")
    provider = _asr({})
    events = asyncio.run(_collect(provider.transcribe_audio(b"", 16000)))
    assert events == []


def test_asr_transcribe_audio_rejects_bad_sample_rate():
    pytest.importorskip("httpx")
    provider = _asr({})

    async def run():
        return [e async for e in provider.transcribe_audio(b"\x00\x00", 0)]

    with pytest.raises(ValueError):
        asyncio.run(run())


def test_asr_transcribe_text_input_passthrough():
    pytest.importorskip("httpx")
    provider = _asr({})
    events = asyncio.run(_collect(provider.transcribe_text_input("  hi  ")))
    assert len(events) == 1
    assert events[0].text == "hi"
    assert events[0].final is True


# ---------------------------------------------------------------------------
# TTS: /v1/audio/speech WAV decoding
# ---------------------------------------------------------------------------


class _FakeWavResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self) -> None:
        pass


def test_tts_stream_audio_decodes_wav_to_pcm_chunk(monkeypatch):
    httpx = pytest.importorskip("httpx")
    captured: dict = {}
    wav = make_tone_wav(duration_s=0.05, sample_rate=16000)
    monkeypatch.setattr(
        httpx, "AsyncClient", _fake_post_client(captured, _FakeWavResponse(wav))
    )

    provider = _tts(
        {
            "base_url": "http://x:1",
            "model": "tts-1",
            "voice": "alloy",
            "api_key": "sk-test",
        }
    )
    chunks = asyncio.run(_collect(provider.stream_audio("Hello there.")))

    assert captured["url"] == "http://x:1/v1/audio/speech"
    assert captured["client_kwargs"]["headers"] == {"Authorization": "Bearer sk-test"}
    payload = captured["post_kwargs"]["json"]
    assert payload["input"] == "Hello there."
    assert payload["model"] == "tts-1"
    assert payload["voice"] == "alloy"
    assert payload["response_format"] == "wav"
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.encoding == "pcm_s16le"
    assert chunk.sample_rate == 16000
    assert chunk.channels == 1
    assert chunk.final is True
    assert len(chunk.data) > 0


def test_tts_stream_audio_raises_on_non_wav_response(monkeypatch):
    httpx = pytest.importorskip("httpx")
    captured: dict = {}
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        _fake_post_client(captured, _FakeWavResponse(b"not audio at all")),
    )

    provider = _tts({"base_url": "http://x:1", "model": "tts-1", "voice": "alloy"})
    with pytest.raises(RuntimeError, match="WAV"):
        asyncio.run(_collect(provider.stream_audio("Hello.")))


def test_tts_stream_audio_empty_text_yields_nothing():
    pytest.importorskip("httpx")
    provider = _tts({})
    chunks = asyncio.run(_collect(provider.stream_audio("   ")))
    assert chunks == []
