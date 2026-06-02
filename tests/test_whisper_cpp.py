"""Tests for the whisper.cpp ASR provider.

These tests cover the Phase 6 contract:

* The provider module imports cleanly when ``httpx`` is installed.
* :func:`build_provider` returns a :class:`WhisperCppASRProvider` when
  asked for the ``whisper-cpp`` ASR provider.
* :meth:`transcribe_text_input` yields a single final transcript.
* :meth:`transcribe_audio` posts WAV bytes to ``/inference`` and parses
  the returned ``{"text": ...}`` payload.
* A missing ``httpx`` import surfaces the whisper-cpp extra hint rather
  than a bare ``ImportError``.
* :meth:`check_status` reports ``ready=True`` on a 200 from ``/health``
  and ``ready=False`` on a connection error.
"""

from __future__ import annotations

import asyncio
import builtins
import importlib
from typing import cast

import pytest

from converse_framework.protocols import ASRProvider, ProviderStatus
from converse_framework.registry import build_provider
from converse_framework.providers.unavailable import extra_hint_for

from converse_framework.providers.whisper_cpp import WhisperCppASRProvider  # noqa: F401

pytestmark = pytest.mark.httpx  # marker so the suite can opt-out via -m


def _wprovider(cfg: dict | None = None):
    """Build whisper-cpp provider cast to its concrete type."""
    return cast(WhisperCppASRProvider, build_provider("asr", "whisper-cpp", cfg or {}))


# ---------------------------------------------------------------------------
# Import + construction
# ---------------------------------------------------------------------------


def test_module_imports_when_httpx_available():
    """``converse_framework.providers.whisper_cpp`` must import cleanly
    when ``httpx`` is present and expose the provider class."""
    httpx = pytest.importorskip("httpx")
    module = importlib.import_module("converse_framework.providers.whisper_cpp")
    assert hasattr(module, "WhisperCppASRProvider")
    assert module.WhisperCppASRProvider.__name__ == "WhisperCppASRProvider"
    # Sanity: httpx is the only heavy dep the module reaches for.
    assert hasattr(httpx, "AsyncClient")


def test_build_provider_returns_whisper_cpp_instance():
    pytest.importorskip("httpx")
    provider = _wprovider({"base_url": "http://example.invalid:8082"})
    assert isinstance(provider, ASRProvider)
    assert provider.status.provider_id == "whisper-cpp"
    assert provider.status.kind == "asr"
    assert provider.status.ready is True
    assert "whisper-server" in provider.status.message
    assert provider.base_url == "http://example.invalid:8082"


def test_default_config_values():
    pytest.importorskip("httpx")
    provider = _wprovider({})
    assert provider.base_url == "http://127.0.0.1:8082"
    assert provider.model == "ggml-small.en.bin"
    assert provider.language == "en"
    assert provider.temperature == 0
    assert provider.timeout_s == 120.0


def test_provider_advertises_no_model_or_voice_management():
    pytest.importorskip("httpx")
    provider = _wprovider({})
    status = provider.status
    assert status.supports_model_management is False
    assert status.supports_voice_selection is False
    assert status.managed_externally is True
    assert status.loaded is False


# ---------------------------------------------------------------------------
# transcribe_text_input
# ---------------------------------------------------------------------------


def test_transcribe_text_input_yields_single_final_event():
    pytest.importorskip("httpx")
    provider = _wprovider({})

    async def run():
        events = [event async for event in provider.transcribe_text_input("hi")]

    events = asyncio.run(_collect(provider.transcribe_text_input("hello world")))
    assert len(events) == 1
    assert events[0].final is True
    assert events[0].text == "hello world"


def test_transcribe_text_input_skips_whitespace_only():
    pytest.importorskip("httpx")
    provider = _wprovider({})

    async def _collect_empty():
        return [event async for event in provider.transcribe_text_input("   ")]

    events = asyncio.run(_collect_empty())
    assert events == []


async def _collect(aiter):
    out = []
    async for item in aiter:
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# transcribe_audio: posts to /inference and parses the response
# ---------------------------------------------------------------------------


def test_transcribe_audio_posts_to_inference_and_parses_text(monkeypatch):
    """``transcribe_audio`` must POST WAV bytes to ``/inference`` and
    surface the ``text`` field of the JSON response."""
    pytest.importorskip("httpx")
    provider = build_provider(
        "asr",
        "whisper-cpp",
        {"base_url": "http://server.test:8082", "language": "en", "temperature": 0},
    )

    captured: dict = {}

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"text": "hello"}

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

    async def run() -> list:
        # 0.05s of silence at 16 kHz mono -> 1600 samples -> 3200 bytes.
        pcm = b"\x00\x00" * 1600
        events = []
        async for event in provider.transcribe_audio(pcm, 16000):  # type: ignore[attr-defined]
            events.append(event)
        return events

    # Force the provider into the legacy /inference path (skip the
    # endpoint probe so the test does not need a real HTTP server).
    provider._endpoint = "/inference"  # type: ignore[attr-defined]

    monkeypatch.setattr("httpx.AsyncClient", _FakeAsyncClient, raising=False)

    events = asyncio.run(run())
    assert len(events) == 1
    assert events[0].text == "hello"
    assert events[0].final is True
    assert captured["url"] == "http://server.test:8082/inference"
    # Body is the WAV bytes; language + temperature are query params.
    body = captured["kwargs"].get("content")
    assert isinstance(body, (bytes, bytearray))
    assert body[:4] == b"RIFF"
    assert captured["kwargs"].get("params", {}).get("language") == "en"
    assert captured["kwargs"].get("params", {}).get("temperature") == "0"


def test_transcribe_audio_falls_back_to_openai_endpoint(monkeypatch):
    """When the server only exposes ``/v1/audio/transcriptions`` the
    provider must POST a multipart form and still parse the ``text``."""
    pytest.importorskip("httpx")
    provider = build_provider(
        "asr",
        "whisper-cpp",
        {"base_url": "http://server.test:8082", "language": "en"},
    )

    captured: dict = {}

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"text": "ok"}

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

    async def run():
        provider._endpoint = "/v1/audio/transcriptions"  # type: ignore[attr-defined]
        pcm = b"\x00\x00" * 1600
        events = [event async for event in provider.transcribe_audio(pcm, 16000)]  # type: ignore[attr-defined]
        return events

    monkeypatch.setattr("httpx.AsyncClient", _FakeAsyncClient, raising=False)

    events = asyncio.run(run())
    assert events[0].text == "ok"
    assert captured["url"] == "http://server.test:8082/v1/audio/transcriptions"
    assert "files" in captured["kwargs"]
    assert "data" in captured["kwargs"]
    assert captured["kwargs"]["data"].get("response_format") == "json"


def test_transcribe_audio_rejects_invalid_sample_rate():
    pytest.importorskip("httpx")
    provider = _wprovider({})

    async def run():
        events = []
        async for event in provider.transcribe_audio(b"\x00\x00", 0):
            events.append(event)
        return events

    with pytest.raises(ValueError, match="positive sample_rate"):
        asyncio.run(run())


def test_transcribe_audio_yields_nothing_for_empty_pcm():
    pytest.importorskip("httpx")
    provider = _wprovider({})

    async def run():
        events = []
        async for event in provider.transcribe_audio(b"", 16000):
            events.append(event)
        return events

    assert asyncio.run(run()) == []


# ---------------------------------------------------------------------------
# Missing-dep friendly message
# ---------------------------------------------------------------------------


def test_transcribe_audio_surfaces_extra_hint_when_httpx_missing(monkeypatch):
    """When ``httpx`` cannot be imported, the provider must surface
    the ``converse-framework[whisper-cpp]`` install hint, not a bare
    ``ImportError``."""
    provider_module = importlib.import_module(
        "converse_framework.providers.whisper_cpp"
    )
    provider_module = importlib.reload(provider_module)
    provider = provider_module.WhisperCppASRProvider({})

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "httpx" or name.startswith("httpx."):
            raise ImportError("simulated missing httpx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    async def run():
        events = []
        async for event in provider.transcribe_audio(b"\x00\x00" * 16, 16000):
            events.append(event)
        return events

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(run())
    assert "converse-framework[whisper-cpp]" in str(excinfo.value)
    assert "Install the required extra" in str(excinfo.value)


def test_extra_hint_table_includes_whisper_cpp():
    """The provider registry's hint table must list whisper-cpp so
    :class:`UnavailableProvider` can build a friendly message."""
    from converse_framework.providers.unavailable import EXTRA_HINTS

    assert ("asr", "whisper-cpp") in EXTRA_HINTS
    assert extra_hint_for("asr", "whisper-cpp") == "converse-framework[whisper-cpp]"


def test_unavailable_provider_for_whisper_cpp_includes_hint():
    """An :class:`UnavailableProvider` constructed for the whisper-cpp
    ASR name must mention the install extra."""
    from converse_framework.providers.unavailable import UnavailableProvider

    p = UnavailableProvider("asr", "whisper-cpp")
    assert "converse-framework[whisper-cpp]" in p.status.message
    assert "whisper-cpp" in p.status.message
    assert p.status.ready is False


# ---------------------------------------------------------------------------
# check_status
# ---------------------------------------------------------------------------


def test_check_status_ready_on_health_200(monkeypatch):
    pytest.importorskip("httpx")
    provider = _wprovider({"base_url": "http://server.test:8082"})

    class _OkResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self) -> "_FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url, *args, **kwargs):
            assert url == "http://server.test:8082/health"
            return _OkResponse()

    monkeypatch.setattr("httpx.AsyncClient", _FakeAsyncClient, raising=False)

    status = asyncio.run(provider.check_status())
    assert isinstance(status, ProviderStatus)
    assert status.ready is True
    assert "reachable" in status.message
    assert provider._endpoint == "/inference"  # type: ignore[attr-defined]


def test_check_status_not_ready_on_connection_error(monkeypatch):
    pytest.importorskip("httpx")
    provider = _wprovider({"base_url": "http://nope.invalid:1"})

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self) -> "_FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url, *args, **kwargs):
            raise ConnectionRefusedError("nope")

    monkeypatch.setattr("httpx.AsyncClient", _FakeAsyncClient, raising=False)

    status = asyncio.run(provider.check_status())
    assert status.ready is False
    assert "Cannot reach whisper-server" in status.message
    assert provider._endpoint is None  # type: ignore[attr-defined]
