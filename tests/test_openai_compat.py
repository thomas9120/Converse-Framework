"""Tests for the generic OpenAI-compatible LLM provider.

Contract:

* ``build_provider("llm", "openai-compatible", ...)`` returns
  :class:`OpenAICompatLLMProvider` with its own provider id.
* ``api_key`` config produces an ``Authorization: Bearer`` header on
  both the status-check client and the persistent streaming client;
  no ``api_key`` means no auth header.
* ``check_status`` goes straight to ``/v1/models`` (no llama.cpp
  ``/health`` probe), while :class:`LlamaCppProvider` keeps probing
  ``/health`` first.
* The ``converse-framework[openai-compat]`` extra hint is registered.
"""

from __future__ import annotations

import asyncio
from typing import cast

import pytest

from converse_framework.providers.llamacpp import LlamaCppProvider
from converse_framework.providers.openai_compat import OpenAICompatLLMProvider
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
