"""Tests for lazy provider registry and mock provider bundle."""

import asyncio

from converse_framework.protocols import (
    ASRProvider,
    LLMProvider,
    TTSProvider,
    VADProvider,
)
from converse_framework.registry import (
    ProviderBundle,
    build_provider,
    build_provider_bundle,
    is_provider_available,
    register_provider,
)


# ---------------------------------------------------------------------------
# is_provider_available
# ---------------------------------------------------------------------------


def test_mock_providers_are_available():
    assert is_provider_available("vad", "mock")
    assert is_provider_available("asr", "mock")
    assert is_provider_available("llm", "mock")
    assert is_provider_available("tts", "mock")


def test_unknown_provider_is_not_available():
    assert not is_provider_available("vad", "nonexistent-xyz")
    assert not is_provider_available("asr", "nonexistent-xyz")


# ---------------------------------------------------------------------------
# build_provider
# ---------------------------------------------------------------------------


def test_build_provider_mock_vad():
    p = build_provider("vad", "mock")
    assert isinstance(p, VADProvider)


def test_build_provider_mock_asr():
    p = build_provider("asr", "mock")
    assert isinstance(p, ASRProvider)


def test_build_provider_mock_llm():
    p = build_provider("llm", "mock")
    assert isinstance(p, LLMProvider)


def test_build_provider_mock_tts():
    p = build_provider("tts", "mock")
    assert isinstance(p, TTSProvider)


def test_build_provider_passes_config():
    p = build_provider("llm", "mock", {"first_token_delay_ms": 42})
    assert p.first_token_delay == 0.042


def test_build_provider_unknown_returns_unavailable():
    p = build_provider("vad", "nonexistent-xyz")
    assert not p.status.ready
    assert not p.status.loaded
    assert "nonexistent-xyz" in p.status.message


# ---------------------------------------------------------------------------
# register_provider
# ---------------------------------------------------------------------------


def test_register_provider_rejects_unknown_kind():
    try:
        register_provider("invalid", "test", "some.module:SomeClass")
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "Unknown provider kind" in str(e)


# ---------------------------------------------------------------------------
# build_provider_bundle
# ---------------------------------------------------------------------------


def test_build_provider_bundle_returns_provider_bundle():
    config = {
        "vad": {"provider": "mock"},
        "asr": {"provider": "mock"},
        "llm": {"provider": "mock"},
        "tts": {"provider": "mock"},
    }
    bundle = build_provider_bundle(config)
    assert isinstance(bundle, ProviderBundle)
    assert isinstance(bundle.vad, VADProvider)
    assert isinstance(bundle.asr, ASRProvider)
    assert isinstance(bundle.llm, LLMProvider)
    assert isinstance(bundle.tts, TTSProvider)


def test_build_provider_bundle_defaults_to_mock():
    bundle = build_provider_bundle({})
    assert bundle.vad.status.ready
    assert bundle.asr.status.ready
    assert bundle.llm.status.ready
    assert bundle.tts.status.ready


def test_build_provider_bundle_tts_provider_override():
    config = {
        "tts": {"provider": "mock"},
    }
    # Build a separate mock TTS to inject
    override = build_provider("tts", "mock", {"first_chunk_delay_ms": 999})
    bundle = build_provider_bundle(config, tts_provider=override)
    assert bundle.tts is override
    assert bundle.tts.first_chunk_delay == 0.999


def test_build_provider_bundle_audio_sample_rate_default():
    config = {
        "vad": {"provider": "mock"},
        "asr": {"provider": "mock"},
        "llm": {"provider": "mock"},
        "tts": {"provider": "mock"},
        "audio": {"sample_rate": 24000},
    }
    bundle = build_provider_bundle(config)
    # VAD config should have sample_rate from audio section
    assert bundle.vad.config.get("sample_rate") == 24000


# ---------------------------------------------------------------------------
# ProviderBundle statuses
# ---------------------------------------------------------------------------


def test_provider_bundle_statuses():
    config = {
        "vad": {"provider": "mock"},
        "asr": {"provider": "mock"},
        "llm": {"provider": "mock"},
        "tts": {"provider": "mock"},
    }
    bundle = build_provider_bundle(config)
    statuses = bundle.statuses()
    assert len(statuses) == 4
    kinds = {s["kind"] for s in statuses}
    assert kinds == {"vad", "asr", "llm", "tts"}
    for s in statuses:
        assert s["ready"] is True
        assert "capabilities" in s


def test_provider_bundle_check_statuses():
    async def run():
        config = {
            "vad": {"provider": "mock"},
            "asr": {"provider": "mock"},
            "llm": {"provider": "mock"},
            "tts": {"provider": "mock"},
        }
        bundle = build_provider_bundle(config)
        statuses = await bundle.check_statuses()
        return statuses

    statuses = asyncio.run(run())
    assert len(statuses) == 4
    for s in statuses:
        assert s["ready"] is True
