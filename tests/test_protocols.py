"""Tests for protocol instantiation via mock implementations."""

from converse_framework.protocols import (
    ASRProvider,
    LLMProvider,
    ProviderCapabilities,
    ProviderStatus,
    TTSProvider,
    VADProvider,
)
from converse_framework.providers.mock import (
    MockASRProvider,
    MockLLMProvider,
    MockTTSProvider,
    MockVADProvider,
)


def test_mock_vad_is_vad_provider():
    provider = MockVADProvider({})
    assert isinstance(provider, VADProvider)


def test_mock_asr_is_asr_provider():
    provider = MockASRProvider({})
    assert isinstance(provider, ASRProvider)


def test_mock_llm_is_llm_provider():
    provider = MockLLMProvider({})
    assert isinstance(provider, LLMProvider)


def test_mock_tts_is_tts_provider():
    provider = MockTTSProvider({})
    assert isinstance(provider, TTSProvider)


def test_mock_providers_report_ready_status():
    vad = MockVADProvider({})
    asr = MockASRProvider({})
    llm = MockLLMProvider({})
    tts = MockTTSProvider({})

    for p in (vad, asr, llm, tts):
        assert p.status.ready
        assert isinstance(p.status, ProviderStatus)
        assert isinstance(p.status.capabilities, ProviderCapabilities)


def test_mock_vad_has_barge_in_capability():
    vad = MockVADProvider({})
    assert vad.status.capabilities.supports_barge_in


def test_mock_asr_has_partials_capability():
    asr = MockASRProvider({})
    assert asr.status.capabilities.supports_partials


def test_mock_tts_has_streaming_capability():
    tts = MockTTSProvider({})
    assert tts.status.capabilities.supports_streaming_tts
