"""Tests for lazy provider registry and mock provider bundle."""

import asyncio
import sys
from collections.abc import AsyncIterator

from converse_framework.protocols import (
    ASRProvider,
    AudioChunk,
    LLMProvider,
    ProviderCapabilities,
    ProviderStatus,
    TTSProvider,
    TranscriptEvent,
    VADEvent,
    VADProvider,
)
from converse_framework.registry import (
    ProviderBundle,
    _serialize_status,
    build_provider,
    build_provider_bundle,
    is_provider_available,
    register_provider,
    status_only,
)


# ---------------------------------------------------------------------------
# Counting provider stubs (module-level so they can be looked up via the
# registry's import-string mechanism). ``status_only`` must never call
# ``load()`` on any provider; each subclass increments a shared counter
# on load so the test can assert the count stays at zero.
# ---------------------------------------------------------------------------

_LOAD_CALL_COUNTS: dict[str, int] = {"vad": 0, "asr": 0, "llm": 0, "tts": 0}


def _reset_load_call_counts() -> None:
    for key in _LOAD_CALL_COUNTS:
        _LOAD_CALL_COUNTS[key] = 0


class _CountingProvider:
    """Provider stub that tracks ``load()`` invocations per kind.

    Only ``check_status`` is exercised by ``status_only``; the remaining
    methods are present so the class is a complete protocol
    implementation and can be instantiated via ``build_provider``.
    """

    kind_name: str = ""

    def __init__(self, config):
        self.config = config

    @property
    def status(self):
        return ProviderStatus(
            name=f"counting-{self.kind_name}",
            kind=self.kind_name,
            ready=True,
            message="counting",
            capabilities=ProviderCapabilities(),
        )

    async def check_status(self):
        return self.status

    async def load(self):
        _LOAD_CALL_COUNTS[self.kind_name] += 1
        return self.status

    async def unload(self):
        return self.status

    async def process_frame(self, frame) -> list[VADEvent]:
        return []

    async def transcribe_text_input(
        self, text: str
    ) -> AsyncIterator[TranscriptEvent]:
        return
        yield  # pragma: no cover

    async def transcribe_audio(
        self, pcm_s16le: bytes, sample_rate: int, progress=None
    ) -> AsyncIterator[TranscriptEvent]:
        return
        yield  # pragma: no cover

    async def stream_response(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        return
        yield  # pragma: no cover

    async def stream_audio(self, text: str) -> AsyncIterator[AudioChunk]:
        return
        yield  # pragma: no cover

    async def stream_audio_with_progress(
        self, text: str, progress=None
    ) -> AsyncIterator[AudioChunk]:
        return
        yield  # pragma: no cover


class _CountingVAD(_CountingProvider):
    kind_name = "vad"


class _CountingASR(_CountingProvider):
    kind_name = "asr"


class _CountingLLM(_CountingProvider):
    kind_name = "llm"


class _CountingTTS(_CountingProvider):
    kind_name = "tts"


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


# ---------------------------------------------------------------------------
# status_only
# ---------------------------------------------------------------------------


def test_status_only_returns_four_entries_in_order():
    config = {
        "vad": {"provider": "mock"},
        "asr": {"provider": "mock"},
        "llm": {"provider": "mock"},
        "tts": {"provider": "mock"},
    }
    statuses = asyncio.run(status_only(config))
    assert len(statuses) == 4
    assert [s["kind"] for s in statuses] == ["vad", "asr", "llm", "tts"]


def test_status_only_serializes_using_existing_shape():
    config = {
        "vad": {"provider": "mock"},
        "asr": {"provider": "mock"},
        "llm": {"provider": "mock"},
        "tts": {"provider": "mock"},
    }
    statuses = asyncio.run(status_only(config))
    # Build the expected key set from the actual helper, so the test
    # stays in sync with any future addition to _serialize_status.
    sample = _serialize_status(
        ProviderStatus(
            name="sample",
            kind="vad",
            ready=True,
            message="",
            capabilities=ProviderCapabilities(),
        )
    )
    expected_keys = set(sample.keys())
    for status in statuses:
        assert expected_keys.issubset(status.keys())


def test_status_only_does_not_call_load(monkeypatch):
    from converse_framework import registry as registry_module

    _reset_load_call_counts()

    # Replace each kind's registry entries with a single counting
    # provider. monkeypatch restores the original entries at teardown
    # so other tests are unaffected.
    kind_to_class = {
        "vad": "_CountingVAD",
        "asr": "_CountingASR",
        "llm": "_CountingLLM",
        "tts": "_CountingTTS",
    }
    for kind, class_name in kind_to_class.items():
        monkeypatch.setitem(
            registry_module._registry,
            kind,
            {
                "counting": registry_module._ProviderEntry(
                    import_path=f"test_registry:{class_name}",
                    availability_probe=None,
                )
            },
        )

    config = {
        "vad": {"provider": "counting"},
        "asr": {"provider": "counting"},
        "llm": {"provider": "counting"},
        "tts": {"provider": "counting"},
    }
    statuses = asyncio.run(status_only(config))

    assert len(statuses) == 4
    for count in _LOAD_CALL_COUNTS.values():
        assert count == 0


def test_status_only_with_missing_extra_returns_unavailable(monkeypatch):
    # Make the framework's faster-whisper provider module unimportable
    # so ``build_provider`` returns an ``UnavailableProvider`` sentinel
    # whose status includes the install hint.
    monkeypatch.setitem(
        sys.modules, "converse_framework.providers.faster_whisper", None
    )

    config = {"asr": {"provider": "faster-whisper"}}
    statuses = asyncio.run(status_only(config))

    asr_status = next(s for s in statuses if s["kind"] == "asr")
    assert asr_status["ready"] is False
    assert "converse-framework[faster-whisper]" in asr_status["message"]
    assert asr_status["install_hint"] == "converse-framework[faster-whisper]"
    assert asr_status["missing_extra"] == "faster-whisper"
