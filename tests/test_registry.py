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

    async def probe_status(self):
        return self.status

    async def load_status(self):
        return await self.load()

    async def load(self):
        _LOAD_CALL_COUNTS[self.kind_name] += 1
        return self.status

    async def unload(self):
        return self.status

    async def process_frame(self, frame) -> list[VADEvent]:
        return []

    async def transcribe_text_input(self, text: str) -> AsyncIterator[TranscriptEvent]:
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
    assert p.first_token_delay == 0.042  # type: ignore[attr-defined]


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
    bundle = build_provider_bundle(config, tts_provider=override)  # type: ignore[arg-type]
    assert bundle.tts is override
    assert bundle.tts.first_chunk_delay == 0.999  # type: ignore[attr-defined]


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
    assert bundle.vad.config.get("sample_rate") == 24000  # type: ignore[attr-defined]


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


# ---------------------------------------------------------------------------
# Phase 2: probe_statuses / load_statuses / new status fields
# ---------------------------------------------------------------------------


def test_provider_bundle_probe_statuses_does_not_call_load(monkeypatch):
    """probe_statuses() must not call load() on any provider."""
    from converse_framework import registry as registry_module

    _reset_load_call_counts()

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

    bundle = build_provider_bundle(
        {
            "vad": {"provider": "counting"},
            "asr": {"provider": "counting"},
            "llm": {"provider": "counting"},
            "tts": {"provider": "counting"},
        }
    )

    statuses = asyncio.run(bundle.probe_statuses())
    assert len(statuses) == 4
    for count in _LOAD_CALL_COUNTS.values():
        assert count == 0


def test_provider_bundle_load_statuses_calls_load():
    """load_statuses() should trigger load() on providers that expose it."""
    bundle = build_provider_bundle(
        {
            "vad": {"provider": "mock"},
            "asr": {"provider": "mock"},
            "llm": {"provider": "mock"},
            "tts": {"provider": "mock"},
        }
    )

    statuses = asyncio.run(bundle.load_statuses())
    assert len(statuses) == 4
    for s in statuses:
        assert s["ready"] is True


def test_serialized_status_includes_phase2_fields():
    """Serialized status dict must include voices, models, status_level."""
    from converse_framework.registry import _serialize_status

    status = ProviderStatus(
        name="test",
        kind="vad",
        ready=True,
        message="test",
        capabilities=ProviderCapabilities(),
        voices=({"id": "v1", "label": "Voice 1"},),
        active_voice="v1",
        models=({"id": "m1", "label": "Model 1"},),
        active_model="m1",
        status_level="configured",
    )
    result = _serialize_status(status)
    assert "voices" in result
    assert "active_voice" in result
    assert "models" in result
    assert "active_model" in result
    assert "status_level" in result
    assert result["active_voice"] == "v1"
    assert result["active_model"] == "m1"
    assert result["status_level"] == "configured"
    # voices/models are serialised as lists
    assert isinstance(result["voices"], list)
    assert len(result["voices"]) == 1


def test_unavailable_provider_has_error_status_level():
    """UnavailableProvider reports status_level='unavailable'."""
    from converse_framework.providers.unavailable import UnavailableProvider

    p = UnavailableProvider("asr", "nonexistent")
    assert p.status.status_level == "unavailable"


def test_faster_whisper_status_level_reflects_model_state():
    """FasterWhisperASRProvider status_level changes with model state."""
    from converse_framework.providers.faster_whisper import FasterWhisperASRProvider

    provider = FasterWhisperASRProvider({"model": "tiny", "language": "en"})
    # No model → configured
    assert provider.status.status_level == "configured"

    # With injected model → ready
    provider = FasterWhisperASRProvider(
        {
            "model": "tiny",
            "language": "en",
            "_model": object(),
        }
    )
    assert provider.status.status_level == "ready"

    # With load error → error
    provider._load_error = "simulated failure"
    assert provider.status.status_level == "error"


def test_pocket_tts_status_includes_voices_and_voice():
    """PocketTTSProvider status includes voices and active_voice."""
    from converse_framework.providers.pocket_tts import PocketTTSProvider

    provider = PocketTTSProvider({"voice": "azelma"})
    s = provider.status
    assert s.active_voice == "azelma"
    assert len(s.voices) > 0
    # Should include azelma in the list
    voice_ids = [v["id"] for v in s.voices]
    assert "azelma" in voice_ids


def test_kokoro_status_includes_active_voice():
    """KokoroOnnxProvider status includes active_voice."""
    from converse_framework.providers.kokoro_onnx import KokoroOnnxProvider

    provider = KokoroOnnxProvider({"voice": "af_heart", "_model": object()})
    s = provider.status
    assert s.active_voice == "af_heart"


def test_silero_status_level_reflects_model_state():
    """SileroVADProvider status_level reflects whether model is loaded."""
    from converse_framework.providers.silero import SileroVADProvider

    provider = SileroVADProvider({})
    assert provider.status.status_level == "configured"

    # With error
    provider._load_error = "failed"
    assert provider.status.status_level == "error"


def test_whisper_cpp_status_includes_active_model():
    """WhisperCppASRProvider status includes active_model."""
    from converse_framework.providers.whisper_cpp import WhisperCppASRProvider

    provider = WhisperCppASRProvider({"model": "ggml-small.en.bin"})
    s = provider.status
    assert s.active_model == "ggml-small.en.bin"


def test_llamacpp_status_level_is_configured_by_default():
    """LlamaCppProvider status_level is 'configured' by default."""
    from converse_framework.providers.llamacpp import LlamaCppProvider

    provider = LlamaCppProvider({"base_url": "http://localhost:9999"})
    s = provider.status
    assert s.status_level == "configured"


# ---------------------------------------------------------------------------
# Phase 3: ProviderBundle.replace / unload_replaced
# ---------------------------------------------------------------------------


def test_provider_bundle_replace_creates_new_bundle():
    """replace() returns a new bundle with specified providers swapped."""
    bundle = build_provider_bundle({})
    original_vad = bundle.vad
    original_tts = bundle.tts

    # Replace TTS only
    new_tts = build_provider("tts", "mock", {"first_chunk_delay_ms": 500})
    replaced = bundle.replace(tts=new_tts)

    assert replaced.tts is new_tts
    assert replaced.vad is original_vad
    assert replaced.asr is bundle.asr
    assert replaced.llm is bundle.llm
    # Original bundle unaffected
    assert bundle.tts is original_tts


def test_provider_bundle_replace_replaces_multiple_providers():
    """replace() with multiple kwargs swaps all specified providers."""
    bundle = build_provider_bundle({})
    new_asr = build_provider("asr", "mock")
    new_llm = build_provider("llm", "mock")

    replaced = bundle.replace(asr=new_asr, llm=new_llm)
    assert replaced.asr is new_asr
    assert replaced.llm is new_llm
    assert replaced.vad is bundle.vad
    assert replaced.tts is bundle.tts


def test_provider_bundle_replace_no_args_returns_copy():
    """replace() with no args returns a bundle sharing the same providers."""
    bundle = build_provider_bundle({})
    same = bundle.replace()
    assert same.vad is bundle.vad
    assert same.asr is bundle.asr
    assert same.llm is bundle.llm
    assert same.tts is bundle.tts


def test_provider_bundle_unload_replaced_unloads_replaced_providers():
    """unload_replaced() calls unload on providers that differ by identity."""
    bundle = build_provider_bundle({})
    new_asr = build_provider("asr", "mock")
    replaced = bundle.replace(asr=new_asr)

    statuses = asyncio.run(ProviderBundle.unload_replaced(bundle, replaced))
    # Only ASR was replaced so only one unload result expected
    assert len(statuses) == 1
    assert statuses[0]["name"] == bundle.asr.status.name


def test_provider_bundle_unload_replaced_skips_identical_providers():
    """unload_replaced() does nothing when bundles share same provider refs."""
    bundle = build_provider_bundle({})
    same = bundle.replace()
    statuses = asyncio.run(ProviderBundle.unload_replaced(bundle, same))
    assert len(statuses) == 0
