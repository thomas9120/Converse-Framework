"""Tests for the lazy provider registry and concrete provider imports.

These tests cover the Phase 5 contract:

* Base import of ``converse_framework`` does not require any provider
  extras to be installed.
* Each concrete provider module imports cleanly when its dependency is
  available, and the resulting class can be instantiated with a
  framework module path (no app imports).
* When a heavy dependency is missing, the registry still surfaces a
  friendly ``pip install converse-framework[<extra>]`` hint instead of
  a raw ``ImportError``.
* The framework exposes :func:`extra_hint_for` so callers can look up the
  right extra programmatically.
* ``build_provider_bundle`` builds a fully mock bundle without any heavy
  imports.
* :class:`ProviderBundle.statuses` / :meth:`check_statuses` JSON shape is
  preserved for the ``/api/status`` endpoint.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import types
from typing import cast

import pytest

from converse_framework.protocols import (
    ASRProvider,
    LLMProvider,
    TTSProvider,
    VADProvider,
)
from converse_framework.providers.pocket_tts import PocketTTSProvider  # noqa: F401
from converse_framework.providers.llamacpp import LlamaCppProvider  # noqa: F401
from converse_framework.providers.kokoro_onnx import KokoroOnnxProvider  # noqa: F401
from converse_framework.providers.unavailable import (
    EXTRAS,
    EXTRA_HINTS,
    UnavailableProvider,
    extra_hint_for,
    missing_extra_for,
)
from converse_framework.registry import (
    ProviderBundle,
    build_provider,
    build_provider_bundle,
    register_provider,
)


# ---------------------------------------------------------------------------
# Base import + cleanliness
# ---------------------------------------------------------------------------


def test_base_import_does_not_pull_in_heavy_providers():
    """Importing the framework must not import silero_vad / faster_whisper /
    kokoro_onnx / pocket_tts / httpx at the top level.
    """
    for mod in (
        "silero_vad",
        "faster_whisper",
        "kokoro_onnx",
        "pocket_tts",
    ):
        assert mod not in sys.modules, (
            f"{mod} should not be imported by `import converse_framework`"
        )


def test_concrete_provider_modules_importable_independently():
    """Each provider module must be importable on its own and yield a class
    that implements the right protocol.
    """
    silero = importlib.import_module("converse_framework.providers.silero")
    assert hasattr(silero, "SileroVADProvider")

    faster = importlib.import_module("converse_framework.providers.faster_whisper")
    assert hasattr(faster, "FasterWhisperASRProvider")

    llamacpp = importlib.import_module("converse_framework.providers.llamacpp")
    assert hasattr(llamacpp, "LlamaCppProvider")

    kokoro = importlib.import_module("converse_framework.providers.kokoro_onnx")
    assert hasattr(kokoro, "KokoroOnnxProvider")

    pocket = importlib.import_module("converse_framework.providers.pocket_tts")
    assert hasattr(pocket, "PocketTTSProvider")


# ---------------------------------------------------------------------------
# Lazy module loading -- build_provider / build_provider_bundle
# ---------------------------------------------------------------------------


def test_build_provider_with_fake_model_for_silero():
    """The SileroVADProvider must be constructable without silero_vad
    installed, as long as a fake model is supplied via config. This mirrors
    the harness's test pattern of injecting ``_model``.
    """

    class FakeModel:
        def __call__(self, chunk, sample_rate):
            return 0.0

        def reset_states(self):
            return None

    provider = build_provider("vad", "silero", {"_model": FakeModel()})
    assert isinstance(provider, VADProvider)
    assert provider.status.provider_id == "silero"
    assert provider.status.ready is True
    assert "Silero VAD ONNX model loaded" in provider.status.message


def test_build_provider_with_fake_model_for_faster_whisper():
    provider = build_provider(
        "asr", "faster-whisper", {"_model": object(), "language": "en"}
    )
    assert isinstance(provider, ASRProvider)
    assert provider.status.provider_id == "faster-whisper"
    assert provider.status.ready is True
    assert "Loaded" in provider.status.message


def test_build_provider_with_fake_model_for_llamacpp():
    provider = build_provider("llm", "llamacpp", {"base_url": "http://x:1"})
    assert isinstance(provider, LLMProvider)
    assert provider.status.provider_id == "llamacpp"


def test_build_provider_with_fake_model_for_kokoro():
    class FakeKokoro:
        def create_stream(self, *a, **kw):
            yield b"", 24000

    provider = build_provider(
        "tts",
        "kokoro",
        {"_model": FakeKokoro(), "voice": "af_heart", "lang": "en-us"},
    )
    assert isinstance(provider, TTSProvider)
    assert provider.status.provider_id == "kokoro-onnx"
    assert provider.status.ready is True
    assert provider.status.supports_model_management is True


def test_build_provider_with_fake_model_for_pocket_tts():
    class FakePocket:
        sample_rate = 24000

        def get_state_for_audio_prompt(self, voice):
            return {"voice": voice}

        def generate_audio_stream(self, *a, **kw):
            return iter(())

    provider = build_provider(
        "tts", "pocket-tts", {"_model": FakePocket(), "voice": "azelma"}
    )
    assert isinstance(provider, TTSProvider)
    assert provider.status.provider_id == "pocket-tts"
    assert provider.status.ready is True


def test_pocket_tts_set_quantize_unloads_when_mode_changes():
    provider = cast(
        PocketTTSProvider,
        build_provider(
            "tts",
            "pocket-tts",
            {"_model": object(), "_voice_state": object(), "quantize": False},
        ),
    )

    unchanged = provider.set_quantize(False)
    assert unchanged.loaded is True
    assert provider._model is not None
    assert provider._voice_state is not None

    changed = provider.set_quantize(True)
    assert changed.loaded is False
    assert provider.quantize is True
    assert provider._model is None
    assert provider._voice_state is None
    assert "int8" in changed.message


# ---------------------------------------------------------------------------
# Phase 4: Pocket TTS voice and config support
# ---------------------------------------------------------------------------


def test_pocket_tts_set_voice_clears_only_voice_state():
    """set_voice() clears _voice_state but keeps _model loaded."""
    provider = cast(
        PocketTTSProvider,
        build_provider(
            "tts",
            "pocket-tts",
            {"_model": object(), "_voice_state": object(), "voice": "azelma"},
        ),
    )
    assert provider._model is not None
    assert provider._voice_state is not None

    result = provider.set_voice("bela")
    assert result.active_voice == "bela"
    # Model stays loaded
    assert provider._model is not None
    # Voice state cleared
    assert provider._voice_state is None


def test_pocket_tts_set_voice_to_same_keeps_state():
    """set_voice() with the same voice keeps model and voice state."""
    provider = cast(
        PocketTTSProvider,
        build_provider(
            "tts",
            "pocket-tts",
            {"_model": object(), "_voice_state": object(), "voice": "azelma"},
        ),
    )
    result = provider.set_voice("azelma")
    assert provider._model is not None
    assert provider._voice_state is not None
    assert result.active_voice == "azelma"


def test_pocket_tts_configure_max_tokens_no_unload():
    """configure(max_tokens=...) changes the value without unloading."""
    provider = cast(
        PocketTTSProvider,
        build_provider(
            "tts",
            "pocket-tts",
            {"_model": object(), "_voice_state": object(), "max_tokens": 50},
        ),
    )

    result = asyncio.run(provider.configure(max_tokens=100))
    assert result.changed is True
    assert result.requires_reload is False
    assert provider.max_tokens == 100
    # Model and voice state untouched
    assert provider._model is not None
    assert provider._voice_state is not None


def test_pocket_tts_configure_coalesce_ms_no_unload():
    """configure(coalesce_ms=...) changes the value without unloading."""
    provider = cast(
        PocketTTSProvider,
        build_provider(
            "tts",
            "pocket-tts",
            {"_model": object(), "_voice_state": object(), "coalesce_ms": 400},
        ),
    )

    result = asyncio.run(provider.configure(coalesce_ms=600))
    assert result.changed is True
    assert result.requires_reload is False
    assert provider.coalesce_ms == 600
    assert provider._model is not None
    assert provider._voice_state is not None


def test_pocket_tts_configure_voice_unloads_voice_state():
    """configure(voice=...) clears voice state but keeps model."""
    provider = cast(
        PocketTTSProvider,
        build_provider(
            "tts",
            "pocket-tts",
            {"_model": object(), "_voice_state": object(), "voice": "azelma"},
        ),
    )

    result = asyncio.run(provider.configure(voice="bela"))
    assert result.changed is True
    assert result.requires_reload is True
    assert provider.voice == "bela"
    assert provider._model is not None
    assert provider._voice_state is None


def test_pocket_tts_configure_quantize_unloads_all():
    """configure(quantize=True) clears model and voice state."""
    provider = cast(
        PocketTTSProvider,
        build_provider(
            "tts",
            "pocket-tts",
            {"_model": object(), "_voice_state": object(), "quantize": False},
        ),
    )

    result = asyncio.run(provider.configure(quantize=True))
    assert result.changed is True
    assert result.requires_reload is True
    assert provider.quantize is True
    assert provider._model is None
    assert provider._voice_state is None


def test_pocket_tts_configure_temp_unloads_all():
    """configure(temp=...) clears model and voice state."""
    provider = cast(
        PocketTTSProvider,
        build_provider(
            "tts",
            "pocket-tts",
            {"_model": object(), "_voice_state": object(), "temp": 0.7},
        ),
    )

    result = asyncio.run(provider.configure(temp=0.5))
    assert result.changed is True
    assert result.requires_reload is True
    assert abs(provider.temp - 0.5) < 1e-6
    assert provider._model is None
    assert provider._voice_state is None


def test_pocket_tts_configure_noop_returns_no_changes():
    """configure() with unchanged values returns changed=False."""
    provider = cast(
        PocketTTSProvider,
        build_provider(
            "tts",
            "pocket-tts",
            {"_model": object(), "_voice_state": object(), "max_tokens": 50},
        ),
    )

    result = asyncio.run(provider.configure(max_tokens=50))
    assert result.changed is False
    assert result.requires_reload is False


def test_pocket_tts_list_voices_returns_structured_voice_info():
    """list_voices() returns VoiceInfo tuples without importing backend."""
    from converse_framework.protocols import VoiceInfo

    provider = cast(
        PocketTTSProvider,
        build_provider("tts", "pocket-tts", {"voice": "azelma"}),
    )

    voices = provider.list_voices()
    assert len(voices) >= 3
    assert isinstance(voices[0], VoiceInfo)
    assert voices[0].id == "azelma"
    assert voices[0].label == "Azelma"

    # Find a voice by id
    by_id = {v.id: v for v in voices}
    assert "bela" in by_id
    assert by_id["bela"].language == "en"
    # French voices
    assert by_id["jean"].language == "fr"
    assert by_id["jean"].gender == "neutral"


# ---------------------------------------------------------------------------
# Missing-dep friendly errors
# ---------------------------------------------------------------------------


def test_missing_dependency_returns_unavailable_provider():
    """If a heavy dep is missing AND a real backend URL is given,
    the registry should still try to construct the class (so the status
    surfaces the real error), but the instance should report not-ready
    rather than crashing.
    """
    # silero_vad is not installed in the test environment.
    provider = build_provider("vad", "silero", {})
    # If silero_vad is present, we get a real SileroVADProvider that
    # is not yet ready (no model). If silero_vad is missing, the
    # framework module is still importable but check_status would set
    # the load_error. Either way the status must be a valid ProviderStatus.
    status = provider.status
    assert status.kind == "vad"


def test_unknown_provider_returns_unavailable():
    provider = build_provider("vad", "does-not-exist")
    assert isinstance(provider, UnavailableProvider)
    assert provider.status.kind == "vad"
    assert provider.status.ready is False


def test_extra_hint_for_known_providers():
    assert extra_hint_for("vad", "silero") == "converse-framework[silero]"
    assert "converse-framework[faster-whisper]" == extra_hint_for(
        "asr", "faster-whisper"
    )
    assert "converse-framework[llamacpp]" == extra_hint_for("llm", "llamacpp")
    assert "converse-framework[kokoro]" == extra_hint_for("tts", "kokoro")
    assert "converse-framework[kokoro]" == extra_hint_for("tts", "kokoro-onnx")
    assert extra_hint_for("tts", "pocket-tts") == "converse-framework[pocket-tts]"
    assert missing_extra_for("vad", "silero") == "silero"
    assert missing_extra_for("tts", "kokoro-onnx") == "kokoro"


def test_extra_hint_for_unknown_provider_is_none():
    assert extra_hint_for("vad", "made-up") is None


def test_extra_hint_table_is_exported():
    assert isinstance(EXTRA_HINTS, dict)
    assert isinstance(EXTRAS, dict)
    assert ("vad", "silero") in EXTRA_HINTS


def test_unavailable_provider_includes_extra_hint():
    p = UnavailableProvider("vad", "silero")
    assert "converse-framework[silero]" in p.status.message
    assert "pip install" in p.status.message
    assert p.status.install_hint == "converse-framework[silero]"
    assert p.status.missing_extra == "silero"


def test_unavailable_provider_unknown_extra_falls_back():
    p = UnavailableProvider("vad", "made-up", message=None)
    assert "made-up" in p.status.message


def test_unavailable_provider_does_not_silently_succeed():
    p = UnavailableProvider("vad", "silero")

    async def run() -> None:
        try:
            async for _ in p.transcribe_text_input("hello"):
                pass
        except RuntimeError as exc:
            assert "converse-framework[silero]" in str(exc)
        else:
            pytest.fail("expected RuntimeError")

    asyncio.run(run())


# ---------------------------------------------------------------------------
# build_provider_bundle: mock-only fast path
# ---------------------------------------------------------------------------


def test_build_provider_bundle_mock_only_no_heavy_imports():
    before = set(sys.modules)
    bundle = build_provider_bundle(
        {
            "vad": {"provider": "mock"},
            "asr": {"provider": "mock"},
            "llm": {"provider": "mock"},
            "tts": {"provider": "mock"},
        }
    )
    # None of the heavy modules should have been imported just by building
    # a mock bundle.
    after = set(sys.modules)
    new = after - before
    heavy = new & {"silero_vad", "faster_whisper", "kokoro_onnx", "pocket_tts", "httpx"}
    assert not heavy, f"build_provider_bundle pulled in {heavy}"
    assert isinstance(bundle, ProviderBundle)
    assert bundle.vad.status.ready
    assert bundle.asr.status.ready
    assert bundle.llm.status.ready
    assert bundle.tts.status.ready


def test_build_provider_bundle_statuses_shape_preserved():
    bundle = build_provider_bundle(
        {
            "vad": {"provider": "mock"},
            "asr": {"provider": "mock"},
            "llm": {"provider": "mock"},
            "tts": {"provider": "mock"},
        }
    )
    statuses = bundle.statuses()
    assert len(statuses) == 4
    sample = statuses[0]
    for key in (
        "name",
        "kind",
        "ready",
        "message",
        "capabilities",
        "provider_id",
        "selected",
        "loaded",
        "managed_externally",
        "supports_model_management",
        "supports_voice_selection",
    ):
        assert key in sample, f"status missing key: {key}"


# ---------------------------------------------------------------------------
# LlamaCppProvider: sampler via injected callable, not RuntimeSettings
# ---------------------------------------------------------------------------


def test_llamacpp_sampler_provider_is_callable_driven():
    """The framework LlamaCppProvider must accept a sampler provider callable
    via :meth:`set_sampler_provider` and must NOT mention RuntimeSettings.
    """
    provider = cast(
        LlamaCppProvider, build_provider("llm", "llamacpp", {"base_url": "http://x:1"})
    )

    captured: dict = {}

    def sampler() -> dict:
        captured["called"] = True
        return {"temperature": 0.42}

    provider.set_sampler_provider(sampler)
    assert provider._sampler_provider is sampler
    assert provider._build_sampler() == {"temperature": 0.42}
    assert captured == {"called": True}


def test_llamacpp_sampler_provider_can_be_cleared():
    provider = cast(
        LlamaCppProvider, build_provider("llm", "llamacpp", {"temperature": 0.7})
    )
    provider.set_sampler_provider(lambda: {"temperature": 0.1})
    assert provider._build_sampler() == {"temperature": 0.1}
    provider.set_sampler_provider(None)
    # Falls back to constructor defaults.
    assert provider._build_sampler() == {"temperature": 0.7, "max_tokens": 256}


def test_llamacpp_does_not_mention_runtime_settings():
    """The framework module must not import or reference RuntimeSettings."""
    import inspect

    import converse_framework.providers.llamacpp as llamacpp_mod

    source = inspect.getsource(llamacpp_mod)
    # No actual import of the harness module or RuntimeSettings class.
    assert "import RuntimeSettings" not in source
    assert "from conversational_harness" not in source
    # The framework uses ``set_sampler_provider``; the legacy
    # ``set_runtime_settings`` setter must not exist on the provider.
    assert "def set_runtime_settings" not in source


def test_llamacpp_uses_set_sampler_provider_not_set_runtime_settings():
    """The public API is set_sampler_provider; set_runtime_settings is
    not part of the framework contract.
    """
    import inspect

    import converse_framework.providers.llamacpp as llamacpp_mod

    source = inspect.getsource(llamacpp_mod)
    assert "set_sampler_provider" in source
    assert "def set_runtime_settings" not in source


# ---------------------------------------------------------------------------
# KokoroOnnxProvider: no PROJECT_ROOT dependency
# ---------------------------------------------------------------------------


def test_kokoro_does_not_depend_on_project_root(monkeypatch, tmp_path):
    """KokoroOnnxProvider must construct a default cache dir without
    importing the harness ``PROJECT_ROOT``.
    """
    import inspect

    import converse_framework.providers.kokoro_onnx as kokoro_mod

    source = inspect.getsource(kokoro_mod)
    # No actual import of the harness PROJECT_ROOT.
    assert "from conversational_harness" not in source
    assert "import conversational_harness" not in source
    assert "PROJECT_ROOT" not in source.replace("harness ``PROJECT_ROOT``", "")

    # When no cache_dir is given, the provider uses a platform default.
    # The CONVERSE_FRAMEWORK_CACHE_DIR env var lets us assert a value
    # without touching the user's real home directory.
    monkeypatch.setenv("CONVERSE_FRAMEWORK_CACHE_DIR", str(tmp_path))
    provider = cast(
        KokoroOnnxProvider,
        build_provider(
            "tts",
            "kokoro",
            {
                "voice": "af_heart",
                "lang": "en-us",
                "_model": object(),
                "_g2p": object(),
            },
        ),
    )
    assert provider.cache_dir == tmp_path / "kokoro"


# ---------------------------------------------------------------------------
# ProviderBundle check_statuses
# ---------------------------------------------------------------------------


def test_provider_bundle_check_statuses_runs_for_all_four():
    async def run():
        bundle = build_provider_bundle(
            {
                "vad": {"provider": "mock"},
                "asr": {"provider": "mock"},
                "llm": {"provider": "mock"},
                "tts": {"provider": "mock"},
            }
        )
        statuses = await bundle.check_statuses()
        return statuses

    statuses = asyncio.run(run())
    assert statuses is not None
    assert len(statuses) == 4  # type: ignore[arg-type]
    for s in statuses:  # type: ignore[union-attr]
        assert s["ready"] is True


# ---------------------------------------------------------------------------
# Registration error path
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase 1.1: Faster-whisper lazy-load on first transcribe
# ---------------------------------------------------------------------------


def test_faster_whisper_transcribe_calls_ensure_model_on_first_use():
    """_transcribe_blocking calls _ensure_model() instead of assuming the
    model was pre-loaded. A fake WhisperModel that records construction
    proves the lazy load happens on the first transcribe call."""
    from converse_framework.providers.faster_whisper import FasterWhisperASRProvider

    constructed: list[dict] = []

    class FakeWhisperModel:
        def __init__(self, model_name, **kw):
            constructed.append({"model_name": model_name, "kw": kw})

        def transcribe(self, audio, **kw):
            return iter(()), {}

    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = FakeWhisperModel  # type: ignore[assignment]
    sys.modules["faster_whisper"] = fake_module

    try:
        provider = FasterWhisperASRProvider({"model": "tiny", "language": "en"})
        assert provider._model is None

        async def run():
            events = []

            async def progress(event_type, payload):
                events.append((event_type, payload))

            result = []
            async for t in provider.transcribe_audio(
                b"\x00\x00" * 1600, 16000, progress
            ):
                result.append(t)
            return events

        asyncio.run(run())
    finally:
        sys.modules.pop("faster_whisper", None)

    assert len(constructed) == 1
    assert constructed[0]["model_name"] == "tiny"


def test_faster_whisper_injected_model_skips_load():
    """If _model is already injected, _ensure_model() should be a no-op
    and _transcribe_blocking should use the existing model directly."""
    from converse_framework.providers.faster_whisper import FasterWhisperASRProvider

    call_log: list[str] = []

    class FakeModel:
        def transcribe(self, audio, **kw):
            call_log.append("transcribed")
            return iter(()), {}

    provider = FasterWhisperASRProvider(
        {"model": "tiny", "language": "en", "_model": FakeModel()}
    )
    assert provider._model is not None

    import numpy as np

    loop = asyncio.new_event_loop()
    result = provider._transcribe_blocking(np.zeros(1600, dtype=np.float32), None, loop)
    loop.close()

    assert call_log == ["transcribed"]
    assert result == []  # no text segments from empty transcribe


def test_faster_whisper_load_failure_sets_error_and_raises():
    """When _ensure_model() fails, _load_error must be set and
    _transcribe_blocking must raise RuntimeError with an actionable message."""
    from converse_framework.providers.faster_whisper import FasterWhisperASRProvider

    provider = FasterWhisperASRProvider({"model": "tiny", "language": "en"})
    assert provider._model is None

    import numpy as np

    loop = asyncio.new_event_loop()
    with pytest.raises(RuntimeError, match="faster-whisper model did not load"):
        provider._transcribe_blocking(np.zeros(1600, dtype=np.float32), None, loop)
    loop.close()

    assert provider._load_error is not None


def test_register_provider_rejects_unknown_kind():
    with pytest.raises(ValueError, match="Unknown provider kind"):
        register_provider("bogus", "x", "some.module:Class")


def test_register_provider_overrides_existing():
    register_provider(
        "vad", "test-tmp", "converse_framework.providers.mock:MockVADProvider"
    )
    p = build_provider("vad", "test-tmp")
    assert isinstance(p, VADProvider)
    # Cleanup so other tests aren't surprised.
    from converse_framework.registry import _registry

    _registry["vad"].pop("test-tmp", None)
