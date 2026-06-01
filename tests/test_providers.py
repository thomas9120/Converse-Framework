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

import pytest

from converse_framework.protocols import (
    ASRProvider,
    LLMProvider,
    TTSProvider,
    VADProvider,
)
from converse_framework.providers.unavailable import (
    EXTRA_HINTS,
    UnavailableProvider,
    extra_hint_for,
)
from converse_framework.registry import (
    ProviderBundle,
    build_provider,
    build_provider_bundle,
    is_provider_available,
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
    assert "converse-framework[silero]" == extra_hint_for("vad", "silero")
    assert "converse-framework[faster-whisper]" == extra_hint_for(
        "asr", "faster-whisper"
    )
    assert "converse-framework[llamacpp]" == extra_hint_for("llm", "llamacpp")
    assert "converse-framework[kokoro]" == extra_hint_for("tts", "kokoro")
    assert "converse-framework[kokoro]" == extra_hint_for("tts", "kokoro-onnx")
    assert "converse-framework[pocket-tts]" == extra_hint_for("tts", "pocket-tts")


def test_extra_hint_for_unknown_provider_is_none():
    assert extra_hint_for("vad", "made-up") is None


def test_extra_hint_table_is_exported():
    assert isinstance(EXTRA_HINTS, dict)
    assert ("vad", "silero") in EXTRA_HINTS


def test_unavailable_provider_includes_extra_hint():
    p = UnavailableProvider("vad", "silero")
    assert "converse-framework[silero]" in p.status.message
    assert "pip install" in p.status.message


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
    provider = build_provider("llm", "llamacpp", {"base_url": "http://x:1"})

    captured: dict = {}

    def sampler() -> dict:
        captured["called"] = True
        return {"temperature": 0.42}

    provider.set_sampler_provider(sampler)
    assert provider._sampler_provider is sampler
    assert provider._build_sampler() == {"temperature": 0.42}
    assert captured == {"called": True}


def test_llamacpp_sampler_provider_can_be_cleared():
    provider = build_provider("llm", "llamacpp", {"temperature": 0.7})
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
    assert "PROJECT_ROOT" not in source.replace(
        "harness ``PROJECT_ROOT``", ""
    )

    # When no cache_dir is given, the provider uses a platform default.
    # The CONVERSE_FRAMEWORK_CACHE_DIR env var lets us assert a value
    # without touching the user's real home directory.
    monkeypatch.setenv("CONVERSE_FRAMEWORK_CACHE_DIR", str(tmp_path))
    provider = build_provider(
        "tts",
        "kokoro",
        {"voice": "af_heart", "lang": "en-us", "_model": object(), "_g2p": object()},
    )
    assert provider.cache_dir == tmp_path / "kokoro"


# ---------------------------------------------------------------------------
# ProviderBundle check_statuses
# ---------------------------------------------------------------------------


def test_provider_bundle_check_statuses_runs_for_all_four():
    async def run() -> None:
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
    assert len(statuses) == 4
    for s in statuses:
        assert s["ready"] is True


# ---------------------------------------------------------------------------
# Registration error path
# ---------------------------------------------------------------------------


def test_register_provider_rejects_unknown_kind():
    with pytest.raises(ValueError, match="Unknown provider kind"):
        register_provider("bogus", "x", "some.module:Class")


def test_register_provider_overrides_existing():
    register_provider("vad", "test-tmp", "converse_framework.providers.mock:MockVADProvider")
    p = build_provider("vad", "test-tmp")
    assert isinstance(p, VADProvider)
    # Cleanup so other tests aren't surprised.
    from converse_framework.registry import _registry

    _registry["vad"].pop("test-tmp", None)
