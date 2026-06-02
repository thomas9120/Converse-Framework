"""Lazy provider registry for the speech stack.

Provider implementations are registered by import string and only loaded
on first use so that ``import converse_framework`` stays lightweight.

Each provider kind (``vad``/``asr``/``llm``/``tts``) supports a small set of
known names. The base install always provides the ``mock`` providers. Other
providers are registered with availability probes so the registry can
report friendlier error messages when an optional dependency is missing.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from converse_framework.protocols import (
    ASRProvider,
    LLMProvider,
    ProviderStatus,
    TTSProvider,
    VADProvider,
)


@dataclass
class _ProviderEntry:
    import_path: str
    availability_probe: Callable[[], bool] | None = None


_registry: dict[str, dict[str, _ProviderEntry]] = {
    "vad": {},
    "asr": {},
    "llm": {},
    "tts": {},
}


def register_provider(
    kind: str,
    name: str,
    import_path: str,
    *,
    availability_probe: Callable[[], bool] | None = None,
) -> None:
    """Register a provider implementation by import string.

    The module is not imported until :func:`build_provider` is called.
    A custom ``availability_probe`` (returning True if the provider is
    ready to use) lets the registry give specific feedback about which
    optional dependency is missing.
    """
    if kind not in _registry:
        raise ValueError(
            f"Unknown provider kind: {kind}. Must be one of vad/asr/llm/tts."
        )
    _registry[kind][name] = _ProviderEntry(import_path, availability_probe)


def is_provider_available(kind: str, name: str) -> bool:
    """Return True if the named provider can be loaded.

    A registered provider is considered available when either:
      * its ``availability_probe`` returns True, or
      * its module imports without raising :class:`ImportError`.
    """
    entry = _registry.get(kind, {}).get(name)
    if entry is None:
        return False
    if entry.availability_probe is not None:
        return entry.availability_probe()
    module_path, _ = entry.import_path.rsplit(":", 1)
    try:
        importlib.import_module(module_path)
        return True
    except ImportError:
        return False


def _probe_module(module_path: str) -> Callable[[], bool]:
    def probe() -> bool:
        try:
            importlib.import_module(module_path)
            return True
        except ImportError:
            return False

    return probe


def build_provider(
    kind: str, name: str, config: dict[str, Any] | None = None
) -> VADProvider | ASRProvider | LLMProvider | TTSProvider:
    """Instantiate a provider by kind and name.

    The framework's own provider modules import cleanly even when their
    heavy third-party dependencies are missing (those imports are deferred
    into provider methods). So the build path only requires the
    ``converse_framework.providers.<x>`` module to be importable, not the
    heavy backend library. :func:`is_provider_available` adds the stricter
    "is the heavy dep present" check for callers that want a definitive
    availability signal; missing deps are also reported by the resulting
    provider's status message.
    """
    if config is None:
        config = {}

    entry = _registry.get(kind, {}).get(name)
    if entry is None:
        from converse_framework.providers.unavailable import UnavailableProvider

        return UnavailableProvider(kind=kind, name=name)

    module_path, class_name = entry.import_path.rsplit(":", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        from converse_framework.providers.unavailable import UnavailableProvider

        return UnavailableProvider(kind=kind, name=name)
    cls = getattr(module, class_name)
    return cls(config)


@dataclass
class ProviderBundle:
    """The four providers a :class:`SpeechPipeline` needs to run a turn.

    Bundles are usually produced by :func:`build_provider_bundle`
    from a config mapping, but apps can construct them by hand
    when they want to inject a provider that lives outside the
    registry (e.g. a harness-managed TTS runtime). The framework
    treats the four attributes as the canonical handles; the
    pipeline never falls back to the registry at runtime, so a
    bundle fully describes the providers a turn will use.

    Attributes:
        vad: VAD provider used by the utterance collector.
        asr: ASR provider used for both audio and text input.
        llm: LLM provider used to generate the assistant reply.
        tts: TTS provider used to synthesise the assistant reply.
    """

    vad: VADProvider
    asr: ASRProvider
    llm: LLMProvider
    tts: TTSProvider

    def statuses(self) -> list[dict[str, Any]]:
        items = [self.vad.status, self.asr.status, self.llm.status, self.tts.status]
        return _serialize_statuses(items)

    async def check_statuses(self) -> list[dict[str, Any]]:
        items = [
            await self.vad.check_status(),
            await self.asr.check_status(),
            await self.llm.check_status(),
            await self.tts.check_status(),
        ]
        return _serialize_statuses(items)


def build_provider_bundle(
    config: Mapping[str, Mapping[str, Any]],
    *,
    tts_provider: TTSProvider | None = None,
) -> ProviderBundle:
    """Build a complete provider bundle from a nested config mapping.

    Expected config shape::

        {
            "vad": {"provider": "mock", ...},
            "asr": {"provider": "mock", ...},
            "llm": {"provider": "mock", ...},
            "tts": {"provider": "mock", ...},
        }

    If *tts_provider* is given it replaces the TTS provider built from
    config, which allows the caller to inject a harness-managed TTS
    runtime.
    """
    vad_cfg = dict(config.get("vad", {}))
    audio_cfg = dict(config.get("audio", {}))
    vad_cfg.setdefault("sample_rate", int(audio_cfg.get("sample_rate", 16000)))

    return ProviderBundle(
        vad=build_provider("vad", vad_cfg.get("provider", "mock"), vad_cfg),
        asr=build_provider(
            "asr",
            config.get("asr", {}).get("provider", "mock"),
            dict(config.get("asr", {})),
        ),
        llm=build_provider(
            "llm",
            config.get("llm", {}).get("provider", "mock"),
            dict(config.get("llm", {})),
        ),
        tts=tts_provider
        or build_provider(
            "tts",
            config.get("tts", {}).get("provider", "mock"),
            dict(config.get("tts", {})),
        ),
    )


async def status_only(
    config: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Report the runtime status of each provider without loading models.

    Cheaper than :func:`build_provider_bundle` for diagnostics: the
    returned list mirrors :meth:`ProviderBundle.statuses` in shape and
    ordering (``vad``, ``asr``, ``llm``, ``tts``), but no provider's
    ``load()`` is called and no full bundle is constructed. A provider
    whose optional dependency is missing shows up via
    :class:`UnavailableProvider` with the install hint in its
    ``message``.

    Args:
        config: Nested provider configuration in the same shape
            :func:`build_provider_bundle` accepts. Missing kinds fall
            back to the ``"mock"`` provider.

    Returns:
        A list of four serialized status dicts (one per kind) in
        ``[vad, asr, llm, tts]`` order. Each dict matches the shape
        produced by :func:`_serialize_status` so callers can use the
        result interchangeably with :meth:`ProviderBundle.statuses`.
    """
    statuses: list[ProviderStatus] = []
    for kind in ("vad", "asr", "llm", "tts"):
        kind_config = dict(config.get(kind, {}))
        name = str(kind_config.get("provider", "mock"))
        provider = build_provider(kind, name, kind_config)
        statuses.append(await provider.check_status())
    return _serialize_statuses(statuses)


def _serialize_status(item: ProviderStatus) -> dict[str, Any]:
    return {
        "name": item.name,
        "kind": item.kind,
        "ready": item.ready,
        "message": item.message,
        "install_hint": item.install_hint,
        "missing_extra": item.missing_extra,
        "capabilities": item.capabilities.__dict__,
        "provider_id": item.provider_id,
        "selected": item.selected,
        "loaded": item.loaded,
        "managed_externally": item.managed_externally,
        "supports_model_management": item.supports_model_management,
        "supports_voice_selection": item.supports_voice_selection,
    }


def _serialize_statuses(items) -> list[dict[str, Any]]:
    return [_serialize_status(item) for item in items]


# ---------------------------------------------------------------------------
# Built-in provider registrations
# ---------------------------------------------------------------------------
#
# Mock and unavailable providers are always available. Concrete providers
# (silero, faster-whisper, llamacpp, kokoro-onnx, pocket-tts) live in their
# own modules so their heavy third-party imports only happen when the user
# actually selects them. The probes below check module importability without
# forcing eager imports.

register_provider("vad", "mock", "converse_framework.providers.mock:MockVADProvider")
register_provider("asr", "mock", "converse_framework.providers.mock:MockASRProvider")
register_provider("llm", "mock", "converse_framework.providers.mock:MockLLMProvider")
register_provider("tts", "mock", "converse_framework.providers.mock:MockTTSProvider")

register_provider(
    "vad",
    "silero",
    "converse_framework.providers.silero:SileroVADProvider",
    availability_probe=_probe_module("silero_vad"),
)
register_provider(
    "asr",
    "faster-whisper",
    "converse_framework.providers.faster_whisper:FasterWhisperASRProvider",
    availability_probe=_probe_module("faster_whisper"),
)
register_provider(
    "asr",
    "whisper-cpp",
    "converse_framework.providers.whisper_cpp:WhisperCppASRProvider",
    availability_probe=_probe_module("httpx"),
)
register_provider(
    "llm",
    "llamacpp",
    "converse_framework.providers.llamacpp:LlamaCppProvider",
    availability_probe=_probe_module("httpx"),
)
# ``kokoro`` is the name used in profiles; the implementation lives in
# ``kokoro_onnx.py`` because that's the model family. ``kokoro-onnx`` is
# kept as a legacy alias for harness compatibility.
register_provider(
    "tts",
    "kokoro",
    "converse_framework.providers.kokoro_onnx:KokoroOnnxProvider",
    availability_probe=_probe_module("kokoro_onnx"),
)
register_provider(
    "tts",
    "kokoro-onnx",
    "converse_framework.providers.kokoro_onnx:KokoroOnnxProvider",
    availability_probe=_probe_module("kokoro_onnx"),
)
register_provider(
    "tts",
    "pocket-tts",
    "converse_framework.providers.pocket_tts:PocketTTSProvider",
    availability_probe=_probe_module("pocket_tts"),
)
