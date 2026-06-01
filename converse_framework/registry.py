"""Lazy provider registry for the speech stack.

Provider implementations are registered by import string and only loaded
on first use so that ``import converse_framework`` stays lightweight.
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
    """
    if kind not in _registry:
        raise ValueError(
            f"Unknown provider kind: {kind}. Must be one of vad/asr/llm/tts."
        )
    _registry[kind][name] = _ProviderEntry(import_path, availability_probe)


def is_provider_available(kind: str, name: str) -> bool:
    """Return True if the named provider can be loaded."""
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


def build_provider(
    kind: str, name: str, config: dict[str, Any] | None = None
) -> VADProvider | ASRProvider | LLMProvider | TTSProvider:
    """Instantiate a provider by kind and name.

    Uses :func:`is_provider_available` to decide whether to attempt
    loading or return an unavailable sentinel.
    """
    if config is None:
        config = {}

    if not is_provider_available(kind, name):
        # Return an unavailable provider for this kind/name
        from converse_framework.providers.unavailable import UnavailableProvider

        return UnavailableProvider(
            kind=kind,
            name=name,
            message=(
                f"Provider '{name}' ({kind}) is not available. "
                f"Ensure the required extra is installed "
                f"(e.g. pip install converse-framework[{name}])."
            ),
        )

    entry = _registry[kind][name]
    module_path, class_name = entry.import_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(config)


@dataclass
class ProviderBundle:
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


def _serialize_status(item: ProviderStatus) -> dict[str, Any]:
    return {
        "name": item.name,
        "kind": item.kind,
        "ready": item.ready,
        "message": item.message,
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
# Bootstrap: register the built-in mock and unavailable providers eagerly
# so they are always available without extras.
# ---------------------------------------------------------------------------

register_provider("vad", "mock", "converse_framework.providers.mock:MockVADProvider")
register_provider("asr", "mock", "converse_framework.providers.mock:MockASRProvider")
register_provider("llm", "mock", "converse_framework.providers.mock:MockLLMProvider")
register_provider("tts", "mock", "converse_framework.providers.mock:MockTTSProvider")

# Concrete providers registered lazily (modules not yet present in Phase 1)
register_provider(
    "vad", "silero", "converse_framework.providers.silero:SileroVADProvider"
)
register_provider(
    "asr",
    "faster-whisper",
    "converse_framework.providers.faster_whisper:FasterWhisperASRProvider",
)
register_provider(
    "llm", "llamacpp", "converse_framework.providers.llamacpp:LlamaCppProvider"
)
register_provider(
    "tts", "kokoro", "converse_framework.providers.kokoro_onnx:KokoroOnnxProvider"
)
register_provider(
    "tts", "pocket-tts", "converse_framework.providers.pocket_tts:PocketTTSProvider"
)
