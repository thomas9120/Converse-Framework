"""Unavailable provider sentinel returned when a real provider cannot load."""

from __future__ import annotations

from collections.abc import AsyncIterator

from converse_framework.protocols import (
    ASRProvider,
    AudioChunk,
    LLMProvider,
    ProviderCapabilities,
    ProviderStatus,
    TTSProvider,
    TranscriptEvent,
    VADProvider,
)  # fmt: skip


# Map of provider name -> optional-dependency extra. Used to construct a
# helpful install hint when a provider's underlying dependency is missing.
EXTRAS: dict[tuple[str, str], str] = {
    ("vad", "silero"): "silero",
    ("asr", "faster-whisper"): "faster-whisper",
    ("asr", "whisper-cpp"): "whisper-cpp",
    ("llm", "llamacpp"): "llamacpp",
    ("tts", "kokoro"): "kokoro",
    ("tts", "kokoro-onnx"): "kokoro",
    ("tts", "pocket-tts"): "pocket-tts",
}

# Backward-compatible table exported for callers that already use it.
EXTRA_HINTS: dict[tuple[str, str], str] = {
    key: f"converse-framework[{extra}]" for key, extra in EXTRAS.items()
}


def extra_hint_for(kind: str, name: str) -> str | None:
    """Return the ``pip install`` extra hint for a missing provider, if known.

    Apps and the :class:`UnavailableProvider` sentinel use this to
    build a friendly installation message when a registered
    provider's heavy backend is not installed in the current
    environment.

    Args:
        kind: Provider category (``"vad"``, ``"asr"``, ``"llm"``,
            ``"tts"``).
        name: Registered provider name (e.g. ``"silero"``,
            ``"faster-whisper"``).

    Returns:
        The matching ``pip install`` extra string (e.g.
        ``"converse-framework[silero]"``) when one is registered,
        otherwise ``None``.
    """
    return EXTRA_HINTS.get((kind, name))


def missing_extra_for(kind: str, name: str) -> str | None:
    """Return the optional-dependency extra for a missing provider, if known."""
    return EXTRAS.get((kind, name))


class UnavailableProvider(VADProvider, ASRProvider, LLMProvider, TTSProvider):
    """Sentinel provider that reports not-ready and raises on use."""

    def __init__(
        self,
        kind: str,
        name: str,
        message: str | None = None,
        requires_gpu: bool = False,
    ):
        if message is None:
            extra = extra_hint_for(kind, name)
            message = (
                f"Provider '{name}' ({kind}) is not available. "
                f"Install the required extra with "
                f"`pip install {extra}`."
                if extra
                else f"Provider '{name}' ({kind}) is not available."
            )
        install_hint = extra_hint_for(kind, name)
        self._status = ProviderStatus(
            name=name,
            kind=kind,
            ready=False,
            message=message,
            capabilities=ProviderCapabilities(requires_gpu=requires_gpu),
            install_hint=install_hint,
            missing_extra=missing_extra_for(kind, name),
            provider_id=name,
            loaded=False,
            status_level="unavailable",
        )

    @property
    def status(self) -> ProviderStatus:
        return self._status

    async def check_status(self) -> ProviderStatus:
        return self.status

    async def process_frame(self, frame) -> list:
        return []

    async def transcribe_text_input(self, text: str) -> AsyncIterator[TranscriptEvent]:
        raise RuntimeError(self._status.message)
        yield  # pragma: no cover

    async def transcribe_audio(
        self, pcm_s16le: bytes, sample_rate: int, progress=None
    ) -> AsyncIterator[TranscriptEvent]:
        raise RuntimeError(self._status.message)
        yield  # pragma: no cover

    async def stream_response(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        raise RuntimeError(self._status.message)
        yield  # pragma: no cover

    async def stream_audio(self, text: str) -> AsyncIterator[AudioChunk]:
        raise RuntimeError(self._status.message)
        yield  # pragma: no cover

    async def stream_audio_with_progress(
        self, text: str, progress=None
    ) -> AsyncIterator[AudioChunk]:
        raise RuntimeError(self._status.message)
        yield  # pragma: no cover

    async def load(self) -> ProviderStatus:
        return self.status

    async def unload(self) -> ProviderStatus:
        return self.status
