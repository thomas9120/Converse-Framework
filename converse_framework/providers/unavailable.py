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
)


# Map of provider name -> ``pip install`` extra hint. Used to construct a
# helpful message when a provider's underlying dependency is missing.
EXTRA_HINTS: dict[tuple[str, str], str] = {
    ("vad", "silero"): "converse-framework[silero]",
    ("asr", "faster-whisper"): "converse-framework[faster-whisper]",
    ("llm", "llamacpp"): "converse-framework[llamacpp]",
    ("tts", "kokoro"): "converse-framework[kokoro]",
    ("tts", "kokoro-onnx"): "converse-framework[kokoro]",
    ("tts", "pocket-tts"): "converse-framework[pocket-tts]",
}


def extra_hint_for(kind: str, name: str) -> str | None:
    """Return the ``pip install`` extra hint for a missing provider, if known."""
    return EXTRA_HINTS.get((kind, name))


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
        self._status = ProviderStatus(
            name=name,
            kind=kind,
            ready=False,
            message=message,
            capabilities=ProviderCapabilities(requires_gpu=requires_gpu),
            provider_id=name,
            loaded=False,
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
