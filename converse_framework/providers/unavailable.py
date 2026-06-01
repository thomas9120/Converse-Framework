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


class UnavailableProvider(VADProvider, ASRProvider, LLMProvider, TTSProvider):
    """Sentinel provider that reports not-ready and raises on use."""

    def __init__(self, kind: str, name: str, message: str, requires_gpu: bool = False):
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
