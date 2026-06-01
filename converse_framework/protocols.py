"""Provider interfaces and shared dataclasses for the speech stack."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ProviderCapabilities:
    supports_partials: bool = False
    supports_streaming_tts: bool = False
    supports_barge_in: bool = False
    requires_gpu: bool = False
    languages: tuple[str, ...] = ("en",)


@dataclass(frozen=True)
class ProviderStatus:
    name: str
    kind: str
    ready: bool
    message: str
    capabilities: ProviderCapabilities
    provider_id: str | None = None
    selected: bool = False
    loaded: bool = True
    managed_externally: bool = False
    supports_model_management: bool = False
    supports_voice_selection: bool = False


@dataclass(frozen=True)
class TranscriptEvent:
    text: str
    final: bool


@dataclass(frozen=True)
class AudioChunk:
    data: bytes
    mime_type: str | None = None
    sample_rate: int | None = None
    channels: int = 1
    encoding: str | None = None
    duration_ms: int | None = None
    final: bool = False


@dataclass(frozen=True)
class VADEvent:
    type: str
    probability: float
    audio_ms: int


ProgressCallback = Callable[[str, dict], Awaitable[None]]


@runtime_checkable
class VADProvider(Protocol):
    @property
    def status(self) -> ProviderStatus: ...

    async def check_status(self) -> ProviderStatus: ...

    async def process_frame(self, frame) -> list[VADEvent]: ...

    async def unload(self) -> ProviderStatus: ...


@runtime_checkable
class ASRProvider(Protocol):
    @property
    def status(self) -> ProviderStatus: ...

    async def check_status(self) -> ProviderStatus: ...

    async def load(self) -> ProviderStatus: ...

    async def transcribe_text_input(
        self, text: str
    ) -> AsyncIterator[TranscriptEvent]: ...

    async def transcribe_audio(
        self,
        pcm_s16le: bytes,
        sample_rate: int,
        progress: ProgressCallback | None = None,
    ) -> AsyncIterator[TranscriptEvent]: ...

    async def unload(self) -> ProviderStatus: ...


@runtime_checkable
class LLMProvider(Protocol):
    @property
    def status(self) -> ProviderStatus: ...

    async def check_status(self) -> ProviderStatus: ...

    async def stream_response(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]: ...


@runtime_checkable
class TTSProvider(Protocol):
    @property
    def status(self) -> ProviderStatus: ...

    async def check_status(self) -> ProviderStatus: ...

    async def load(self) -> ProviderStatus: ...

    async def unload(self) -> ProviderStatus: ...

    async def stream_audio(self, text: str) -> AsyncIterator[AudioChunk]: ...

    async def stream_audio_with_progress(
        self, text: str, progress: ProgressCallback | None = None
    ) -> AsyncIterator[AudioChunk]: ...
