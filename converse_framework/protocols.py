"""Provider interfaces and shared dataclasses for the speech stack."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ProviderCapabilities:
    """Static capability flags advertised by a provider implementation.

    The pipeline and transport layers consult these flags to decide
    which features (partial transcripts, streaming TTS, barge-in, GPU
    requirements, supported languages) are available without
    instantiating the provider.

    Attributes:
        supports_partials: Provider can emit non-final transcript
            chunks while audio is still arriving.
        supports_streaming_tts: TTS can start streaming audio chunks
            before the full text is known.
        supports_barge_in: Provider can detect user speech while TTS
            is still playing and signal cancellation.
        requires_gpu: Provider needs a GPU at runtime. UI layers use
            this to warn the user before they select the provider.
        languages: ISO language codes the provider can handle.
    """

    supports_partials: bool = False
    supports_streaming_tts: bool = False
    supports_barge_in: bool = False
    requires_gpu: bool = False
    languages: tuple[str, ...] = ("en",)


@dataclass(frozen=True)
class ProviderStatus:
    """Runtime status snapshot of a provider.

    Returned by the ``status`` property / ``check_status`` coroutine of
    every provider protocol. ``ready`` is the headline boolean the UI
    uses to enable / disable a provider row. ``message`` carries the
    human-readable explanation (missing dependency, model not loaded,
    GPU absent, ...).

    Attributes:
        name: Provider name as registered in :mod:`registry` (e.g.
            ``"mock"``, ``"silero"``, ``"faster-whisper"``).
        kind: Provider category, one of ``"vad"``, ``"asr"``,
            ``"llm"``, ``"tts"``.
        ready: True if the provider can be used right now.
        message: Human-readable status, surfaced verbatim in the UI.
        install_hint: Optional package spec to install when this
            provider is unavailable because an optional dependency is
            missing, e.g. ``"converse-framework[silero]"``.
        missing_extra: Optional extra name for UI display when the
            framework knows which optional dependency group is missing.
        capabilities: Static feature flags for this provider.
        provider_id: Stable identifier for UI selection when the
            registered ``name`` is aliased.
        selected: True if this provider is the one currently bound
            into the active :class:`ProviderBundle`.
        loaded: True if the heavy backend has been initialised.
        managed_externally: Provider lifecycle is owned by another
            runtime (e.g. a TTS preset manager) and the framework
            should not call :meth:`load` / :meth:`unload` on it.
        supports_model_management: Provider exposes model hot-swap.
        supports_voice_selection: Provider exposes voice selection.
    """

    name: str
    kind: str
    ready: bool
    message: str
    capabilities: ProviderCapabilities
    install_hint: str | None = None
    missing_extra: str | None = None
    provider_id: str | None = None
    selected: bool = False
    loaded: bool = True
    managed_externally: bool = False
    supports_model_management: bool = False
    supports_voice_selection: bool = False


@dataclass(frozen=True)
class TranscriptEvent:
    """A single incremental transcript chunk produced by an ASR provider.

    ASR providers stream a sequence of these events for every audio
    turn. Non-final events (``final=False``) are hypothesis updates
    that may still change; only the last ``final=True`` event in a
    stream is authoritative for the utterance.

    Attributes:
        text: Transcript text for this chunk. For non-final chunks
            this is the running hypothesis; for the final chunk it is
            the committed utterance text.
        final: True iff this is the closing, committed transcript for
            the current utterance.
    """

    text: str
    final: bool


@dataclass(frozen=True)
class AudioChunk:
    """A single chunk of encoded audio emitted by a TTS provider.

    TTS providers yield a stream of these chunks. The framework does
    not interpret the audio bytes directly -- it forwards them to
    transports -- but it does attach enough metadata for downstream
    consumers to render or persist the audio correctly.

    Attributes:
        data: Raw encoded audio bytes (the encoding is described by
            ``encoding`` / ``mime_type``).
        mime_type: Optional MIME hint (``"audio/wav"``,
            ``"audio/mpeg"``, ...). ``None`` if the provider cannot
            name the encoding.
        sample_rate: Samples per second of the decoded audio, or
            ``None`` if not applicable (e.g. compressed formats
            served whole).
        channels: Channel count of the decoded audio.
        encoding: Encoding name (``"pcm_s16le"``, ``"mp3"``,
            ``"wav"`` ...). Matches the value the
            :mod:`audio_utils` helpers expect.
        duration_ms: Duration of this chunk in milliseconds, when the
            provider can compute it. ``None`` for the first chunk of
            streaming codecs.
        final: True if this is the last chunk for the current
            synthesis request.
    """

    data: bytes
    mime_type: str | None = None
    sample_rate: int | None = None
    channels: int = 1
    encoding: str | None = None
    duration_ms: int | None = None
    final: bool = False


@dataclass(frozen=True)
class VADEvent:
    """A single VAD decision produced by a VAD provider.

    The :class:`AudioUtteranceCollector` state machine consumes a
    stream of these events per :class:`AudioFrame` to drive the
    recording lifecycle.

    Attributes:
        type: Event kind. ``"vad.speech_start"`` marks the leading
            edge of detected speech; ``"vad.speech_end"`` marks the
            trailing edge; ``"vad.probability"`` is an intermediate
            level readout that does not change the recording state.
        probability: Confidence of the decision, in ``[0.0, 1.0]``.
        audio_ms: Position in the current utterance, in milliseconds
            from the first frame of the turn.
    """

    type: str
    probability: float
    audio_ms: int


ProgressCallback = Callable[[str, dict], Awaitable[None]]


@runtime_checkable
class VADProvider(Protocol):
    """Voice-activity-detection provider.

    Implementations consume a stream of parsed :class:`AudioFrame`
    objects and emit :class:`VADEvent` decisions that the utterance
    collector turns into utterance boundaries.

    The ``status`` property exposes the current
    :class:`ProviderStatus`; :meth:`check_status` is the async form
    that performs a real probe (file existence, model loaded, ...).
    """

    @property
    def status(self) -> ProviderStatus: ...

    async def check_status(self) -> ProviderStatus: ...

    async def process_frame(self, frame) -> list[VADEvent]: ...

    async def unload(self) -> ProviderStatus: ...


@runtime_checkable
class ASRProvider(Protocol):
    """Automatic-speech-recognition provider.

    Implementations accept either raw 16-bit signed-LE mono PCM
    bytes (audio path) or a transcript seed string (text path) and
    stream :class:`TranscriptEvent` chunks. The text-input path is
    used by the pipeline to keep the public API symmetric between
    audio and chat front-ends.
    """

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
    """Large-language-model provider.

    Implementations take an OpenAI-style ``messages`` list
    (``[{"role": ..., "content": ...}, ...]``) and stream token
    strings. The pipeline feeds these tokens into the TTS chunker,
    so implementations do not need to do their own sentence
    splitting -- a simple token stream is the contract.
    """

    @property
    def status(self) -> ProviderStatus: ...

    async def check_status(self) -> ProviderStatus: ...

    async def stream_response(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]: ...


@runtime_checkable
class TTSProvider(Protocol):
    """Text-to-speech provider.

    Implementations accept a single text string and stream
    :class:`AudioChunk` objects back. The :meth:`stream_audio` form
    is the simple contract; :meth:`stream_audio_with_progress` adds
    an optional progress callback the pipeline uses to emit
    ``tts.progress`` events to the transport layer.
    """

    @property
    def status(self) -> ProviderStatus: ...

    async def check_status(self) -> ProviderStatus: ...

    async def load(self) -> ProviderStatus: ...

    async def unload(self) -> ProviderStatus: ...

    async def stream_audio(self, text: str) -> AsyncIterator[AudioChunk]: ...

    async def stream_audio_with_progress(
        self, text: str, progress: ProgressCallback | None = None
    ) -> AsyncIterator[AudioChunk]: ...
