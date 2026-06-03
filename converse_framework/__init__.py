"""Converse Framework -- provider-agnostic speech stack."""

from converse_framework.audio_utils import (
    AudioFrame,
    AudioFrameStats,
    compute_pcm16_level,
    float_audio_to_pcm_s16le_bytes,
    float_audio_to_wav_bytes,
    make_tone_wav,
    parse_audio_frame,
    pcm_s16le_to_float32,
    trim_pcm16_silence,
)
from converse_framework.events import (
    EventSink,
    FrameworkEvent,
    QueueEventSink,
    TransportEventSink,
)
from converse_framework.pipeline import (
    PipelineConfig,
    SpeechPipeline,
)
from converse_framework.provider_events import (
    provider_error_event,
    provider_loaded_event,
    provider_loading_event,
)
from converse_framework.protocols import (
    ASRProvider,
    AudioChunk,
    LLMProvider,
    ProviderCapabilities,
    ProviderConfigResult,
    ProviderStatus,
    TTSProvider,
    TranscriptEvent,
    VADEvent,
    VADProvider,
    VoiceInfo,
)
from converse_framework.providers.unavailable import extra_hint_for, missing_extra_for
from converse_framework.registry import (
    ProviderBundle,
    build_provider,
    build_provider_bundle,
    is_provider_available,
    register_provider,
    status_only,
)
from converse_framework.transport import (
    QueueTransport,
    Transport,
)
from converse_framework.utterance_collector import (
    AudioUtteranceCollector,
    UtteranceCollectorConfig,
)

# Compatibility alias for harness consumers
HarnessEvent = FrameworkEvent

__all__ = [
    "ASRProvider",
    "AudioChunk",
    "AudioFrame",
    "AudioFrameStats",
    "AudioUtteranceCollector",
    "EventSink",
    "FrameworkEvent",
    "HarnessEvent",
    "LLMProvider",
    "ProviderBundle",
    "ProviderCapabilities",
    "ProviderConfigResult",
    "ProviderStatus",
    "PipelineConfig",
    "provider_error_event",
    "provider_loaded_event",
    "provider_loading_event",
    "QueueEventSink",
    "QueueTransport",
    "SpeechPipeline",
    "TTSProvider",
    "TranscriptEvent",
    "Transport",
    "TransportEventSink",
    "UtteranceCollectorConfig",
    "VADEvent",
    "VoiceInfo",
    "VADProvider",
    "build_provider",
    "build_provider_bundle",
    "compute_pcm16_level",
    "extra_hint_for",
    "float_audio_to_pcm_s16le_bytes",
    "float_audio_to_wav_bytes",
    "is_provider_available",
    "make_tone_wav",
    "missing_extra_for",
    "parse_audio_frame",
    "pcm_s16le_to_float32",
    "register_provider",
    "status_only",
    "trim_pcm16_silence",
]

__version__ = "0.1.0"
