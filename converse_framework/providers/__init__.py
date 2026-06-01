"""Built-in provider implementations.

Mock and unavailable providers are imported eagerly because they have no
heavy dependencies. The concrete providers (``silero``, ``faster-whisper``,
``llamacpp``, ``kokoro-onnx``, ``pocket-tts``) are not imported here --
they are registered with :func:`converse_framework.registry.register_provider`
by import string and loaded lazily on first use.
"""

from converse_framework.providers.mock import (
    MockASRProvider,
    MockLLMProvider,
    MockTTSProvider,
    MockVADProvider,
)
from converse_framework.providers.unavailable import (
    UnavailableProvider,
    extra_hint_for,
)

__all__ = [
    "MockASRProvider",
    "MockLLMProvider",
    "MockTTSProvider",
    "MockVADProvider",
    "UnavailableProvider",
    "extra_hint_for",
]
