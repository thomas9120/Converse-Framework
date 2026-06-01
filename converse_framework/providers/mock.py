"""Mock provider implementations with no external dependencies.

These are always available and useful for testing pipeline orchestration
without real model backends.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from converse_framework.audio_utils import make_tone_wav
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


class MockVADProvider(VADProvider):
    def __init__(self, config: dict):
        self.config = config

    @property
    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name="mock-vad",
            kind="vad",
            ready=True,
            message="Mock VAD accepts browser speech-start/speech-end events.",
            capabilities=ProviderCapabilities(supports_barge_in=True),
            provider_id="mock",
        )

    async def check_status(self) -> ProviderStatus:
        return self.status

    async def process_frame(self, frame) -> list:
        return []

    async def unload(self) -> ProviderStatus:
        return self.status


class MockASRProvider(ASRProvider):
    def __init__(self, config: dict):
        self.config = config

    @property
    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name="mock-asr",
            kind="asr",
            ready=True,
            message="Text input is treated as the final transcript.",
            capabilities=ProviderCapabilities(supports_partials=True),
            provider_id="mock",
        )

    async def check_status(self) -> ProviderStatus:
        return self.status

    async def load(self) -> ProviderStatus:
        return self.status

    async def transcribe_text_input(self, text: str) -> AsyncIterator[TranscriptEvent]:
        words = text.strip().split()
        if not words:
            return
        partial = []
        for word in words:
            partial.append(word)
            await asyncio.sleep(0.01)
            yield TranscriptEvent(text=" ".join(partial), final=False)
        yield TranscriptEvent(text=" ".join(words), final=True)

    async def transcribe_audio(
        self, pcm_s16le: bytes, sample_rate: int, progress=None
    ) -> AsyncIterator[TranscriptEvent]:
        if progress:
            await progress(
                "asr.progress",
                {"stage": "mock", "message": "Mock ASR transcribing audio."},
            )
        yield TranscriptEvent(text="Mock ASR heard audio input.", final=True)

    async def unload(self) -> ProviderStatus:
        return self.status


class MockLLMProvider(LLMProvider):
    def __init__(self, config: dict):
        self.first_token_delay = float(config.get("first_token_delay_ms", 120)) / 1000
        self.token_delay = float(config.get("token_delay_ms", 28)) / 1000

    @property
    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name="mock-llm",
            kind="llm",
            ready=True,
            message="Mock LLM streams a deterministic local response.",
            capabilities=ProviderCapabilities(),
            provider_id="mock",
        )

    async def check_status(self) -> ProviderStatus:
        return self.status

    async def stream_response(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        user_text = messages[-1]["content"] if messages else ""
        response = (
            f"I heard: {user_text}. This is the harness running in mock mode, "
            "with provider boundaries and latency events active."
        )
        await asyncio.sleep(self.first_token_delay)
        for token in response.split(" "):
            yield token + " "
            await asyncio.sleep(self.token_delay)


class MockTTSProvider(TTSProvider):
    def __init__(self, config: dict):
        self.first_chunk_delay = float(config.get("first_chunk_delay_ms", 80)) / 1000
        self.chunk_delay = float(config.get("chunk_delay_ms", 40)) / 1000

    @property
    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name="mock-tts",
            kind="tts",
            ready=True,
            message="Mock TTS emits short WAV tones so playback paths can be tested.",
            capabilities=ProviderCapabilities(supports_streaming_tts=True),
            provider_id="mock",
        )

    async def check_status(self) -> ProviderStatus:
        return self.status

    async def load(self) -> ProviderStatus:
        return self.status

    async def unload(self) -> ProviderStatus:
        return self.status

    async def stream_audio(self, text: str) -> AsyncIterator[AudioChunk]:
        await asyncio.sleep(self.first_chunk_delay)
        frequency = 420 + (len(text) % 160)
        yield AudioChunk(
            data=make_tone_wav(frequency=frequency), mime_type="audio/wav", final=True
        )
        await asyncio.sleep(self.chunk_delay)

    async def stream_audio_with_progress(
        self, text: str, progress=None
    ) -> AsyncIterator[AudioChunk]:
        if progress:
            await progress(
                "tts.progress",
                {"stage": "mock", "message": "Mock TTS generating tone."},
            )
        async for chunk in self.stream_audio(text):
            yield chunk
