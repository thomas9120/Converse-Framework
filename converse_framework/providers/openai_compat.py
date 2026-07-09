"""Generic OpenAI-compatible providers (LLM, ASR, TTS).

Talk to any server that implements the OpenAI HTTP API surface:

* ``POST /v1/chat/completions`` -- :class:`OpenAICompatLLMProvider`
  (Ollama, LM Studio, vLLM, llama.cpp, Groq, OpenRouter, Together,
  OpenAI itself).
* ``POST /v1/audio/transcriptions`` -- :class:`OpenAICompatASRProvider`
  (OpenAI Whisper, Groq hosted Whisper, local servers such as
  ``speaches`` / faster-whisper-server).
* ``POST /v1/audio/speech`` -- :class:`OpenAICompatTTSProvider`
  (OpenAI TTS, Kokoro-FastAPI, openedai-speech).

Install with::

    pip install 'converse-framework[openai-compat]'

Configuration (each kind takes the same connection keys)::

    {
        "llm": {
            "provider": "openai-compatible",
            "base_url": "https://api.openai.com",  # no /v1 suffix
            "model": "gpt-4.1-mini",               # "auto" = first listed
            "api_key": "sk-...",                   # optional Bearer token
        },
        "asr": {
            "provider": "openai-compatible",
            "base_url": "https://api.groq.com/openai",
            "model": "whisper-large-v3",
            "api_key": "gsk_...",
        },
        "tts": {
            "provider": "openai-compatible",
            "base_url": "http://localhost:8880",   # e.g. Kokoro-FastAPI
            "model": "kokoro",
            "voice": "af_heart",
        },
    }

``base_url`` must not include the ``/v1`` path segment -- the providers
append the versioned paths themselves. The servers are managed
externally: the framework never starts or stops them.

The ``httpx`` package is imported lazily inside async methods so the
base :mod:`converse_framework` package stays light.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from converse_framework.audio_utils import (
    float_audio_to_wav_bytes,
    pcm_s16le_to_float32,
    wav_bytes_to_pcm_s16le,
)
from converse_framework.protocols import (
    ASRProvider,
    AudioChunk,
    ProgressCallback,
    ProviderCapabilities,
    ProviderStatus,
    TranscriptEvent,
    TTSProvider,
)
from converse_framework.providers.llamacpp import LlamaCppProvider
from converse_framework.providers.unavailable import extra_hint_for

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://127.0.0.1:8080"
_DEFAULT_EXTRA_HINT = "converse-framework[openai-compat]"


def _missing_dep_message(kind: str, exc: Exception) -> str:
    hint = extra_hint_for(kind, "openai-compatible") or _DEFAULT_EXTRA_HINT
    return (
        f"Provider 'openai-compatible' ({kind}) is not available. "
        f"Install the required extra with `pip install {hint}`. ({exc})"
    )


async def _probe_models(
    base_url: str, headers: dict[str, str]
) -> tuple[bool, str, list[str]]:
    """GET ``{base_url}/v1/models`` and return ``(reachable, message, model_ids)``.

    ``/v1/models`` is the one read-only endpoint every OpenAI-compatible
    server exposes, so it doubles as the health check. Short timeouts so
    a status screen does not hang on an unresponsive server.
    """
    import httpx  # type: ignore[import-not-found]

    timeout = httpx.Timeout(connect=2.0, read=5.0, write=5.0, pool=2.0)
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            response = await client.get(f"{base_url}/v1/models")
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        return (
            False,
            f"Cannot reach OpenAI-compatible server at {base_url}/v1/models: {exc}",
            [],
        )

    model_ids = [
        str(item.get("id", "unknown")) for item in payload.get("data", [])
    ]
    return True, f"OpenAI-compatible server reachable at {base_url}.", model_ids


class OpenAICompatLLMProvider(LlamaCppProvider):
    """Chat-completions LLM against any OpenAI-compatible server.

    Shares its implementation with
    :class:`~converse_framework.providers.llamacpp.LlamaCppProvider` but
    skips the llama.cpp-specific ``/health`` probe -- ``check_status``
    goes straight to ``/v1/models``. ``model`` defaults to ``"auto"``,
    which resolves to the first entry reported by ``/v1/models``; hosted
    services list many models, so set it explicitly for anything other
    than a single-model local server.
    """

    display_name = "openai-compatible"
    default_provider_id = "openai-compatible"
    install_extra = "openai-compat"
    use_health_endpoint = False


class _OpenAICompatAudioBase:
    """Shared connection/status plumbing for the ASR and TTS providers."""

    kind: str

    def __init__(self, config: dict):
        self.base_url = str(config.get("base_url", _DEFAULT_BASE_URL)).rstrip("/")
        self.model = str(config.get("model", ""))
        self.api_key = str(config.get("api_key", "")) or None
        self.timeout_s = float(config.get("timeout_s", 120))
        self._last_error: str | None = None
        self._ready: bool = False

    def _headers(self) -> dict[str, str]:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    async def check_status(self) -> ProviderStatus:
        return await self._http_check_status()

    async def probe_status(self) -> ProviderStatus:
        """Cheap probe: check httpx import; no HTTP call."""
        try:
            import httpx  # type: ignore[import-not-found]  # noqa: F401
        except Exception as exc:  # pragma: no cover - import path
            self._last_error = _missing_dep_message(self.kind, exc)
            return self.status

        return self.status

    async def load_status(self) -> ProviderStatus:
        return await self.probe_status()

    async def load(self) -> ProviderStatus:
        return await self.check_status()

    async def unload(self) -> ProviderStatus:
        # No state to release -- the server is owned by the user.
        return self.status

    async def _http_check_status(self) -> ProviderStatus:
        try:
            import httpx  # type: ignore[import-not-found]  # noqa: F401
        except Exception as exc:  # pragma: no cover - import path
            self._last_error = _missing_dep_message(self.kind, exc)
            return self.status

        reachable, message, model_ids = await _probe_models(
            self.base_url, self._headers()
        )
        if not reachable:
            self._last_error = message
            self._ready = False
            return self.status
        self._last_error = None
        if self.model and model_ids and self.model not in model_ids:
            self._ready = False
            self._last_error = (
                f"Server is up, but configured model '{self.model}' is not in "
                f"/v1/models. Known ids: {', '.join(model_ids[:5])}"
            )
            return self.status
        self._ready = True
        return self.status

    @property
    def status(self) -> ProviderStatus:  # pragma: no cover - overridden
        raise NotImplementedError


class OpenAICompatASRProvider(_OpenAICompatAudioBase, ASRProvider):
    """ASR against the OpenAI ``/v1/audio/transcriptions`` endpoint.

    Uploads the utterance as a multipart WAV (the standard OpenAI shape,
    unlike the ``audio-cpp`` provider's server-local file path), so it
    works with remote/hosted servers.

    Config keys (all optional unless the server requires them):

    * ``base_url`` -- server address without ``/v1``. Default
      ``"http://127.0.0.1:8080"``.
    * ``model`` -- model id sent with each request (e.g. ``"whisper-1"``
      for OpenAI, ``"whisper-large-v3"`` for Groq). Hosted services
      require it; some single-model local servers do not.
    * ``api_key`` -- optional Bearer token.
    * ``language`` -- optional ISO language code forwarded to the server.
    * ``temperature`` -- optional sampling temperature.
    * ``timeout_s`` -- request timeout in seconds. Default ``120``.
    """

    kind = "asr"

    def __init__(self, config: dict):
        super().__init__(config)
        self.language = config.get("language")
        self.temperature = config.get("temperature")

    @property
    def status(self) -> ProviderStatus:
        if self._last_error:
            message = self._last_error
            status_level = "error"
        elif self._ready:
            message = (
                f"Ready; OpenAI-compatible transcription at {self.base_url}, "
                f"model: {self.model or '(unset)'}."
            )
            status_level = "ready"
        else:
            message = (
                f"Configured for OpenAI-compatible transcription at "
                f"{self.base_url} (model: {self.model or '(unset)'}). The "
                "server is managed externally; the framework will not start "
                "it for you."
            )
            status_level = "configured"
        return ProviderStatus(
            name="openai-compatible",
            kind="asr",
            ready=self._ready and self._last_error is None,
            message=message,
            capabilities=ProviderCapabilities(
                languages=(str(self.language),) if self.language else ("en",)
            ),
            provider_id="openai-compatible",
            loaded=False,
            managed_externally=True,
            supports_model_management=False,
            supports_voice_selection=False,
            active_model=self.model or None,
            models=({"id": self.model, "label": self.model},) if self.model else (),
            status_level=status_level,
        )

    async def transcribe_text_input(self, text: str) -> AsyncIterator[TranscriptEvent]:
        stripped = text.strip()
        if stripped:
            yield TranscriptEvent(text=stripped, final=True)

    async def transcribe_audio(
        self,
        pcm_s16le: bytes,
        sample_rate: int,
        progress: ProgressCallback | None = None,
    ) -> AsyncIterator[TranscriptEvent]:
        try:
            import httpx  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - import path
            raise RuntimeError(_missing_dep_message("asr", exc)) from exc

        if sample_rate <= 0:
            raise ValueError(
                f"openai-compatible ASR needs a positive sample_rate, "
                f"got {sample_rate}"
            )
        if not pcm_s16le:
            return

        audio = pcm_s16le_to_float32(pcm_s16le)
        wav_bytes = float_audio_to_wav_bytes(audio, sample_rate)
        if not wav_bytes:
            return

        if progress:
            await progress(
                "asr.progress",
                {
                    "stage": "queued",
                    "message": (
                        f"Queued {round(audio.size / sample_rate, 2)}s utterance "
                        "for OpenAI-compatible transcription."
                    ),
                },
            )

        form: dict[str, str] = {"response_format": "json"}
        if self.model:
            form["model"] = self.model
        if self.language:
            form["language"] = str(self.language)
        if self.temperature is not None:
            form["temperature"] = str(self.temperature)

        url = f"{self.base_url}/v1/audio/transcriptions"
        timeout = httpx.Timeout(connect=5.0, read=self.timeout_s, write=10.0, pool=5.0)
        try:
            async with httpx.AsyncClient(
                timeout=timeout, headers=self._headers()
            ) as client:
                response = await asyncio.wait_for(
                    client.post(
                        url,
                        data=form,
                        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                    ),
                    timeout=self.timeout_s,
                )
                response.raise_for_status()
                payload = response.json()
        except asyncio.TimeoutError as exc:
            self._last_error = (
                f"OpenAI-compatible transcription timed out after {self.timeout_s}s"
            )
            raise RuntimeError(self._last_error) from exc
        except Exception as exc:
            self._last_error = f"Transcription request to {url} failed: {exc}"
            raise RuntimeError(self._last_error) from exc

        text = self._extract_text(payload)
        if progress:
            await progress(
                "asr.progress",
                {"stage": "complete", "message": "ASR transcription complete."},
            )
        if text:
            yield TranscriptEvent(text=text, final=True)

    @staticmethod
    def _extract_text(payload) -> str:
        if not isinstance(payload, dict):
            return ""
        text = payload.get("text")
        if isinstance(text, str):
            return text.strip()
        return ""


class OpenAICompatTTSProvider(_OpenAICompatAudioBase, TTSProvider):
    """TTS against the OpenAI ``/v1/audio/speech`` endpoint.

    Requests ``response_format: "wav"`` so the returned audio can be
    decoded to PCM s16le without a codec dependency, then yields it as a
    single final :class:`AudioChunk` (same contract as the ``audio-cpp``
    TTS provider).

    Config keys:

    * ``base_url`` -- server address without ``/v1``. Default
      ``"http://127.0.0.1:8080"``.
    * ``model`` -- model id (e.g. ``"tts-1"`` for OpenAI, ``"kokoro"``
      for Kokoro-FastAPI). Hosted services require it.
    * ``voice`` -- voice id (e.g. ``"alloy"``, ``"af_heart"``). OpenAI
      requires it; some local servers have a default.
    * ``api_key`` -- optional Bearer token.
    * ``speed`` -- optional playback speed multiplier.
    * ``timeout_s`` -- request timeout in seconds. Default ``120``.
    """

    kind = "tts"

    def __init__(self, config: dict):
        super().__init__(config)
        self.voice = config.get("voice")
        self.speed = config.get("speed")

    @property
    def status(self) -> ProviderStatus:
        if self._last_error:
            message = self._last_error
            status_level = "error"
        elif self._ready:
            message = (
                f"Ready; OpenAI-compatible speech at {self.base_url}, "
                f"model: {self.model or '(unset)'}, "
                f"voice: {self.voice or '(server default)'}."
            )
            status_level = "ready"
        else:
            message = (
                f"Configured for OpenAI-compatible speech at {self.base_url} "
                f"(model: {self.model or '(unset)'}, "
                f"voice: {self.voice or '(server default)'}). The server is "
                "managed externally; the framework will not start it for you."
            )
            status_level = "configured"
        return ProviderStatus(
            name="openai-compatible",
            kind="tts",
            ready=self._ready and self._last_error is None,
            message=message,
            capabilities=ProviderCapabilities(supports_streaming_tts=False),
            provider_id="openai-compatible",
            loaded=False,
            managed_externally=True,
            supports_model_management=False,
            supports_voice_selection=self.voice is not None,
            active_voice=self.voice,
            active_model=self.model or None,
            models=({"id": self.model, "label": self.model},) if self.model else (),
            status_level=status_level,
        )

    def _build_speech_payload(self, text: str) -> dict:
        payload: dict = {"input": text, "response_format": "wav"}
        if self.model:
            payload["model"] = self.model
        if self.voice:
            payload["voice"] = self.voice
        if self.speed is not None:
            payload["speed"] = self.speed
        return payload

    async def _request_speech(self, text: str) -> bytes:
        """POST to ``/v1/audio/speech`` and return the raw WAV bytes."""
        try:
            import httpx  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - import path
            raise RuntimeError(_missing_dep_message("tts", exc)) from exc

        url = f"{self.base_url}/v1/audio/speech"
        timeout = httpx.Timeout(connect=5.0, read=self.timeout_s, write=10.0, pool=5.0)
        try:
            async with httpx.AsyncClient(
                timeout=timeout, headers=self._headers()
            ) as client:
                response = await asyncio.wait_for(
                    client.post(url, json=self._build_speech_payload(text)),
                    timeout=self.timeout_s,
                )
                response.raise_for_status()
        except asyncio.TimeoutError as exc:
            self._last_error = (
                f"OpenAI-compatible speech request timed out after {self.timeout_s}s"
            )
            raise RuntimeError(self._last_error) from exc
        except Exception as exc:
            self._last_error = f"Speech request to {url} failed: {exc}"
            raise RuntimeError(self._last_error) from exc

        content = response.content
        if not content or content[:4] != b"RIFF":
            self._last_error = (
                "OpenAI-compatible server returned no decodable WAV audio "
                "(is response_format 'wav' supported?)"
            )
            raise RuntimeError(self._last_error)
        return content

    async def stream_audio(self, text: str) -> AsyncIterator[AudioChunk]:
        async for chunk in self.stream_audio_with_progress(text, None):
            yield chunk

    async def stream_audio_with_progress(
        self, text: str, progress: ProgressCallback | None = None
    ) -> AsyncIterator[AudioChunk]:
        stripped = text.strip()
        if not stripped:
            return
        if progress:
            await progress(
                "tts.progress",
                {
                    "stage": "started",
                    "message": "Synthesising with OpenAI-compatible server.",
                },
            )
        wav_bytes = await self._request_speech(stripped)
        pcm, sample_rate, channels = wav_bytes_to_pcm_s16le(wav_bytes)
        if not pcm:
            return
        chunk_channels = channels or 1
        duration_ms = (
            int((len(pcm) // (2 * chunk_channels)) * 1000 / sample_rate)
            if sample_rate and chunk_channels
            else None
        )
        if progress:
            await progress(
                "tts.progress",
                {"stage": "complete", "message": "Speech synthesis complete."},
            )
        yield AudioChunk(
            data=pcm,
            sample_rate=sample_rate or None,
            channels=chunk_channels,
            encoding="pcm_s16le",
            duration_ms=duration_ms,
            final=True,
        )


__all__ = [
    "OpenAICompatASRProvider",
    "OpenAICompatLLMProvider",
    "OpenAICompatTTSProvider",
]
