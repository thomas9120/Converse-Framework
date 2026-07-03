"""audio.cpp HTTP providers (OpenAI-compatible server).

Talks to a running ``audiocpp_server`` (the official audio.cpp HTTP
server binary) over HTTP. The provider does NOT manage the server
subprocess -- the user is expected to start ``audiocpp_server`` with a
config that registers the desired model ids (e.g. ``pocket-tts``,
``qwen3-tts``, ``qwen3-asr``) and configure this provider with the
matching ``model`` id. The framework only needs ``httpx`` to talk to
it; install with::

    pip install 'converse-framework[audio-cpp]'

The ``httpx`` package is imported lazily inside async methods so the
base :mod:`converse_framework` package stays light.

Two endpoints are used:

* ``POST /v1/audio/speech`` -- OpenAI-style text-to-audio. The server
  returns ``audio/wav`` by default; the TTS provider decodes it to PCM
  s16le and yields a single :class:`AudioChunk`.
* ``POST /v1/audio/transcriptions`` -- JSON transcription. The server
  reads the audio from a **server-local file path**, so the ASR
  provider writes the caller's PCM to a temp WAV inside a
  ``shared_dir`` that the server process can read. This means the
  provider and ``audiocpp_server`` must run on the same host (or share
  a mounted filesystem) -- the same local-server assumption the
  ``whisper-cpp`` and ``llamacpp`` providers already make.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import uuid
from collections.abc import AsyncIterator

from converse_framework.audio_utils import (
    float_audio_to_wav_bytes,
    pcm_s16le_to_float32,
    wav_bytes_to_pcm_s16le,
)
from converse_framework.protocols import (
    ASRProvider,
    AudioChunk,
    ProviderCapabilities,
    ProviderStatus,
    ProgressCallback,
    TTSProvider,
    TranscriptEvent,
)
from converse_framework.providers.unavailable import extra_hint_for

logger = logging.getLogger(__name__)

_DEFAULT_EXTRA_HINT = "converse-framework[audio-cpp]"
_DEFAULT_BASE_URL = "http://127.0.0.1:8080"


def _missing_dep_message(exc: Exception) -> str:
    hint = extra_hint_for("tts", "audio-cpp") or _DEFAULT_EXTRA_HINT
    return (
        f"Provider 'audio-cpp' is not available. "
        f"Install the required extra with `pip install {hint}`. ({exc})"
    )


def _missing_model_message(kind: str) -> str:
    return (
        f"audio-cpp {kind} provider requires a 'model' id registered in the "
        "audiocpp_server config."
    )


# ---------------------------------------------------------------------------
# Shared HTTP helpers
# ---------------------------------------------------------------------------


async def _import_httpx():
    try:
        import httpx  # type: ignore[import-not-found]

        return httpx
    except Exception as exc:  # pragma: no cover - import path
        raise RuntimeError(_missing_dep_message(exc)) from exc


async def _probe_server(base_url: str) -> tuple[bool, str, list[str]]:
    """Probe ``{base_url}/health`` and return ``(reachable, message, model_ids)``.

    On success also fetches ``/v1/models`` so callers can verify the
    configured model id is registered. Both calls use short timeouts so
    a status screen does not hang on an unresponsive server.
    """
    httpx = await _import_httpx()
    timeout = httpx.Timeout(connect=2.0, read=5.0, write=5.0, pool=2.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{base_url}/health")
            response.raise_for_status()
            health = response.json()
    except Exception as exc:
        return False, f"Cannot reach audiocpp_server at {base_url}: {exc}", []

    if health.get("status") != "ok":
        return (
            False,
            f"audiocpp_server reachable at {base_url} but not ready: {health}",
            [],
        )

    model_ids: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            models_response = await client.get(f"{base_url}/v1/models")
            models_response.raise_for_status()
            for item in models_response.json().get("data", []):
                model_id = item.get("id")
                if model_id:
                    model_ids.append(str(model_id))
    except Exception:
        # Health is the source of truth for readiness; model listing is
        # best-effort enrichment for diagnostics.
        pass

    return True, f"audiocpp_server reachable at {base_url} (/health ok).", model_ids


# ---------------------------------------------------------------------------
# TTS provider
# ---------------------------------------------------------------------------


class AudioCppTTSProvider(TTSProvider):
    """TTS provider that proxies text to ``audiocpp_server``'s speech endpoint.

    Config keys:

    * ``base_url`` -- audiocpp_server address. Default
      ``"http://127.0.0.1:8080"``.
    * ``model`` -- the model id registered in the server config
      (e.g. ``"pocket-tts"``, ``"qwen3-tts"``). This is required and
      must match an id the server knows about.
    * ``voice`` -- optional cached voice id the server recognises.
    * ``voice_ref`` -- optional path (read by the server on its host) to
      a reference audio file for voice cloning.
    * ``language`` -- optional language code forwarded to the model.
    * ``seed`` -- optional integer seed for reproducible synthesis.
    * ``temperature``, ``max_tokens``, ``max_steps`` -- optional
      generation parameters forwarded to the server.
    * ``timeout_s`` -- request timeout in seconds. Default ``120``.
    """

    def __init__(self, config: dict):
        self.base_url = str(config.get("base_url", _DEFAULT_BASE_URL)).rstrip("/")
        self.model = str(config.get("model", ""))
        self.voice = config.get("voice")
        self.voice_ref = config.get("voice_ref")
        self.language = config.get("language")
        self.seed = config.get("seed")
        self.temperature = config.get("temperature")
        self.max_tokens = config.get("max_tokens")
        self.max_steps = config.get("max_steps")
        self.timeout_s = float(config.get("timeout_s", 120))
        self._last_error: str | None = None
        self._ready: bool = False

    def _capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(supports_streaming_tts=False)

    @property
    def status(self) -> ProviderStatus:
        if self._last_error:
            message = self._last_error
            status_level = "error"
        elif self._ready:
            message = (
                f"Ready; audiocpp_server at {self.base_url}, model: {self.model or '(unset)'}."
            )
            status_level = "ready"
        else:
            message = (
                f"Configured for audiocpp_server at {self.base_url} "
                f"(model: {self.model or '(unset)'}). The server is managed "
                "externally; the framework will not start it for you."
            )
            status_level = "configured"
        return ProviderStatus(
            name="audio-cpp",
            kind="tts",
            ready=self._ready and self._last_error is None,
            message=message,
            capabilities=self._capabilities(),
            provider_id="audio-cpp",
            loaded=False,
            managed_externally=True,
            supports_model_management=False,
            supports_voice_selection=self.voice is not None,
            active_voice=self.voice,
            active_model=self.model or None,
            models=({"id": self.model, "label": self.model},) if self.model else (),
            status_level=status_level,
        )

    async def check_status(self) -> ProviderStatus:
        return await self._http_check_status()

    async def probe_status(self) -> ProviderStatus:
        """Cheap probe: check httpx import; no HTTP call."""
        try:
            import httpx  # type: ignore[import-not-found]  # noqa: F401
        except Exception as exc:  # pragma: no cover - import path
            self._last_error = _missing_dep_message(exc)
            return self.status
        return self.status

    async def load_status(self) -> ProviderStatus:
        return await self.probe_status()

    async def _http_check_status(self) -> ProviderStatus:
        if not self.model:
            self._ready = False
            self._last_error = _missing_model_message("TTS")
            return self.status

        try:
            import httpx  # type: ignore[import-not-found]  # noqa: F401
        except Exception as exc:  # pragma: no cover - import path
            self._last_error = _missing_dep_message(exc)
            return self.status

        reachable, message, model_ids = await _probe_server(self.base_url)
        if not reachable:
            self._last_error = message
            self._ready = False
            return self.status
        self._last_error = None
        if self.model and model_ids and self.model not in model_ids:
            self._ready = False
            self._last_error = (
                f"audiocpp_server is up, but configured model '{self.model}' is not "
                f"registered. Known ids: {', '.join(model_ids[:5])}"
            )
            return self.status
        self._ready = True
        return self.status

    async def load(self) -> ProviderStatus:
        return await self.check_status()

    def _build_speech_payload(self, text: str) -> dict:
        payload: dict = {"model": self.model, "input": text}
        if self.voice:
            payload["voice"] = self.voice
        if self.voice_ref:
            payload["voice_ref"] = self.voice_ref
        if self.language:
            payload["language"] = self.language
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.max_steps is not None:
            payload["max_steps"] = self.max_steps
        return payload

    async def _request_speech(self, text: str) -> bytes:
        """POST to ``/v1/audio/speech`` and return the raw WAV bytes."""
        if not self.model:
            raise RuntimeError(_missing_model_message("TTS"))
        httpx = await _import_httpx()
        url = f"{self.base_url}/v1/audio/speech"
        timeout = httpx.Timeout(connect=5.0, read=self.timeout_s, write=10.0, pool=5.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await asyncio.wait_for(
                    client.post(url, json=self._build_speech_payload(text)),
                    timeout=self.timeout_s,
                )
                response.raise_for_status()
        except asyncio.TimeoutError as exc:
            self._last_error = (
                f"audiocpp_server speech request timed out after {self.timeout_s}s"
            )
            raise RuntimeError(self._last_error) from exc
        except Exception as exc:
            self._last_error = f"audiocpp_server request to {url} failed: {exc}"
            raise RuntimeError(self._last_error) from exc

        # Default response_format is raw audio/wav bytes.
        content = response.content
        if not content or content[:4] != b"RIFF":
            # Some builds may answer with a JSON body when response_format=json
            # is ever set; decode base64 WAV defensively.
            try:
                payload = response.json()
                import base64

                wav = base64.b64decode(payload["audio"])
                if wav[:4] == b"RIFF":
                    return wav
            except Exception:
                pass
            self._last_error = "audiocpp_server returned no decodable WAV audio"
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
                {"stage": "started", "message": "Synthesising with audiocpp_server."},
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
                {"stage": "complete", "message": "audio.cpp synthesis complete."},
            )
        yield AudioChunk(
            data=pcm,
            sample_rate=sample_rate or None,
            channels=chunk_channels,
            encoding="pcm_s16le",
            duration_ms=duration_ms,
            final=True,
        )

    async def unload(self) -> ProviderStatus:
        # No state to release -- the server is owned by the user.
        return await self.check_status()


# ---------------------------------------------------------------------------
# ASR provider
# ---------------------------------------------------------------------------


class AudioCppASRProvider(ASRProvider):
    """ASR provider that proxies audio to ``audiocpp_server``'s transcription endpoint.

    The server reads the audio from a **server-local file path**, so this
    provider writes the caller's PCM s16le bytes to a temp WAV inside a
    ``shared_dir`` the server process can read. The provider and
    ``audiocpp_server`` must therefore run on the same host (or share a
    mounted filesystem).

    Config keys:

    * ``base_url`` -- audiocpp_server address. Default
      ``"http://127.0.0.1:8080"``.
    * ``model`` -- the model id registered in the server config
      (e.g. ``"qwen3-asr"``, ``"citrinet-asr"``). Required.
    * ``language`` -- optional language code forwarded to the server.
    * ``shared_dir`` -- directory on the **server host** where temp WAV
      files are written. Default is the OS temp dir; set this when the
      server runs in a different working directory or container mount.
    * ``cleanup_files`` -- delete the temp WAV after each request.
      Default ``True``.
    * ``timeout_s`` -- request timeout in seconds. Default ``120``.
    """

    def __init__(self, config: dict):
        self.base_url = str(config.get("base_url", _DEFAULT_BASE_URL)).rstrip("/")
        self.model = str(config.get("model", ""))
        self.language = config.get("language")
        self.shared_dir = config.get("shared_dir") or tempfile.gettempdir()
        self.cleanup_files = bool(config.get("cleanup_files", True))
        self.timeout_s = float(config.get("timeout_s", 120))
        self._last_error: str | None = None
        self._ready: bool = False

    @property
    def status(self) -> ProviderStatus:
        if self._last_error:
            message = self._last_error
            status_level = "error"
        elif self._ready:
            message = (
                f"Ready; audiocpp_server at {self.base_url}, model: {self.model or '(unset)'}."
            )
            status_level = "ready"
        else:
            message = (
                f"Configured for audiocpp_server at {self.base_url} "
                f"(model: {self.model or '(unset)'}). Transcription reads audio "
                f"from a server-local path written under '{self.shared_dir}' -- "
                "the provider and server must share a filesystem."
            )
            status_level = "configured"
        return ProviderStatus(
            name="audio-cpp",
            kind="asr",
            ready=self._ready and self._last_error is None,
            message=message,
            capabilities=ProviderCapabilities(
                languages=(str(self.language),) if self.language else ("en",)
            ),
            provider_id="audio-cpp",
            loaded=False,
            managed_externally=True,
            supports_model_management=False,
            supports_voice_selection=False,
            active_model=self.model or None,
            models=({"id": self.model, "label": self.model},) if self.model else (),
            status_level=status_level,
        )

    async def check_status(self) -> ProviderStatus:
        return await self._http_check_status()

    async def probe_status(self) -> ProviderStatus:
        """Cheap probe: check httpx import; no HTTP call."""
        try:
            import httpx  # type: ignore[import-not-found]  # noqa: F401
        except Exception as exc:  # pragma: no cover - import path
            self._last_error = _missing_dep_message(exc)
            return self.status
        return self.status

    async def load_status(self) -> ProviderStatus:
        return await self.probe_status()

    async def load(self) -> ProviderStatus:
        return await self.check_status()

    async def _http_check_status(self) -> ProviderStatus:
        if not self.model:
            self._ready = False
            self._last_error = _missing_model_message("ASR")
            return self.status

        try:
            import httpx  # type: ignore[import-not-found]  # noqa: F401
        except Exception as exc:  # pragma: no cover - import path
            self._last_error = _missing_dep_message(exc)
            return self.status

        reachable, message, model_ids = await _probe_server(self.base_url)
        if not reachable:
            self._last_error = message
            self._ready = False
            return self.status
        self._last_error = None
        if self.model and model_ids and self.model not in model_ids:
            self._ready = False
            self._last_error = (
                f"audiocpp_server is up, but configured model '{self.model}' is not "
                f"registered. Known ids: {', '.join(model_ids[:5])}"
            )
            return self.status
        self._ready = True
        return self.status

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
            raise RuntimeError(_missing_dep_message(exc)) from exc

        if sample_rate <= 0:
            raise ValueError(
                f"audio-cpp needs a positive sample_rate, got {sample_rate}"
            )
        if not pcm_s16le:
            return

        if not self.model:
            raise RuntimeError(_missing_model_message("ASR"))

        # The server reads audio from a local path, so materialise the
        # caller's PCM as a WAV inside the shared directory first.
        os.makedirs(self.shared_dir, exist_ok=True)
        wav_path = os.path.join(
            self.shared_dir, f"audiocpp_asr_{uuid.uuid4().hex}.wav"
        )
        float_audio = pcm_s16le_to_float32(pcm_s16le)
        wav_bytes = float_audio_to_wav_bytes(float_audio, sample_rate)
        if not wav_bytes:
            return
        with open(wav_path, "wb") as handle:
            handle.write(wav_bytes)

        if progress:
            await progress(
                "asr.progress",
                {
                    "stage": "queued",
                    "message": (
                        f"Queued {round(len(pcm_s16le) / 2 / sample_rate, 2)}s utterance "
                        "for audiocpp_server."
                    ),
                },
            )

        try:
            payload: dict = {"model": self.model, "audio": wav_path}
            if self.language:
                payload["language"] = self.language
            url = f"{self.base_url}/v1/audio/transcriptions"
            timeout = httpx.Timeout(connect=5.0, read=self.timeout_s, write=10.0, pool=5.0)
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await asyncio.wait_for(
                        client.post(url, json=payload),
                        timeout=self.timeout_s,
                    )
                    response.raise_for_status()
                    body = response.json()
            except asyncio.TimeoutError as exc:
                self._last_error = (
                    f"audiocpp_server transcription timed out after {self.timeout_s}s"
                )
                raise RuntimeError(self._last_error) from exc
            except Exception as exc:
                self._last_error = f"audiocpp_server request to {url} failed: {exc}"
                raise RuntimeError(self._last_error) from exc

            text = self._extract_text(body)
            if progress:
                await progress(
                    "asr.progress",
                    {"stage": "complete", "message": "ASR transcription complete."},
                )
            if text:
                yield TranscriptEvent(text=text, final=True)
        finally:
            if self.cleanup_files:
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

    async def unload(self) -> ProviderStatus:
        return await self.check_status()

    @staticmethod
    def _extract_text(payload) -> str:
        if not isinstance(payload, dict):
            return ""
        text = payload.get("text")
        if isinstance(text, str):
            return text.strip()
        return ""
