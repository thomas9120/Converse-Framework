"""whisper.cpp HTTP ASR provider.

Talks to a running ``whisper-server`` (whisper.cpp's official HTTP server
binary) over HTTP. The provider does NOT manage the server subprocess --
the user is expected to start ``whisper-server`` themselves and configure
it with the desired model. The framework only needs ``httpx`` to talk
to it; install with::

    pip install 'converse-framework[whisper-cpp]'

The ``httpx`` package is imported lazily inside async methods so the
base :mod:`converse_framework` package stays light. The provider probes
``{base_url}/health`` on :meth:`check_status` to decide whether to use
the legacy ``/inference`` endpoint (older whisper-server releases) or
the OpenAI-compatible ``/v1/audio/transcriptions`` endpoint (newer
releases). Both endpoints return a JSON object with a ``text`` field.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

from converse_framework.audio_utils import (
    float_audio_to_wav_bytes,
    pcm_s16le_to_float32,
)
from converse_framework.protocols import (
    ASRProvider,
    ProgressCallback,
    ProviderCapabilities,
    ProviderStatus,
    TranscriptEvent,
)
from converse_framework.providers.unavailable import extra_hint_for

logger = logging.getLogger(__name__)

_DEFAULT_EXTRA_HINT = "converse-framework[whisper-cpp]"


class WhisperCppASRProvider(ASRProvider):
    """ASR provider that proxies audio to a ``whisper-server`` HTTP endpoint.

    Config keys (all optional):

    * ``base_url`` -- whisper-server address. Default
      ``"http://127.0.0.1:8082"``.
    * ``model`` -- model filename hint stored alongside the server
      config (e.g. ``"ggml-small.en.bin"``). Default
      ``"ggml-small.en.bin"``. whisper-server itself picks the
      model on its own command line, so this key is for diagnostics
      and status messages only.
    * ``language`` -- ISO language code sent with each request.
      Default ``"en"``.
    * ``temperature`` -- sampling temperature forwarded to
      whisper-server. Default ``0`` (greedy).
    * ``timeout_s`` -- request timeout in seconds. Default ``120``.
    """

    def __init__(self, config: dict):
        self.base_url = str(
            config.get("base_url", "http://127.0.0.1:8082")
        ).rstrip("/")
        self.model = str(config.get("model", "ggml-small.en.bin"))
        self.language = config.get("language", "en")
        self.temperature = config.get("temperature", 0)
        self.timeout_s = float(config.get("timeout_s", 120))
        # The provider never starts or stops the server, so the bundle
        # has no subprocess lifecycle to track.
        self._endpoint: str | None = None
        self._last_error: str | None = None

    @property
    def status(self) -> ProviderStatus:
        if self._last_error:
            message = self._last_error
        else:
            message = (
                f"Configured for whisper-server at {self.base_url} "
                f"(model hint: {self.model}, language: {self.language}). "
                "The server is managed externally; the framework will not "
                "start it for you."
            )
        return ProviderStatus(
            name="whisper-cpp",
            kind="asr",
            ready=self._last_error is None,
            message=message,
            capabilities=ProviderCapabilities(
                languages=(str(self.language),) if self.language else ("en",)
            ),
            provider_id="whisper-cpp",
            # The provider does not own the heavy backend; the user does.
            loaded=False,
            managed_externally=True,
            supports_model_management=False,
            supports_voice_selection=False,
        )

    async def check_status(self) -> ProviderStatus:
        """Probe the server by GETting ``{base_url}/health``.

        Returns a :class:`ProviderStatus` whose ``ready`` reflects
        whether the server is reachable. As a side effect it caches
        the preferred transcription endpoint so the next
        :meth:`transcribe_audio` call skips the probe.
        """
        try:
            import httpx  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - import path
            self._last_error = self._missing_dep_message(exc)
            return self.status

        timeout = httpx.Timeout(connect=2.0, read=5.0, write=5.0, pool=2.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(f"{self.base_url}/health")
                response.raise_for_status()
        except Exception as exc:
            self._last_error = (
                f"Cannot reach whisper-server at {self.base_url}: {exc}"
            )
            self._endpoint = None
            return self.status

        # Server is up -- pick the endpoint once. Older whisper-server
        # builds only expose /inference; newer builds also expose the
        # OpenAI-compatible /v1/audio/transcriptions. We always try
        # /inference first when /health answers, mirroring the
        # upstream docs.
        self._endpoint = "/inference"
        self._last_error = None
        return ProviderStatus(
            name="whisper-cpp",
            kind="asr",
            ready=True,
            message=(
                f"whisper-server reachable at {self.base_url} "
                f"(/health 200, will use {self._endpoint}). "
                f"Model hint: {self.model}."
            ),
            capabilities=ProviderCapabilities(
                languages=(str(self.language),) if self.language else ("en",)
            ),
            provider_id="whisper-cpp",
            loaded=False,
            managed_externally=True,
            supports_model_management=False,
            supports_voice_selection=False,
        )

    async def load(self) -> ProviderStatus:
        """No-op: the server lifecycle is owned by the user.

        We still call :meth:`check_status` so callers see the
        current reachability state in the returned snapshot.
        """
        return await self.check_status()

    async def transcribe_text_input(
        self, text: str
    ) -> AsyncIterator[TranscriptEvent]:
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
            raise RuntimeError(self._missing_dep_message(exc)) from exc

        if sample_rate <= 0:
            raise ValueError(
                f"whisper-cpp needs a positive sample_rate, got {sample_rate}"
            )
        if not pcm_s16le:
            return

        audio = pcm_s16le_to_float32(pcm_s16le)
        wav_bytes = float_audio_to_wav_bytes(audio, sample_rate)
        if not wav_bytes:
            return

        endpoint = await self._resolve_endpoint()
        url = f"{self.base_url}{endpoint}"

        if progress:
            await progress(
                "asr.progress",
                {
                    "stage": "queued",
                    "message": (
                        f"Queued {round(audio.size / sample_rate, 2)}s utterance "
                        "for whisper.cpp."
                    ),
                },
            )

        params: dict[str, str] = {}
        if self.language:
            params["language"] = str(self.language)
        if self.temperature is not None:
            params["temperature"] = str(self.temperature)
        # whisper-server's /inference endpoint reads the language and
        # temperature from query string; /v1/audio/transcriptions reads
        # them from the multipart form. Use the right shape for each.
        if endpoint == "/inference":
            request_kwargs: dict = {"params": params} if params else {}
        else:
            form: dict[str, str] = {}
            if self.language:
                form["language"] = str(self.language)
            if self.temperature is not None:
                form["temperature"] = str(self.temperature)
            form["response_format"] = "json"
            request_kwargs = {"data": form, "files": {"file": wav_bytes}}

        timeout = httpx.Timeout(
            connect=5.0, read=self.timeout_s, write=10.0, pool=5.0
        )
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await asyncio.wait_for(
                    client.post(url, content=wav_bytes, **request_kwargs),
                    timeout=self.timeout_s,
                )
                response.raise_for_status()
                payload = response.json()
        except asyncio.TimeoutError as exc:
            self._last_error = (
                f"whisper-server request timed out after {self.timeout_s}s"
            )
            raise RuntimeError(self._last_error) from exc
        except Exception as exc:
            self._last_error = (
                f"whisper-server request to {url} failed: {exc}"
            )
            raise RuntimeError(self._last_error) from exc

        text = self._extract_text(payload)
        if progress:
            await progress(
                "asr.progress",
                {"stage": "complete", "message": "ASR transcription complete."},
            )
        if text:
            yield TranscriptEvent(text=text, final=True)

    async def unload(self) -> ProviderStatus:
        # No state to release -- the server is owned by the user.
        self._endpoint = None
        return await self.check_status()

    async def _resolve_endpoint(self) -> str:
        """Pick the transcription endpoint the running server actually exposes.

        First call probes the legacy ``/inference`` endpoint with a tiny
        HEAD-ish request; if it 404s or returns an error we fall back to
        the OpenAI-compatible ``/v1/audio/transcriptions``. The result
        is cached on the instance so subsequent calls are cheap.
        """
        if self._endpoint is not None:
            return self._endpoint

        import httpx  # type: ignore[import-not-found]

        timeout = httpx.Timeout(connect=2.0, read=3.0, write=3.0, pool=2.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            for candidate in ("/inference", "/v1/audio/transcriptions"):
                try:
                    response = await client.get(f"{self.base_url}{candidate}")
                except Exception:
                    continue
                # 405 (method not allowed) means the endpoint exists
                # and only accepts POST; that is the exact response we
                # expect from /inference. 404 means it is not there.
                if response.status_code in (200, 405):
                    self._endpoint = candidate
                    return candidate

        # Nothing matched -- default to the legacy endpoint and let the
        # POST surface a clear error.
        self._endpoint = "/inference"
        return self._endpoint

    @staticmethod
    def _extract_text(payload) -> str:
        """Pull the transcript out of either endpoint's JSON shape."""
        if not isinstance(payload, dict):
            return ""
        text = payload.get("text")
        if isinstance(text, str):
            return text.strip()
        return ""

    @staticmethod
    def _missing_dep_message(exc: Exception) -> str:
        hint = extra_hint_for("asr", "whisper-cpp") or _DEFAULT_EXTRA_HINT
        return (
            f"Provider 'whisper-cpp' (asr) is not available. "
            f"Install the required extra with `pip install {hint}`. ({exc})"
        )
