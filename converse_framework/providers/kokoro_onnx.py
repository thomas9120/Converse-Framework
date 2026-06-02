"""Kokoro ONNX TTS provider.

The ``kokoro_onnx``, ``misaki``, and ``httpx`` packages are imported lazily
inside :meth:`_ensure_model` and :meth:`_download_asset` so the base
:mod:`converse_framework` package stays light. Install with::

    pip install 'converse-framework[kokoro]'

The default cache directory is platform-aware and does not depend on the
harness ``PROJECT_ROOT``. The cache location can be overridden via the
``cache_dir`` config key, or by setting the ``CONVERSE_FRAMEWORK_CACHE_DIR``
environment variable (the provider appends ``/kokoro`` to that path).
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

from converse_framework.audio_utils import float_audio_to_pcm_s16le_bytes
from converse_framework.protocols import (
    AudioChunk,
    ProgressCallback,
    ProviderCapabilities,
    ProviderStatus,
    TTSProvider,
)


DEFAULT_KOKORO_MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
    "kokoro-v1.0.int8.onnx"
)
DEFAULT_KOKORO_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
    "voices-v1.0.bin"
)


def _default_cache_dir() -> Path:
    """Return a platform-appropriate cache directory for Kokoro assets.

    Resolution order:
      1. ``CONVERSE_FRAMEWORK_CACHE_DIR`` environment variable, with ``kokoro``
         appended.
      2. ``~/.cache/converse-framework/kokoro`` on POSIX-likes.
      3. ``%LOCALAPPDATA%/converse-framework/kokoro`` (or
         ``~/.cache/converse-framework/kokoro``) on Windows.
    """
    env_root = os.environ.get("CONVERSE_FRAMEWORK_CACHE_DIR")
    if env_root:
        return Path(env_root) / "kokoro"
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "converse-framework" / "kokoro"
    return Path.home() / ".cache" / "converse-framework" / "kokoro"


class KokoroOnnxProvider(TTSProvider):
    def __init__(self, config: dict):
        self.voice = str(config.get("voice", "af_heart"))
        self.lang = str(config.get("lang", "en-us"))
        self.speed = float(config.get("speed", 1.0))
        self.trim = bool(config.get("trim", True))
        self.timeout_s = float(config.get("timeout_s", 300))
        configured_cache = config.get("cache_dir")
        self.cache_dir = (
            Path(str(configured_cache)) if configured_cache else _default_cache_dir()
        )
        self.model_filename = str(config.get("model_filename", "kokoro-v1.0.int8.onnx"))
        self.voices_filename = str(config.get("voices_filename", "voices-v1.0.bin"))
        self.model_url = str(config.get("model_url", DEFAULT_KOKORO_MODEL_URL))
        self.voices_url = str(config.get("voices_url", DEFAULT_KOKORO_VOICES_URL))
        self.onnx_intra_op_num_threads = int(config.get("onnx_intra_op_num_threads", 4))
        self.onnx_inter_op_num_threads = int(config.get("onnx_inter_op_num_threads", 1))
        self.preload_g2p = bool(config.get("preload_g2p", True))
        self._model = config.get("_model")
        self._g2p = config.get("_g2p")
        self._load_error: str | None = None
        self._g2p_error: str | None = None
        self._lock = threading.Lock()
        self._generation_lock = asyncio.Lock()

    @property
    def status(self) -> ProviderStatus:
        loaded = self._model is not None
        if self._load_error:
            return ProviderStatus(
                name="kokoro-onnx",
                kind="tts",
                ready=False,
                message=f"Kokoro ONNX failed to load: {self._load_error}",
                capabilities=ProviderCapabilities(
                    supports_streaming_tts=True, languages=("en",)
                ),
                provider_id="kokoro-onnx",
                loaded=False,
                supports_model_management=True,
                supports_voice_selection=True,
                active_voice=self.voice,
                status_level="error",
            )
        if self._g2p_error:
            return ProviderStatus(
                name="kokoro-onnx",
                kind="tts",
                ready=False,
                message=f"Kokoro English G2P failed: {self._g2p_error}",
                capabilities=ProviderCapabilities(
                    supports_streaming_tts=True, languages=("en",)
                ),
                provider_id="kokoro-onnx",
                loaded=loaded,
                supports_model_management=True,
                supports_voice_selection=True,
                active_voice=self.voice,
                status_level="error",
            )

        if loaded:
            message = f"Loaded Kokoro v1.0 ONNX voice '{self.voice}' ({self.lang})."
            status_level = "ready"
        else:
            message = (
                f"Configured for Kokoro v1.0 ONNX voice '{self.voice}' ({self.lang}). "
                "Model loads on first TTS request."
            )
            status_level = "configured"
        return ProviderStatus(
            name="kokoro-onnx",
            kind="tts",
            ready=True,
            message=message,
            capabilities=ProviderCapabilities(
                supports_streaming_tts=True, languages=("en",)
            ),
            provider_id="kokoro-onnx",
            loaded=loaded,
            supports_model_management=True,
            supports_voice_selection=True,
            active_voice=self.voice,
            status_level=status_level,
        )

    async def check_status(self) -> ProviderStatus:
        return await self.probe_status()

    async def probe_status(self) -> ProviderStatus:
        """Cheap probe: check import availability, no model load."""
        try:
            import kokoro_onnx  # type: ignore[import-not-found]  # noqa: F401

            if self._should_use_misaki():
                from misaki import en as _en  # type: ignore[import-not-found]  # noqa: F401
                from misaki import espeak as _espeak  # type: ignore[import-not-found]  # noqa: F401
        except Exception as exc:  # pragma: no cover - import path
            if self._should_use_misaki():
                self._g2p_error = str(exc)
            else:
                self._load_error = str(exc)
        return self.status

    async def load_status(self) -> ProviderStatus:
        """May load heavy resources."""
        return await self.load()

    async def load(self) -> ProviderStatus:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._ensure_model)
        if self.preload_g2p and self._should_use_misaki():
            await loop.run_in_executor(None, self._ensure_g2p)
        return self.status

    async def unload(self) -> ProviderStatus:
        def release() -> None:
            with self._lock:
                self._model = None
                self._load_error = None
                self._g2p = None
                self._g2p_error = None

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, release)
        return self.status

    async def stream_audio(self, text: str) -> AsyncIterator[AudioChunk]:
        async for chunk in self.stream_audio_with_progress(text):
            yield chunk

    async def stream_audio_with_progress(
        self,
        text: str,
        progress: ProgressCallback | None = None,
    ) -> AsyncIterator[AudioChunk]:
        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        self._emit_progress(
            loop,
            progress,
            "loading",
            f"Loading Kokoro voice '{self.voice}'.",
            started=started,
        )
        await loop.run_in_executor(None, self._ensure_model)
        self._emit_progress(loop, progress, "loaded", "Kokoro ready.", started=started)

        async with self._generation_lock:
            stream_text = text
            is_phonemes = False
            if self._should_use_misaki():
                self._emit_progress(
                    loop,
                    progress,
                    "phonemizing",
                    "Preparing English phonemes with Misaki.",
                    started=started,
                )
                stream_text = await loop.run_in_executor(
                    None, self._phonemize_english, text
                )
                is_phonemes = True

            self._emit_progress(
                loop,
                progress,
                "generating",
                "Generating speech.",
                started=started,
            )
            index = 0
            previous_chunk: AudioChunk | None = None
            assert self._model is not None
            async for audio, sample_rate in self._model.create_stream(
                stream_text,
                voice=self.voice,
                speed=self.speed,
                lang=self.lang,
                is_phonemes=is_phonemes,
                trim=self.trim,
            ):
                pcm_bytes = float_audio_to_pcm_s16le_bytes(audio)
                if not pcm_bytes:
                    continue
                index += 1
                current_chunk = AudioChunk(
                    pcm_bytes,
                    sample_rate=sample_rate,
                    channels=1,
                    encoding="pcm_s16le",
                    duration_ms=(
                        int((len(pcm_bytes) // 2) * 1000 / sample_rate)
                        if sample_rate
                        else None
                    ),
                    final=False,
                )
                self._emit_progress(
                    loop,
                    progress,
                    "chunk",
                    f"Generated audio chunk {index}.",
                    started=started,
                    chunk_index=index,
                    first_chunk=index == 1,
                    duration_ms=current_chunk.duration_ms,
                )
                if previous_chunk is not None:
                    yield previous_chunk
                previous_chunk = current_chunk

            if previous_chunk is not None:
                yield replace(previous_chunk, final=True)
            self._emit_progress(
                loop,
                progress,
                "complete",
                "TTS complete.",
                started=started,
                chunks=index,
            )

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        model_path = self._download_asset(self.model_url, self.model_filename)
        voices_path = self._download_asset(self.voices_url, self.voices_filename)
        from kokoro_onnx import Kokoro  # type: ignore[import-not-found]

        self._model = Kokoro(str(model_path), str(voices_path))
        self._apply_onnx_session_options(str(model_path))
        self._load_error = None
        self._g2p_error = None

    def _apply_onnx_session_options(self, model_path: str) -> None:
        if self.onnx_intra_op_num_threads <= 0 and self.onnx_inter_op_num_threads <= 0:
            return
        if self._model is None or not hasattr(self._model, "sess"):
            return
        import onnxruntime as ort  # type: ignore[import-not-found]

        options = ort.SessionOptions()
        if self.onnx_intra_op_num_threads > 0:
            options.intra_op_num_threads = self.onnx_intra_op_num_threads
        if self.onnx_inter_op_num_threads > 0:
            options.inter_op_num_threads = self.onnx_inter_op_num_threads
        providers = ["CPUExecutionProvider"]
        self._model.sess = ort.InferenceSession(
            model_path, sess_options=options, providers=providers
        )

    def _ensure_g2p(self):
        if self._g2p is not None:
            return self._g2p
        british = self._use_british_english()
        from misaki import en, espeak  # type: ignore[import-not-found]

        self._g2p = en.G2P(
            trf=False,
            british=british,
            fallback=espeak.EspeakFallback(british=british),
        )
        self._g2p_error = None
        return self._g2p

    def _phonemize_english(self, text: str) -> str:
        try:
            g2p = self._ensure_g2p()
            phonemes, _tokens = g2p(text)
            return str(phonemes).strip()
        except Exception as exc:  # pragma: no cover - exercised via tests
            self._g2p_error = str(exc)
            raise RuntimeError(f"Misaki English phonemization failed: {exc}") from exc

    def _should_use_misaki(self) -> bool:
        return self.lang.lower().startswith("en")

    def _use_british_english(self) -> bool:
        lang = self.lang.lower()
        return lang.startswith("en-gb") or self.voice.lower().startswith("b")

    def _download_asset(self, url: str, filename: str) -> Path:
        try:
            import httpx  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - import path
            raise RuntimeError(
                "kokoro-onnx provider requires httpx; install with "
                "pip install 'converse-framework[kokoro]'."
            ) from exc
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        target = self.cache_dir / filename
        if target.exists():
            return target
        temp = target.with_suffix(target.suffix + ".part")
        with (
            httpx.Client(follow_redirects=True, timeout=self.timeout_s) as client,
            temp.open("wb") as handle,
        ):
            with client.stream("GET", url) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    if chunk:
                        handle.write(chunk)
        temp.replace(target)
        return target

    def _emit_progress(
        self,
        loop: asyncio.AbstractEventLoop,
        progress: ProgressCallback | None,
        stage: str,
        message: str,
        *,
        started: float | None = None,
        **extra,
    ) -> None:
        if not progress:
            return
        payload = {"stage": stage, "message": message, **extra}
        if started is not None:
            payload["elapsed_ms"] = int((time.perf_counter() - started) * 1000)

        async def _fire() -> None:
            await progress("tts.progress", payload)

        loop.call_soon_threadsafe(asyncio.create_task, _fire())
