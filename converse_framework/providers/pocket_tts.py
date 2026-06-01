"""Pocket TTS provider.

The ``pocket_tts`` package is imported lazily inside :meth:`_ensure_model`
so the base :mod:`converse_framework` package stays light. Install with::

    pip install 'converse-framework[pocket-tts]'
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import AsyncIterator

from converse_framework.audio_utils import float_audio_to_pcm_s16le_bytes
from converse_framework.protocols import (
    AudioChunk,
    ProgressCallback,
    ProviderCapabilities,
    ProviderStatus,
    TTSProvider,
)


class PocketTTSProvider(TTSProvider):
    def __init__(self, config: dict):
        self.voice = str(config.get("voice", "azelma"))
        self.language = config.get("language")
        self.temp = float(config.get("temp", 0.7))
        self.max_tokens = int(config.get("max_tokens", 50))
        self.quantize = bool(config.get("quantize", False))
        self.coalesce_ms = int(config.get("coalesce_ms", 400))
        self._model = config.get("_model")
        self._voice_state = config.get("_voice_state")
        self._load_error: str | None = None
        self._lock = threading.Lock()

    @property
    def status(self) -> ProviderStatus:
        if self._load_error:
            return ProviderStatus(
                name="pocket-tts",
                kind="tts",
                ready=False,
                message=f"Pocket TTS failed to load: {self._load_error}",
                capabilities=ProviderCapabilities(supports_streaming_tts=True),
                provider_id="pocket-tts",
                loaded=False,
                supports_model_management=True,
                supports_voice_selection=True,
            )
        mode = "int8" if self.quantize else "fp32"
        if self._model is not None and self._voice_state is not None:
            message = f"Loaded Pocket TTS voice '{self.voice}' ({mode})."
        else:
            message = (
                f"Configured for Pocket TTS voice '{self.voice}' ({mode}). "
                "Model and voice load on first TTS request."
            )
        return ProviderStatus(
            name="pocket-tts",
            kind="tts",
            ready=True,
            message=message,
            capabilities=ProviderCapabilities(
                supports_streaming_tts=True,
                languages=("en", "fr", "de", "pt", "it", "es"),
            ),
            provider_id="pocket-tts",
            loaded=self._model is not None and self._voice_state is not None,
            supports_model_management=True,
            supports_voice_selection=True,
        )

    async def check_status(self) -> ProviderStatus:
        try:
            import pocket_tts  # type: ignore[import-not-found]  # noqa: F401
        except Exception as exc:  # pragma: no cover - import path
            self._load_error = str(exc)
        return self.status

    async def load(self) -> ProviderStatus:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._ensure_model)
        return self.status

    async def unload(self) -> ProviderStatus:
        def release() -> None:
            with self._lock:
                self._model = None
                self._voice_state = None
                self._load_error = None

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
        queue: asyncio.Queue[AudioChunk | Exception | None] = asyncio.Queue()

        def worker() -> None:
            try:
                with self._lock:
                    started = time.perf_counter()
                    self._emit_progress(
                        loop, progress, "loading", f"Loading Pocket TTS voice '{self.voice}'."
                    )
                    self._ensure_model()
                    self._emit_progress(
                        loop,
                        progress,
                        "loaded",
                        f"Pocket TTS ready after {round(time.perf_counter() - started, 1)}s.",
                    )
                    self._emit_progress(loop, progress, "generating", "Generating speech.")
                    assert self._model is not None
                    target_samples = max(
                        1, int(self._model.sample_rate * self.coalesce_ms / 1000)
                    )
                    pending = bytearray()
                    pending_samples = 0
                    assert self._voice_state is not None
                    chunks = self._model.generate_audio_stream(
                        self._voice_state,
                        text,
                        max_tokens=self.max_tokens,
                        copy_state=True,
                    )
                    for index, audio in enumerate(chunks):
                        pcm_bytes = float_audio_to_pcm_s16le_bytes(audio)
                        if not pcm_bytes:
                            continue
                        pending.extend(pcm_bytes)
                        pending_samples += len(pcm_bytes) // 2
                        if pending_samples >= target_samples:
                            self._emit_progress(
                                loop,
                                progress,
                                "chunk",
                                f"Generated audio chunk {index + 1}.",
                            )
                            asyncio.run_coroutine_threadsafe(
                                queue.put(
                                    AudioChunk(
                                        bytes(pending),
                                        sample_rate=self._model.sample_rate,
                                        channels=1,
                                        encoding="pcm_s16le",
                                        duration_ms=int(
                                            pending_samples * 1000 / self._model.sample_rate
                                        ),
                                        final=False,
                                    )
                                ),
                                loop,
                            )
                            pending.clear()
                            pending_samples = 0
                    if pending:
                        asyncio.run_coroutine_threadsafe(
                            queue.put(
                                AudioChunk(
                                    bytes(pending),
                                    sample_rate=self._model.sample_rate,
                                    channels=1,
                                    encoding="pcm_s16le",
                                    duration_ms=int(
                                        pending_samples * 1000 / self._model.sample_rate
                                    ),
                                    final=True,
                                )
                            ),
                            loop,
                        )
                    self._emit_progress(loop, progress, "complete", "TTS complete.")
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop)
            except Exception as exc:  # pragma: no cover - threaded path
                self._load_error = str(exc)
                asyncio.run_coroutine_threadsafe(queue.put(exc), loop)

        threading.Thread(target=worker, daemon=True).start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    def _ensure_model(self) -> None:
        if self._model is not None and self._voice_state is not None:
            return
        from pocket_tts import TTSModel  # type: ignore[import-not-found]

        if self._model is None:
            kwargs: dict = {"temp": self.temp, "quantize": self.quantize}
            if self.language:
                kwargs["language"] = self.language
            self._model = TTSModel.load_model(**kwargs)
        if self._voice_state is None:
            assert self._model is not None
            self._voice_state = self._model.get_state_for_audio_prompt(self.voice)

    def _emit_progress(
        self,
        loop: asyncio.AbstractEventLoop,
        progress: ProgressCallback | None,
        stage: str,
        message: str,
    ) -> None:
        if not progress:
            return
        asyncio.run_coroutine_threadsafe(
            progress("tts.progress", {"stage": stage, "message": message}), loop
        )
