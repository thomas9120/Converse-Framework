"""faster-whisper ASR provider.

The ``faster_whisper`` package is imported lazily inside
:meth:`_ensure_model` and :meth:`check_status` so the base
:mod:`converse_framework` package stays light. Install with::

    pip install 'converse-framework[faster-whisper]'
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator

from converse_framework.audio_utils import pcm_s16le_to_float32
from converse_framework.protocols import (
    ASRProvider,
    ProgressCallback,
    ProviderCapabilities,
    ProviderStatus,
    TranscriptEvent,
)

logger = logging.getLogger(__name__)


class FasterWhisperASRProvider(ASRProvider):
    def __init__(self, config: dict):
        self.model_name = str(config.get("model", "large-v3-turbo"))
        self.device = str(config.get("device", "auto"))
        self.compute_type = str(config.get("compute_type", "auto"))
        self.language = config.get("language", "en")
        self.beam_size = int(config.get("beam_size", 1))
        self.vad_filter = bool(config.get("vad_filter", False))
        self.initial_prompt = config.get("initial_prompt")
        self.condition_on_previous_text = bool(
            config.get("condition_on_previous_text", False)
        )
        self.temperature = config.get("temperature", 0)
        self.compression_ratio_threshold = config.get("compression_ratio_threshold", 2.4)
        self.log_prob_threshold = config.get("log_prob_threshold", -0.5)
        self.no_speech_threshold = config.get("no_speech_threshold", 0.2)
        self.suppress_tokens = config.get("suppress_tokens")
        self.timeout_s = float(config.get("timeout_s", 120))
        self._model = config.get("_model")
        self._load_error: str | None = None

    @property
    def status(self) -> ProviderStatus:
        if self._load_error:
            return ProviderStatus(
                name="faster-whisper",
                kind="asr",
                ready=False,
                message=f"faster-whisper failed to load: {self._load_error}",
                capabilities=ProviderCapabilities(),
            )
        if self._model is not None:
            message = f"Loaded {self.model_name} on {self.device}/{self.compute_type}."
        else:
            message = (
                f"Configured for {self.model_name} on {self.device}/{self.compute_type}. "
                "Model loads on first voice transcription and may download if not cached."
            )
        return ProviderStatus(
            name="faster-whisper",
            kind="asr",
            ready=True,
            message=message,
            capabilities=ProviderCapabilities(
                languages=(str(self.language),) if self.language else ("auto",)
            ),
            provider_id="faster-whisper",
        )

    async def check_status(self) -> ProviderStatus:
        if self._model is None:
            try:
                import faster_whisper  # type: ignore[import-not-found]  # noqa: F401
            except Exception as exc:  # pragma: no cover - import path
                self._load_error = str(exc)
        return self.status

    async def load(self) -> ProviderStatus:
        if self._model is not None:
            return self.status
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._ensure_model), timeout=self.timeout_s
            )
        except asyncio.TimeoutError:
            self._load_error = f"Model load timed out after {self.timeout_s}s"
            raise
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
        if sample_rate != 16000:
            raise ValueError(
                f"faster-whisper expects 16000 Hz audio, got {sample_rate}"
            )
        audio = pcm_s16le_to_float32(pcm_s16le)
        if audio.size == 0:
            return
        if progress:
            await progress(
                "asr.progress",
                {
                    "stage": "queued",
                    "message": (
                        f"Queued {round(audio.size / sample_rate, 2)}s utterance "
                        "for faster-whisper."
                    ),
                },
            )
        loop = asyncio.get_running_loop()
        try:
            segments_text = await asyncio.wait_for(
                asyncio.to_thread(self._transcribe_blocking, audio, progress, loop),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError:
            if self._model is None:
                self._load_error = (
                    f"Model load or transcription timed out after {self.timeout_s}s"
                )
            raise
        text = " ".join(part for part in segments_text if part).strip()
        if progress:
            await progress(
                "asr.progress",
                {"stage": "complete", "message": "ASR transcription complete."},
            )
        if text:
            yield TranscriptEvent(text=text, final=True)

    def _transcribe_blocking(
        self,
        audio,
        progress: ProgressCallback | None,
        loop: asyncio.AbstractEventLoop,
    ) -> list[str]:
        assert self._model is not None
        started = time.perf_counter()
        logger.info(
            "[ASR] transcribe_blocking called, audio length=%d samples (%.2fs)",
            audio.size,
            audio.size / 16000,
        )
        self._emit_progress_threadsafe(
            loop, progress, "loading", f"Loading faster-whisper model {self.model_name}."
        )
        logger.info(
            "[ASR] model loaded in %.1fs, starting inference on %s/%s",
            time.perf_counter() - started,
            self.device,
            self.compute_type,
        )
        self._emit_progress_threadsafe(
            loop,
            progress,
            "loaded",
            f"Model ready after {round(time.perf_counter() - started, 1)}s. "
            "Running inference.",
        )
        transcribe_options = {
            "language": self.language,
            "beam_size": self.beam_size,
            "vad_filter": self.vad_filter,
            "initial_prompt": self.initial_prompt,
            "condition_on_previous_text": self.condition_on_previous_text,
            "temperature": self.temperature,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            "log_prob_threshold": self.log_prob_threshold,
            "no_speech_threshold": self.no_speech_threshold,
        }
        if self.suppress_tokens is not None:
            transcribe_options["suppress_tokens"] = self.suppress_tokens
        segments, _info = self._model.transcribe(audio, **transcribe_options)
        logger.info("[ASR] inference call returned, iterating segments...")
        texts: list[str] = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                texts.append(text)
                start = getattr(segment, "start", None)
                end = getattr(segment, "end", None)
                prefix = ""
                if start is not None and end is not None:
                    prefix = (
                        f"Segment {round(float(start), 2)}-"
                        f"{round(float(end), 2)}s: "
                    )
                self._emit_progress_threadsafe(
                    loop, progress, "segment", f"{prefix}{text}"
                )
        logger.info(
            "[ASR] all segments collected in %.1fs, %d segments with text",
            time.perf_counter() - started,
            len(texts),
        )
        return texts

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]

            self._model = WhisperModel(
                self.model_name, device=self.device, compute_type=self.compute_type
            )
        except Exception as exc:  # pragma: no cover - import path
            self._load_error = str(exc)
            raise

    async def unload(self) -> ProviderStatus:
        if self._model is not None:
            logger.info(
                "[ASR] unloading faster-whisper model (%s/%s)",
                self.device,
                self.compute_type,
            )
            self._model = None
        self._load_error = None
        return self.status

    def _emit_progress_threadsafe(
        self,
        loop: asyncio.AbstractEventLoop,
        progress: ProgressCallback | None,
        stage: str,
        message: str,
    ) -> None:
        if not progress:
            return
        asyncio.run_coroutine_threadsafe(
            progress("asr.progress", {"stage": stage, "message": message}), loop
        )
