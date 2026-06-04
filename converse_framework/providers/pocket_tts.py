"""Pocket TTS provider.

The ``pocket_tts`` package is imported lazily inside :meth:`_ensure_model`
so the base :mod:`converse_framework` package stays light. Install with::

    pip install 'converse-framework[pocket-tts]'
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator

from converse_framework.audio_utils import float_audio_to_pcm_s16le_bytes
from converse_framework.protocols import (
    AudioChunk,
    ProgressCallback,
    ProviderCapabilities,
    ProviderConfigResult,
    ProviderStatus,
    TTSProvider,
    VoiceInfo,
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
        # Known voice identifiers for pocket-tts; listed here so status
        # can advertise them without importing the heavy backend.
        self._known_voices = (
            {"id": "alba", "label": "Alba", "language": "en"},
            {"id": "giovanni", "label": "Giovanni", "language": "it"},
            {"id": "lola", "label": "Lola", "language": "es"},
            {"id": "juergen", "label": "Juergen", "language": "de"},
            {"id": "rafael", "label": "Rafael", "language": "pt"},
            {"id": "estelle", "label": "Estelle", "language": "fr"},
            {"id": "anna", "label": "Anna", "language": "en"},
            {"id": "azelma", "label": "Azelma", "language": "en"},
            {"id": "bill_boerst", "label": "Bill Boerst", "language": "en"},
            {"id": "caro_davy", "label": "Caro Davy", "language": "en"},
            {"id": "charles", "label": "Charles", "language": "en"},
            {"id": "cosette", "label": "Cosette", "language": "en"},
            {"id": "eponine", "label": "Eponine", "language": "en"},
            {"id": "eve", "label": "Eve", "language": "en"},
            {"id": "fantine", "label": "Fantine", "language": "en"},
            {"id": "george", "label": "George", "language": "en"},
            {"id": "jane", "label": "Jane", "language": "en"},
            {"id": "jean", "label": "Jean", "language": "en"},
            {"id": "javert", "label": "Javert", "language": "en"},
            {"id": "marius", "label": "Marius", "language": "en"},
            {"id": "mary", "label": "Mary", "language": "en"},
            {"id": "michael", "label": "Michael", "language": "en"},
            {"id": "paul", "label": "Paul", "language": "en"},
            {"id": "peter_yearsley", "label": "Peter Yearsley", "language": "en"},
            {"id": "stuart_bell", "label": "Stuart Bell", "language": "en"},
            {"id": "vera", "label": "Vera", "language": "en"},
        )

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
                active_voice=self.voice,
                voices=self._known_voices,
                status_level="error",
            )
        mode = "int8" if self.quantize else "fp32"
        loaded = self._model is not None and self._voice_state is not None
        if loaded:
            message = f"Loaded Pocket TTS voice '{self.voice}' ({mode})."
            status_level = "ready"
        else:
            message = (
                f"Configured for Pocket TTS voice '{self.voice}' ({mode}). "
                "Model and voice load on first TTS request."
            )
            status_level = "configured"
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
            loaded=loaded,
            supports_model_management=True,
            supports_voice_selection=True,
            active_voice=self.voice,
            voices=self._known_voices,
            status_level=status_level,
        )

    async def check_status(self) -> ProviderStatus:
        return await self.probe_status()

    async def probe_status(self) -> ProviderStatus:
        """Cheap probe: check import availability, no model load."""
        try:
            import pocket_tts  # type: ignore[import-not-found]  # noqa: F401
        except Exception as exc:  # pragma: no cover - import path
            self._load_error = str(exc)
        return self.status

    async def load_status(self) -> ProviderStatus:
        """May load heavy resources."""
        return await self.load()

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

    def set_quantize(self, quantize: bool) -> ProviderStatus:
        """Switch quantization mode and unload cached model state if needed.

        The next :meth:`load` or synthesis request reloads Pocket TTS
        with the updated mode. If the requested mode is already active,
        loaded model state is kept.
        """
        requested = bool(quantize)
        with self._lock:
            if self.quantize == requested:
                return self.status
            self.quantize = requested
            self._model = None
            self._voice_state = None
            self._load_error = None
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
                        loop,
                        progress,
                        "loading",
                        f"Loading Pocket TTS voice '{self.voice}'.",
                    )
                    self._ensure_model()
                    self._emit_progress(
                        loop,
                        progress,
                        "loaded",
                        f"Pocket TTS ready after {round(time.perf_counter() - started, 1)}s.",
                    )
                    self._emit_progress(
                        loop, progress, "generating", "Generating speech."
                    )
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
                                            pending_samples
                                            * 1000
                                            / self._model.sample_rate
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

    def set_voice(self, voice: str) -> ProviderStatus:
        """Change the active voice without reloading the full model.

        Clears ``_voice_state`` (so the next synthesis reloads voice
        state) but keeps ``_model`` when the language / temp / quantize
        are unchanged.
        """
        with self._lock:
            if self.voice == voice:
                return self.status
            self.voice = voice
            # Clear only voice state — model stays loaded
            self._voice_state = None
            self._load_error = None
            return self.status

    async def configure(self, **options) -> ProviderConfigResult:
        """Apply configuration changes.

        Supported options:

        * ``voice`` — changes voice, reloads voice state only.
        * ``quantize`` — changes quantization, unloads model and voice.
        * ``language`` — changes language, unloads model and voice.
        * ``temp`` — changes temperature, unloads model and voice.
        * ``max_tokens`` — changes max tokens, no unload.
        * ``coalesce_ms`` — changes coalesce window, no unload.
        """
        changed = False
        requires_reload = False
        parts: list[str] = []

        with self._lock:
            if "voice" in options:
                v = str(options["voice"])
                if v != self.voice:
                    self.voice = v
                    self._voice_state = None
                    changed = True
                    requires_reload = True
                    parts.append(f"voice={v}")

            if "quantize" in options:
                q = bool(options["quantize"])
                if q != self.quantize:
                    self.quantize = q
                    self._model = None
                    self._voice_state = None
                    changed = True
                    requires_reload = True
                    parts.append(f"quantize={q}")

            if "language" in options:
                lang = options["language"]
                if lang != self.language:
                    self.language = lang
                    self._model = None
                    self._voice_state = None
                    changed = True
                    requires_reload = True
                    parts.append(f"language={lang}")

            if "temp" in options:
                t = float(options["temp"])
                if abs(t - self.temp) > 1e-6:
                    self.temp = t
                    self._model = None
                    self._voice_state = None
                    changed = True
                    requires_reload = True
                    parts.append(f"temp={t}")

            if "max_tokens" in options:
                m = int(options["max_tokens"])
                if m != self.max_tokens:
                    self.max_tokens = m
                    changed = True
                    parts.append(f"max_tokens={m}")

            if "coalesce_ms" in options:
                c = int(options["coalesce_ms"])
                if c != self.coalesce_ms:
                    self.coalesce_ms = c
                    changed = True
                    parts.append(f"coalesce_ms={c}")

            self._load_error = None
            message = ", ".join(parts) if parts else "no changes"

        return ProviderConfigResult(
            status=self.status,
            changed=changed,
            requires_reload=requires_reload,
            message=message,
        )

    def list_voices(self) -> tuple[VoiceInfo, ...]:
        """Return structured voice metadata.

        Returns the known voice list without importing the heavy
        ``pocket_tts`` backend.
        """
        return tuple(
            VoiceInfo(
                id=v["id"],
                label=v["label"],
                language=v.get("language", "en"),
                description=v.get("description", ""),
                gender=v.get("gender", "neutral"),
            )
            for v in self._known_voices
        )

    def _emit_progress(
        self,
        loop: asyncio.AbstractEventLoop,
        progress: ProgressCallback | None,
        stage: str,
        message: str,
    ) -> None:
        if not progress:
            return

        async def _fire() -> None:
            await progress("tts.progress", {"stage": stage, "message": message})

        asyncio.run_coroutine_threadsafe(_fire(), loop)
