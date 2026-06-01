"""Silero VAD provider.

Heavy dependencies (``silero-vad``, ``torch``, ``onnxruntime``) are imported
lazily inside :meth:`_ensure_model` so the base :mod:`converse_framework`
package stays light. Install with::

    pip install 'converse-framework[silero]'
"""

from __future__ import annotations

import struct
from typing import Protocol

from converse_framework.audio_utils import AudioFrame
from converse_framework.protocols import (
    ProviderCapabilities,
    ProviderStatus,
    VADEvent,
    VADProvider,
)


class SileroModel(Protocol):
    def __call__(self, chunk, sample_rate: int): ...

    def reset_states(self) -> None: ...


class SileroVADProvider(VADProvider):
    def __init__(self, config: dict):
        self.threshold = float(config.get("speech_threshold", 0.6))
        self.neg_threshold = float(
            config.get("neg_threshold", max(0.15, self.threshold - 0.15))
        )
        self.hangover_ms = int(config.get("hangover_ms", 450))
        self.window_samples = int(config.get("window_samples", 512))
        self.sample_rate = int(config.get("sample_rate", 16000))
        self._model: SileroModel | None = config.get("_model")
        self._torch = None
        self._buffer = bytearray()
        self._speaking = False
        self._silence_ms = 0
        self._audio_ms = 0
        self._load_error: str | None = None

    @property
    def status(self) -> ProviderStatus:
        ready = self._model is not None and self._load_error is None
        if self._load_error:
            message = f"Silero VAD failed to load: {self._load_error}"
        elif ready:
            message = "Silero VAD ONNX model loaded."
        else:
            message = (
                "Silero VAD is configured and will load on first status check."
            )
        return ProviderStatus(
            name="silero-vad",
            kind="vad",
            ready=ready,
            message=message,
            capabilities=ProviderCapabilities(supports_barge_in=True),
            provider_id="silero",
        )

    async def check_status(self) -> ProviderStatus:
        self._ensure_model()
        return self.status

    async def process_frame(self, frame: AudioFrame) -> list[VADEvent]:
        self._ensure_model()
        if self._model is None or self._torch is None:
            return []
        if frame.sample_rate != self.sample_rate:
            raise ValueError(
                f"Silero VAD expected {self.sample_rate} Hz audio, "
                f"got {frame.sample_rate}"
            )

        self._buffer.extend(frame.data)
        events: list[VADEvent] = []
        window_bytes = self.window_samples * 2
        while len(self._buffer) >= window_bytes:
            chunk_bytes = bytes(self._buffer[:window_bytes])
            del self._buffer[:window_bytes]
            probability = self._infer_probability(chunk_bytes)
            self._audio_ms += int(self.window_samples * 1000 / self.sample_rate)
            transition = self._update_state(probability)
            events.append(VADEvent("vad.probability", probability, self._audio_ms))
            if transition:
                events.append(VADEvent(transition, probability, self._audio_ms))
        return events

    def reset(self) -> None:
        self._buffer.clear()
        self._speaking = False
        self._silence_ms = 0
        self._audio_ms = 0
        if self._model:
            self._model.reset_states()

    async def unload(self) -> ProviderStatus:
        self._model = None
        self._torch = None
        self._buffer.clear()
        self._load_error = None
        return self.status

    def _ensure_model(self) -> None:
        if self._model is not None or self._load_error:
            return
        try:
            from silero_vad import load_silero_vad  # type: ignore[import-not-found]
            import torch  # type: ignore[import-not-found]

            self._torch = torch
            self._model = load_silero_vad(onnx=True)
        except Exception as exc:  # pragma: no cover - import path
            self._load_error = str(exc)

    def _infer_probability(self, chunk_bytes: bytes) -> float:
        assert self._model is not None and self._torch is not None
        samples = struct.unpack(f"<{self.window_samples}h", chunk_bytes)
        tensor = self._torch.tensor(samples, dtype=self._torch.float32) / 32768.0
        result = self._model(tensor, self.sample_rate)
        return round(float(result.item()), 4)

    def _update_state(self, probability: float) -> str | None:
        if probability >= self.threshold:
            self._silence_ms = 0
            if not self._speaking:
                self._speaking = True
                return "vad.speech_start"
            return None

        if self._speaking and probability < self.neg_threshold:
            self._silence_ms += int(self.window_samples * 1000 / self.sample_rate)
            if self._silence_ms >= self.hangover_ms:
                self._speaking = False
                self._silence_ms = 0
                return "vad.speech_end"
        return None
