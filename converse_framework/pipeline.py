"""Turn orchestration for speech-to-speech applications."""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from converse_framework.events import EventSink
from converse_framework.registry import ProviderBundle

logger = logging.getLogger(__name__)

SystemPromptBuilder = Callable[[str, str, list[dict[str, str]]], str]
SamplerBuilder = Callable[[str], dict[str, Any]]


@dataclass(frozen=True)
class PipelineConfig:
    tts_chunk_chars: int = 120
    min_tts_chars: int = 0
    default_mode: str = "chat"


@dataclass
class _TurnState:
    messages: list[dict[str, str]] = field(default_factory=list)
    active_tts_tasks: set[asyncio.Task] = field(default_factory=set)
    system_prompt: str = ""
    turn_id: int = 0
    tts_tail: asyncio.Task | None = None


class SpeechPipeline:
    def __init__(
        self,
        providers: ProviderBundle,
        sink: EventSink,
        config: PipelineConfig | None = None,
        system_prompt_builder: SystemPromptBuilder | None = None,
    ) -> None:
        self.providers = providers
        self.sink = sink
        self.config = config or PipelineConfig()
        self.tts_chunk_chars = self.config.tts_chunk_chars
        self.min_tts_chars = self.config.min_tts_chars
        self._default_mode = self.config.default_mode
        self._system_prompt_builder = system_prompt_builder
        self._states: dict[str, _TurnState] = {self._default_mode: _TurnState()}
        self.state = self._states[self._default_mode]

    def update_turn_config(self, *, tts_chunk_chars: int, min_tts_chars: int) -> None:
        self.tts_chunk_chars = tts_chunk_chars
        self.min_tts_chars = min_tts_chars

    async def clear_conversation(self, mode: str = "chat") -> None:
        self._select_mode(mode)
        await self.cancel_tts("conversation_clear")
        self.state.messages.clear()
        await self.sink.emit("conversation.cleared", mode=self._mode)

    def set_system_prompt(self, prompt: str, mode: str = "chat") -> None:
        self._select_mode(mode)
        self.state.system_prompt = prompt.strip()

    async def cancel_tts(self, reason: str) -> None:
        active = [task for task in self.state.active_tts_tasks if not task.done()]
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        self.state.tts_tail = None
        if active:
            await self.sink.emit("tts.cancelled", reason=reason)

    async def handle_text_turn(self, text: str, mode: str = "chat") -> None:
        self._select_mode(mode)
        turn_state = self.state
        turn_mode = self._mode
        started = time.perf_counter()
        turn_id = self._next_turn_id(turn_state)
        await self.cancel_tts("new_user_turn")
        await self.sink.emit("turn.started", mode=turn_mode, turn_id=turn_id)
        await self.sink.emit("vad.speech_start", mode=turn_mode, source="text")

        final_transcript = ""
        async for transcript in self.providers.asr.transcribe_text_input(text):
            await self.sink.emit(
                "asr.transcript",
                mode=turn_mode,
                text=transcript.text,
                final=transcript.final,
                latency_ms=elapsed_ms(started),
            )
            if transcript.final:
                final_transcript = transcript.text

        await self.sink.emit(
            "vad.speech_end",
            mode=turn_mode,
            source="text",
            latency_ms=elapsed_ms(started),
        )
        if not final_transcript:
            await self.sink.emit("turn.finished", mode=turn_mode, reason="empty_transcript")
            return

        await self._respond_to_transcript(
            final_transcript, started, turn_id, turn_state, turn_mode
        )

    async def handle_audio_turn(
        self, pcm_s16le: bytes, sample_rate: int, mode: str = "chat"
    ) -> None:
        self._select_mode(mode)
        turn_state = self.state
        turn_mode = self._mode
        started = time.perf_counter()
        turn_id = self._next_turn_id(turn_state)
        await self.cancel_tts("new_audio_turn")
        await self.sink.emit("turn.started", mode=turn_mode, source="audio", turn_id=turn_id)
        await self.sink.emit(
            "asr.started", mode=turn_mode, sample_rate=sample_rate, bytes=len(pcm_s16le)
        )

        final_transcript = ""
        try:

            async def progress(event_type: str, payload: dict) -> None:
                await self.sink.emit(event_type, **payload, latency_ms=elapsed_ms(started))

            async for transcript in self.providers.asr.transcribe_audio(
                pcm_s16le, sample_rate, progress
            ):
                await self.sink.emit(
                    "asr.transcript",
                    mode=turn_mode,
                    text=transcript.text,
                    final=transcript.final,
                    latency_ms=elapsed_ms(started),
                )
                if transcript.final:
                    final_transcript = transcript.text
        except Exception as exc:
            await self.sink.emit("asr.error", message=str(exc), latency_ms=elapsed_ms(started))
            await self.sink.emit(
                "turn.finished",
                mode=turn_mode,
                reason="asr_error",
                latency_ms=elapsed_ms(started),
            )
            return

        if not final_transcript:
            await self.sink.emit(
                "turn.finished",
                mode=turn_mode,
                reason="empty_transcript",
                latency_ms=elapsed_ms(started),
            )
            return

        await self._respond_to_transcript(
            final_transcript, started, turn_id, turn_state, turn_mode
        )

    async def handle_continue(self, mode: str = "chat") -> None:
        self._select_mode(mode)
        turn_state = self.state
        turn_mode = self._mode
        if not turn_state.messages or turn_state.messages[-1]["role"] != "assistant":
            await self.sink.emit(
                "turn.error",
                mode=turn_mode,
                message="No previous assistant message to continue.",
            )
            return

        started = time.perf_counter()
        turn_id = self._next_turn_id(turn_state)
        await self.cancel_tts("continue_turn")
        await self.sink.emit(
            "turn.started", mode=turn_mode, source="continue", turn_id=turn_id
        )
        prefix = turn_state.messages[-1]["content"]
        turn_state.messages.pop()
        turn_state.messages.append({"role": "assistant", "content": prefix})

        try:
            response_text = await self._stream_llm_and_tts(
                prefix, started, turn_id, turn_state, turn_mode
            )
            turn_state.messages[-1] = {"role": "assistant", "content": response_text.strip()}
            await self.sink.emit("turn.finished", mode=turn_mode, latency_ms=elapsed_ms(started))
        except Exception as exc:
            await self.sink.emit(
                "turn.error",
                mode=turn_mode,
                message=str(exc),
                latency_ms=elapsed_ms(started),
            )

    def messages_for_mode(self, mode: str) -> list[dict[str, str]]:
        return list(self._state_for_mode(mode).messages)

    async def _respond_to_transcript(
        self,
        final_transcript: str,
        started: float,
        turn_id: int,
        turn_state: _TurnState,
        turn_mode: str,
    ) -> None:
        turn_state.messages.append({"role": "user", "content": final_transcript})
        try:
            response_text = await self._stream_llm_and_tts(
                "", started, turn_id, turn_state, turn_mode
            )
            turn_state.messages.append({"role": "assistant", "content": response_text.strip()})
            await self.sink.emit("turn.finished", mode=turn_mode, latency_ms=elapsed_ms(started))
        except Exception as exc:
            await self.sink.emit(
                "turn.error",
                mode=turn_mode,
                message=str(exc),
                latency_ms=elapsed_ms(started),
            )

    async def _stream_llm_and_tts(
        self,
        response_text: str,
        started: float,
        turn_id: int,
        turn_state: _TurnState,
        turn_mode: str,
    ) -> str:
        first_token_seen = False
        sentence_buffer = ""
        async for token in self.providers.llm.stream_response(
            self._llm_messages(turn_state, turn_mode)
        ):
            if not first_token_seen:
                first_token_seen = True
                await self.sink.emit(
                    "llm.first_token", mode=turn_mode, latency_ms=elapsed_ms(started)
                )
            response_text += token
            sentence_buffer += token
            await self.sink.emit(
                "llm.token", mode=turn_mode, text=token, accumulated=response_text
            )

            if should_flush_tts(
                sentence_buffer, self.tts_chunk_chars, self.min_tts_chars
            ):
                await self._start_tts_chunk(
                    sentence_buffer.strip(), started, turn_id, turn_state, turn_mode
                )
                sentence_buffer = ""

        if sentence_buffer.strip():
            await self._start_tts_chunk(
                sentence_buffer.strip(), started, turn_id, turn_state, turn_mode
            )
        return response_text

    async def _start_tts_chunk(
        self,
        text: str,
        turn_started: float,
        turn_id: int,
        turn_state: _TurnState,
        turn_mode: str,
    ) -> None:
        previous = turn_state.tts_tail
        task = asyncio.create_task(
            self._stream_tts_after(previous, text, turn_started, turn_id, turn_mode)
        )
        turn_state.tts_tail = task
        turn_state.active_tts_tasks.add(task)
        task.add_done_callback(turn_state.active_tts_tasks.discard)

    async def _stream_tts_after(
        self,
        previous: asyncio.Task | None,
        text: str,
        turn_started: float,
        turn_id: int,
        turn_mode: str,
    ) -> None:
        if previous is not None:
            try:
                await previous
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Previous TTS task failed: %s", exc)
        await self._stream_tts(text, turn_started, turn_id, turn_mode)

    async def _stream_tts(
        self, text: str, turn_started: float, turn_id: int, turn_mode: str
    ) -> None:
        first_chunk_seen = False
        chunk_index = 0
        try:

            async def progress(event_type: str, payload: dict) -> None:
                await self.sink.emit(
                    event_type, **payload, latency_ms=elapsed_ms(turn_started)
                )

            async for chunk in self.providers.tts.stream_audio_with_progress(text, progress):
                chunk_index += 1
                if not first_chunk_seen:
                    first_chunk_seen = True
                    await self.sink.emit(
                        "tts.first_chunk",
                        mode=turn_mode,
                        latency_ms=elapsed_ms(turn_started),
                        text=text,
                        turn_id=turn_id,
                    )
                encoded = base64.b64encode(chunk.data).decode("ascii")
                await self.sink.emit(
                    "tts.audio",
                    mode=turn_mode,
                    mime_type=chunk.mime_type,
                    sample_rate=chunk.sample_rate,
                    channels=chunk.channels,
                    encoding=chunk.encoding,
                    duration_ms=chunk.duration_ms,
                    data=encoded,
                    final=chunk.final,
                    text=text,
                    turn_id=turn_id,
                    chunk_index=chunk_index,
                    text_chars=len(text),
                    byte_length=len(chunk.data),
                    latency_ms=elapsed_ms(turn_started),
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.sink.emit(
                "tts.error",
                mode=turn_mode,
                message=str(exc),
                latency_ms=elapsed_ms(turn_started),
                text=text,
            )

    def _llm_messages(self, turn_state: _TurnState, turn_mode: str) -> list[dict[str, str]]:
        prompt = self._effective_system_prompt(turn_state, turn_mode)
        if not prompt:
            return list(turn_state.messages)
        return [{"role": "system", "content": prompt}, *turn_state.messages]

    def _effective_system_prompt(self, turn_state: _TurnState, turn_mode: str) -> str:
        if self._system_prompt_builder is not None:
            return self._system_prompt_builder(
                turn_mode, turn_state.system_prompt, list(turn_state.messages)
            ).strip()
        return turn_state.system_prompt

    def _next_turn_id(self, turn_state: _TurnState | None = None) -> int:
        selected = turn_state or self.state
        selected.turn_id += 1
        return selected.turn_id

    @property
    def _mode(self) -> str:
        for mode, state in self._states.items():
            if state is self.state:
                return mode
        return self._default_mode

    def _select_mode(self, mode: str) -> None:
        self.state = self._state_for_mode(mode)

    def _state_for_mode(self, mode: str) -> _TurnState:
        selected = mode or self._default_mode
        if selected not in self._states:
            self._states[selected] = _TurnState()
        return self._states[selected]


def elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def should_flush_tts(text: str, limit: int, minimum: int = 0) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if len(stripped) >= limit:
        return True
    if len(stripped) < minimum:
        return False
    return stripped.endswith((".", "!", "?", ";", ":"))
