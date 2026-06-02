import asyncio

from converse_framework.events import QueueEventSink
from converse_framework.pipeline import PipelineConfig, SpeechPipeline, should_flush_tts
from converse_framework.protocols import (
    AudioChunk,
    ProviderCapabilities,
    ProviderStatus,
)
from converse_framework.registry import build_provider_bundle


class RecordingLLM:
    def __init__(self, tokens=None):
        self._messages: list[list[dict[str, str]]] = []
        self.tokens = tokens or ["ok."]

    @property
    def messages(self) -> list[list[dict[str, str]]]:
        return self._messages

    @property
    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name="recording-llm",
            kind="llm",
            ready=True,
            message="Recording LLM for tests.",
            capabilities=ProviderCapabilities(),
        )

    async def check_status(self) -> ProviderStatus:
        return self.status

    async def probe_status(self) -> ProviderStatus:
        return self.status

    async def load_status(self) -> ProviderStatus:
        return self.status

    async def stream_response(self, messages: list[dict[str, str]]):
        self._messages.append(messages)
        for token in self.tokens:
            await asyncio.sleep(0)
            yield token


class SlowTTS:
    @property
    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name="slow-tts",
            kind="tts",
            ready=True,
            message="Slow TTS for tests.",
            capabilities=ProviderCapabilities(),
        )

    async def check_status(self) -> ProviderStatus:
        return self.status

    async def probe_status(self) -> ProviderStatus:
        return self.status

    async def load_status(self) -> ProviderStatus:
        return self.status

    async def load(self) -> ProviderStatus:
        return self.status

    async def unload(self) -> ProviderStatus:
        return self.status

    async def stream_audio(self, text: str):
        yield AudioChunk(data=b"x", final=True)  # type: ignore[unreachable]

    async def stream_audio_with_progress(self, text: str, progress=None):
        await asyncio.sleep(10)
        yield AudioChunk(data=b"late", mime_type="audio/wav", final=True)


def mock_bundle():
    return build_provider_bundle(
        {
            "vad": {"provider": "mock"},
            "asr": {"provider": "mock"},
            "llm": {"provider": "mock", "first_token_delay_ms": 0, "token_delay_ms": 0},
            "tts": {"provider": "mock", "first_chunk_delay_ms": 0, "chunk_delay_ms": 0},
        }
    )


def drain(queue):
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


def test_should_flush_tts_on_sentence_or_limit():
    assert should_flush_tts("Hello there.", 120)
    assert should_flush_tts("x" * 121, 120)
    assert not should_flush_tts("still speaking", 120)
    assert not should_flush_tts("Too soon.", 120, 20)
    assert should_flush_tts("This sentence is long enough.", 120, 20)


def test_text_turn_with_mock_providers_emits_core_events():
    async def run_turn():
        queue = asyncio.Queue()
        pipeline = SpeechPipeline(
            mock_bundle(),
            QueueEventSink(queue),
            PipelineConfig(tts_chunk_chars=60),
        )

        await pipeline.handle_text_turn("hello framework")
        await asyncio.sleep(0.05)
        return [event["type"] for event in drain(queue)]

    events = asyncio.run(run_turn())

    assert "turn.started" in events
    assert "asr.transcript" in events
    assert "llm.first_token" in events
    assert "llm.token" in events
    assert "tts.first_chunk" in events
    assert "tts.audio" in events
    assert "turn.finished" in events


def test_text_turn_marks_vad_events_text_only():
    async def run_turn():
        queue = asyncio.Queue()
        pipeline = SpeechPipeline(
            mock_bundle(),
            QueueEventSink(queue),
            PipelineConfig(tts_chunk_chars=60),
        )

        await pipeline.handle_text_turn("hello framework")
        await asyncio.sleep(0.05)
        return drain(queue)

    events = asyncio.run(run_turn())
    vad_events = [
        event
        for event in events
        if event["type"] in {"vad.speech_start", "vad.speech_end"}
    ]
    assert len(vad_events) == 2
    assert all(event["payload"]["source"] == "text" for event in vad_events)
    assert all(event["payload"]["text_only"] is True for event in vad_events)


def test_audio_turn_with_mock_asr_emits_response_events():
    async def run_turn():
        queue = asyncio.Queue()
        pipeline = SpeechPipeline(
            mock_bundle(),
            QueueEventSink(queue),
            PipelineConfig(tts_chunk_chars=60),
        )

        await pipeline.handle_audio_turn(b"\x00\x00" * 1600, 16000)
        await asyncio.sleep(0.05)
        return [event["type"] for event in drain(queue)]

    events = asyncio.run(run_turn())

    assert "asr.started" in events
    assert "asr.transcript" in events
    assert "llm.first_token" in events
    assert "turn.finished" in events


def test_continue_turn_extends_previous_assistant_message():
    async def run_turn():
        queue = asyncio.Queue()
        bundle = mock_bundle()
        bundle.llm = RecordingLLM(tokens=[" more."])
        pipeline = SpeechPipeline(
            bundle, QueueEventSink(queue), PipelineConfig(tts_chunk_chars=60)
        )
        pipeline.state.messages.append({"role": "assistant", "content": "Start"})

        await pipeline.handle_continue()
        await asyncio.sleep(0.05)
        return pipeline.messages_for_mode("chat")

    messages = asyncio.run(run_turn())

    assert messages == [{"role": "assistant", "content": "Start more."}]


def test_tts_chunk_flushing_creates_multiple_audio_events():
    async def run_turn():
        queue = asyncio.Queue()
        bundle = mock_bundle()
        bundle.llm = RecordingLLM(tokens=["One. ", "Two."])
        pipeline = SpeechPipeline(
            bundle, QueueEventSink(queue), PipelineConfig(tts_chunk_chars=60)
        )

        await pipeline.handle_text_turn("hello")
        await asyncio.sleep(0.05)
        return [event for event in drain(queue) if event["type"] == "tts.audio"]

    audio_events = asyncio.run(run_turn())

    assert len(audio_events) == 2
    assert audio_events[0]["payload"]["text"] == "One."
    assert audio_events[1]["payload"]["text"] == "Two."


def test_cancel_tts_cleans_stale_task_and_emits_cancelled():
    async def run_turn():
        queue = asyncio.Queue()
        bundle = mock_bundle()
        bundle.llm = RecordingLLM(tokens=["slow speech."])
        bundle.tts = SlowTTS()
        pipeline = SpeechPipeline(
            bundle, QueueEventSink(queue), PipelineConfig(tts_chunk_chars=60)
        )

        await pipeline.handle_text_turn("hello")
        await asyncio.sleep(0)
        assert pipeline.state.active_tts_tasks

        await pipeline.cancel_tts("barge_in")
        await asyncio.sleep(0)
        return pipeline.state.active_tts_tasks, [
            event["type"] for event in drain(queue)
        ]

    active_tasks, event_types = asyncio.run(run_turn())

    assert active_tasks == set()
    assert "tts.cancelled" in event_types


def test_separate_conversation_histories_for_arbitrary_modes():
    async def run_turns():
        queue = asyncio.Queue()
        pipeline = SpeechPipeline(
            mock_bundle(),
            QueueEventSink(queue),
            PipelineConfig(tts_chunk_chars=60),
        )

        await pipeline.handle_text_turn("chat hello", mode="chat")
        await pipeline.handle_text_turn("custom hello", mode="custom")
        await asyncio.sleep(0.05)
        return pipeline.messages_for_mode("chat"), pipeline.messages_for_mode("custom")

    chat_messages, custom_messages = asyncio.run(run_turns())

    assert chat_messages[0]["content"] == "chat hello"
    assert custom_messages[0]["content"] == "custom hello"


def test_system_prompt_builder_receives_mode_prompt_and_messages():
    async def run_turn():
        queue = asyncio.Queue()
        bundle = mock_bundle()
        bundle.llm = RecordingLLM()

        def build_prompt(mode, manual_prompt, messages):
            assert mode == "companion"
            assert manual_prompt == "manual"
            assert messages == [{"role": "user", "content": "hello"}]
            return "built prompt"

        pipeline = SpeechPipeline(
            bundle,
            QueueEventSink(queue),
            PipelineConfig(tts_chunk_chars=60),
            system_prompt_builder=build_prompt,
        )
        pipeline.set_system_prompt("manual", mode="companion")
        await pipeline.handle_text_turn("hello", mode="companion")
        await asyncio.sleep(0.05)
        return bundle.llm.messages[0]

    messages = asyncio.run(run_turn())

    assert messages[0] == {"role": "system", "content": "built prompt"}
    assert messages[1] == {"role": "user", "content": "hello"}


# ---------------------------------------------------------------------------
# Phase 1.2: exception_payload and richer error events
# ---------------------------------------------------------------------------


def test_exception_payload_with_non_empty_message():
    from converse_framework.pipeline import exception_payload

    exc = ValueError("something broke")
    result = exception_payload(exc, fallback="default")
    assert result["message"] == "something broke"
    assert result["error_type"] == "ValueError"


def test_exception_payload_uses_fallback_for_empty_message():
    from converse_framework.pipeline import exception_payload

    exc = AssertionError()
    result = exception_payload(exc, fallback="ASR provider failed with AssertionError.")
    assert result["message"] == "ASR provider failed with AssertionError."
    assert result["error_type"] == "AssertionError"


def test_exception_payload_includes_repr():
    from converse_framework.pipeline import exception_payload

    exc = RuntimeError()
    result = exception_payload(exc, fallback="TTS provider failed with RuntimeError.")
    assert result["message"] == "TTS provider failed with RuntimeError."
    assert result["error_type"] == "RuntimeError"
    assert "repr" in result


class FailingASR:
    """ASR provider that always raises AssertionError()."""

    @property
    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name="failing-asr",
            kind="asr",
            ready=False,
            message="Failing ASR for tests.",
            capabilities=ProviderCapabilities(),
        )

    async def check_status(self) -> ProviderStatus:
        return self.status

    async def probe_status(self) -> ProviderStatus:
        return self.status

    async def load_status(self) -> ProviderStatus:
        return self.status

    async def load(self) -> ProviderStatus:
        return self.status

    async def unload(self) -> ProviderStatus:
        return self.status

    async def transcribe_text_input(self, text: str):
        raise AssertionError()
        yield  # type: ignore[unreachable]  # noqa

    async def transcribe_audio(self, pcm_s16le: bytes, sample_rate: int, progress=None):
        raise AssertionError()
        yield  # type: ignore[unreachable]  # noqa


class FailingTTS:
    """TTS provider that always raises RuntimeError()."""

    @property
    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name="failing-tts",
            kind="tts",
            ready=False,
            message="Failing TTS for tests.",
            capabilities=ProviderCapabilities(),
        )

    async def check_status(self) -> ProviderStatus:
        return self.status

    async def probe_status(self) -> ProviderStatus:
        return self.status

    async def load_status(self) -> ProviderStatus:
        return self.status

    async def load(self) -> ProviderStatus:
        return self.status

    async def unload(self) -> ProviderStatus:
        return self.status

    async def stream_audio(self, text: str):
        raise RuntimeError()
        yield  # type: ignore[unreachable]  # noqa

    async def stream_audio_with_progress(self, text: str, progress=None):
        raise RuntimeError()
        yield  # type: ignore[unreachable]  # noqa


def test_audio_turn_asr_error_includes_message_and_error_type():
    async def run_turn():
        queue = asyncio.Queue()
        bundle = mock_bundle()
        bundle.asr = FailingASR()
        pipeline = SpeechPipeline(bundle, QueueEventSink(queue), PipelineConfig())

        await pipeline.handle_audio_turn(b"\x00\x00" * 1600, 16000)
        await asyncio.sleep(0.05)
        return drain(queue)

    events = asyncio.run(run_turn())
    asr_errors = [e for e in events if e["type"] == "asr.error"]
    assert len(asr_errors) == 1
    payload = asr_errors[0]["payload"]
    assert payload["message"]  # non-empty
    assert payload["error_type"] == "AssertionError"


def test_text_turn_tts_error_includes_message_and_error_type():
    async def run_turn():
        queue = asyncio.Queue()
        bundle = mock_bundle()
        bundle.tts = FailingTTS()
        pipeline = SpeechPipeline(
            bundle, QueueEventSink(queue), PipelineConfig(tts_chunk_chars=60)
        )

        await pipeline.handle_text_turn("hello")
        await asyncio.sleep(0.05)
        return drain(queue)

    events = asyncio.run(run_turn())
    tts_errors = [e for e in events if e["type"] == "tts.error"]
    assert len(tts_errors) >= 1
    payload = tts_errors[0]["payload"]
    assert payload["message"]  # non-empty
    assert payload["error_type"] == "RuntimeError"
