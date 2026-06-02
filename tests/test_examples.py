"""Tests for the second-consumer example in ``converse_framework.examples``.

Phase 6 requires a CLI/example text conversation that uses the
framework **without importing the harness**. These tests cover the
core :func:`run_text_chat` driver with mock providers and the CLI
parser's provider-override handling.
"""

from __future__ import annotations

import asyncio

import pytest

from converse_framework.examples import text_chat
from converse_framework.examples.text_chat import (
    TextChatExampleConfig,
    _format_event_for_cli,
    _parse_provider_args,
    build_example_bundle,
    build_example_pipeline,
    run_text_chat,
)


# ---------------------------------------------------------------------------
# run_text_chat
# ---------------------------------------------------------------------------


def test_run_text_chat_uses_only_framework_imports():
    """The example must not pull in any harness module.

    Success criterion #7 of the plan: a second consumer can run a text
    conversation using the framework without importing the harness.
    """
    import converse_framework.examples.text_chat as module

    module_path = module.__file__ or ""
    assert "conversational_harness" not in module_path
    # And the module's namespace must not carry any harness symbols.
    assert not any(name.startswith("conversational_") for name in dir(module))


def test_run_text_chat_with_mock_providers_emits_full_event_stream():
    async def run():
        return await run_text_chat(
            ["hello framework"],
            TextChatExampleConfig(
                tts_chunk_chars=60,
                min_tts_chars=0,
                system_prompt="You are concise.",
            ),
        )

    summary = asyncio.run(run())

    assert summary["mode"] == "chat"
    assert len(summary["turns"]) == 1
    turn = summary["turns"][0]
    # The same event types the harness browser UI relies on.
    for event_type in (
        "turn.started",
        "asr.transcript",
        "llm.first_token",
        "llm.token",
        "tts.first_chunk",
        "tts.audio",
        "turn.finished",
    ):
        assert event_type in turn["events"], f"missing event {event_type!r}"

    # Mock LLM echoes the input back, so we expect to see the user's
    # text somewhere in the joined LLM tokens.
    assert "hello framework" in turn["llm_text"].lower()
    # At least one TTS audio chunk is produced for the response.
    assert turn["tts_audio_chunks"] >= 1

    # The pipeline recorded the user message in the chat history.
    roles = [message["role"] for message in summary["messages"]]
    assert "user" in roles
    assert "assistant" in roles


def test_run_text_chat_keeps_separate_history_per_mode():
    async def run():
        config = TextChatExampleConfig(mode="custom")
        return await run_text_chat(["first turn"], config)

    summary = asyncio.run(run())

    assert summary["mode"] == "custom"
    assert summary["messages"][0]["content"] == "first turn"


def test_run_text_chat_with_custom_provider_overrides():
    """Custom provider names are forwarded to ``build_provider_bundle``.

    The test only checks the bundle wiring; the providers themselves
    may be ``UnavailableProvider`` when their extras are not installed
    in the test environment, but the bundle should still build cleanly.
    """

    config = TextChatExampleConfig(
        providers={"vad": "mock", "asr": "mock", "llm": "mock", "tts": "mock"},
    )
    bundle = build_example_bundle(config)
    statuses = bundle.statuses()
    # One entry per kind in declaration order.
    assert [item["kind"] for item in statuses] == ["vad", "asr", "llm", "tts"]


# ---------------------------------------------------------------------------
# Pipeline + bundle builders
# ---------------------------------------------------------------------------


def test_build_example_pipeline_uses_supplied_sink_and_system_prompt():
    async def run():
        from converse_framework.events import QueueEventSink

        queue: asyncio.Queue = asyncio.Queue()
        sink = QueueEventSink(queue)
        config = TextChatExampleConfig(system_prompt="Speak like a pirate.")
        pipeline, bundle = build_example_pipeline(config, sink=sink)

        assert pipeline.state.system_prompt == "Speak like a pirate."
        assert pipeline.sink is sink
        return bundle.statuses()

    statuses = asyncio.run(run())
    # Every kind should be present; mock providers are always ready.
    assert all(item["ready"] for item in statuses)


# ---------------------------------------------------------------------------
# CLI parser helpers
# ---------------------------------------------------------------------------


def test_parse_provider_args_keeps_vad_mock_default():
    parsed = _parse_provider_args([])
    assert parsed == {"vad": "mock", "asr": "mock", "llm": "mock", "tts": "mock"}


def test_parse_provider_args_overrides_kinds():
    parsed = _parse_provider_args(
        ["asr=faster-whisper", "llm=llamacpp", "tts=kokoro"]
    )
    assert parsed["asr"] == "faster-whisper"
    assert parsed["llm"] == "llamacpp"
    assert parsed["tts"] == "kokoro"
    # VAD still defaults to mock in the text-only example.
    assert parsed["vad"] == "mock"


def test_parse_provider_args_rejects_unknown_kind():
    with pytest.raises(SystemExit):
        _parse_provider_args(["weird=faster-whisper"])


def test_parse_provider_args_rejects_empty_name():
    with pytest.raises(SystemExit):
        _parse_provider_args(["asr="])


# ---------------------------------------------------------------------------
# Event formatting
# ---------------------------------------------------------------------------


def test_format_event_for_cli_dispatches_per_event_type():
    formatted = _format_event_for_cli(
        {"type": "llm.token", "payload": {"text": "hi"}}
    )
    assert "hi" in formatted

    formatted = _format_event_for_cli(
        {"type": "asr.transcript", "payload": {"text": "hello", "final": True}}
    )
    assert "asr(final)" in formatted
    assert "hello" in formatted

    formatted = _format_event_for_cli(
        {"type": "tts.audio", "payload": {"data": "", "mime_type": "audio/wav"}}
    )
    assert "audio/wav" in formatted

    formatted = _format_event_for_cli({"type": "turn.finished", "payload": {}})
    assert "turn.finished" in formatted

    formatted = _format_event_for_cli(
        {"type": "turn.error", "payload": {"message": "boom"}}
    )
    assert "boom" in formatted
