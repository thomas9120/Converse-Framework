# Converse-Framework feedback (Test-run-plan build)

Findings from building the Converse Companion web app
(2026-06-02). The app is a FastAPI + WebSocket consumer that drives
the framework's `SpeechPipeline` and `AudioUtteranceCollector`
end-to-end through a browser mic + TTS playback. The
[`README.md`](./README.md) has the architecture / usage; this file
is only the framework-level feedback.

## What worked well

### Public API was enough to build the whole app
The framework's public surface (`SpeechPipeline`,
`AudioUtteranceCollector`, `build_provider_bundle`, `status_only`,
`parse_audio_frame`, `ProviderBundle.statuses`, `FrameworkEvent`,
`ExtraHintFor`) was sufficient to wire every layer of a real browser
consumer without dropping into the framework's internals. The
boundaries from `plan.md` and `MIGRATION.md` held: app policy lives
in the app, provider mechanics live in the framework, and the
`Transport` protocol was easy to implement against a FastAPI
WebSocket.

### Lazy registry made the UI's status screen trivial
`status_only(config)` produced a four-item list (VAD, ASR, LLM, TTS)
that maps 1:1 onto the four rows in the settings drawer. Each row
shows the provider's `ready` flag and a human-readable `message`
that already names the missing `pip install` extra when the
provider's heavy dep is absent. This is the single biggest reason
the settings UI didn't need its own diagnostics layer.

### `parse_audio_frame` wire contract is solid
The browser-side `pcm-worklet.js` produces base64 PCM s16le frames
with the exact shape the Python side expects, and a malformed
payload (wrong sample rate, wrong channels, wrong frame size, bad
base64, etc.) raises a `ValueError` with a specific message. The
runtime catches the `ValueError` and forwards it to the client as
`audio.frame_error` so the browser can drop the offending frame.
This contract was trivial to test from a `python -c` script and
the failure modes are informative.

### `TtsAudioPlayer` reference is drop-in usable
The reference JS client at `converse_framework/js/tts-audio-player.js`
worked unchanged inside our vanilla-JS / no-bundler setup. The
coalescing logic is what kept Pocket TTS from sounding choppy when
the test app actually ran the full stack end-to-end.

### `extra_hint_for` is a one-liner that solved two problems
The companion's settings drawer shows `pip install
converse-framework[silero]`-style messages, and the
`start-cloudflare.ps1` script's first-run guidance uses the same
helper to tell the user which provider is missing. One helper, two
UI surfaces.

### `LlamaCppProvider` OpenAI-compat is dead simple
The OpenAI-compatible endpoint contract means the test app can swap
in any other provider (Ollama, vLLM, LM Studio, ...) just by
changing `llm.base_url` in settings. No code changes, no restart
needed beyond the runtime's lazy rebuild.

## Suggestions for the framework

### 1. `PipelineConfig` exposes `tts_chunk_chars` and `min_tts_chars` but no default-mode knob
The `default_mode` field exists but the pipeline keeps separate
state per mode key. A consumer that wants to surface "chat" /
"companion" / "voice-clone" tabs needs to manage that state
externally (or always pass `mode=...` explicitly). A `modes` helper
that lists active mode keys or a `clear_all_conversations()`
method would make multi-mode UIs less manual.

## Summary

The framework is ready to be used as a third-party dependency. The
public API is small, the wire shape is stable, and the
framework/app boundary is clean enough to build a real consumer
without touching the framework code. The feedback above is all
about smoothing rough edges that the test app hit on the way to a
working product — none of it blocks the build.

## Completed follow-up

Implemented on 2026-06-02:

- Added structured unavailable-provider fields:
  `ProviderStatus.install_hint` and `ProviderStatus.missing_extra`.
- Added `TransportEventSink`, a generic `EventSink` adapter for any
  `Transport`.
- Added collector config persistence/update helpers:
  `UtteranceCollectorConfig.to_dict()`,
  `AudioUtteranceCollector.serialize_config()`, and
  `AudioUtteranceCollector.update_config(...)`.
- Marked text-path VAD events with `text_only: true` while preserving
  `source: "text"`.
- Documented the `parse_audio_frame` consumer convention for
  `audio.frame_error`.
- Added `converse_framework.examples.websocket_voice_chat`, a small
  optional-FastAPI WebSocket voice-chat recipe.
- Added `PocketTTSProvider.set_quantize(bool)` so apps can switch
  quantization mode without rebuilding the provider bundle.
