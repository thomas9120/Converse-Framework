# Changelog

## Unreleased

- **OpenAI-compatible providers (LLM, ASR, TTS)** — new
  `openai-compatible` provider name (extra: `openai-compat`) registered
  for all three inference kinds. LLM covers any
  `/v1/chat/completions` + `/v1/models` server (Ollama, LM Studio,
  vLLM, Groq, OpenRouter, Together, OpenAI itself) and shares its
  implementation with the `llamacpp` provider but skips the
  llama.cpp-specific `/health` probe. ASR uploads a multipart WAV to
  `/v1/audio/transcriptions` (OpenAI Whisper, Groq hosted Whisper,
  `speaches` / faster-whisper-server). TTS requests WAV from
  `/v1/audio/speech` and yields a single final PCM chunk (OpenAI TTS,
  Kokoro-FastAPI, openedai-speech).
- **`api_key` support** — both `llamacpp` and `openai-compatible` LLM
  providers accept an `api_key` config option, sent as an
  `Authorization: Bearer` header (matches llama.cpp's `--api-key`).
- **Eager first TTS chunk** — new `PipelineConfig.first_chunk_chars`
  (default `40`) applies a lower flush threshold to the *first* TTS
  chunk of each turn and also flushes on a comma, so the voice starts
  as soon as the opening clause is available. Subsequent chunks use
  the normal `tts_chunk_chars` thresholds. Set to `0` to restore the
  previous single-threshold behaviour.
  `SpeechPipeline.update_turn_config` accepts an optional
  `first_chunk_chars` argument.
- **Persistent LLM connection** — `stream_response` reuses one
  `httpx.AsyncClient` across turns instead of reconnecting per turn;
  the client is closed by `unload()`.
- **Sampler merge fix** — sampler-provider overrides are now merged over
  the constructor defaults instead of replacing them, so returning only
  e.g. `{"top_p": 0.9}` no longer drops `temperature` / `max_tokens`.
- Fixed dead `import httpx` checks in the llamacpp and whisper-cpp
  `probe_status` methods.

## v0.2.0 — Provider lifecycle, session helper, and browser mic support

### Highlights

- **Provider lifecycle events** — `provider.loading`, `provider.loaded`,
  `provider.error` events with a consistent cross-provider shape.
- **Provider status tiers** — `status` (cached property), `probe_status()`
  (no I/O, no model load), `load_status()` (may initialise resources).
- **Runtime provider swap** — `ProviderBundle.replace()`,
  `SpeechPipeline.update_providers()`,
  `AudioUtteranceCollector.update_vad_provider()` for safe in-flight
  provider replacement.
- **Provider configuration** — `TTSProvider.configure()` and
  `PocketTTSProvider.list_voices()` for first-class voice changes
  without replacing the provider instance.
- **Reusable WebSocket session** — `WebSocketSession` at
  `converse_framework.session` handles 7 built-in message types,
  freeing apps from copying the recipe state machine.
- **Browser mic capture** — `mic-frame-sender.js` using `AudioWorklet`,
  `speaker-echo-guard.js` for echo-aware frame gating, and
  `browser-voice-client.js` combining mic, TTS playback, and echo
  guard into one class.
- **CUDA DLL discovery** — `cuda_utils` helper for Windows NVIDIA wheel
  DLL path resolution, auto-integrated into `FasterWhisperASRProvider`.
- **Richer error payloads** — `FrameworkEvent` payloads always include
  `message` and `exception` fields for `turn.error` and
  `provider.error` events.
- **Faster Whisper lazy load** — model loads on first audio turn, not
  at instantiation, fixing stale-file assertion failures.

### Breaking changes

- `ProviderStatus` now carries optional `status_level` (`str`),
  `backend` (`str | None`), and `voices` (`list[VoiceInfo] | None`).
  Old field access (`status.ready`, `status.message`) is unchanged.
- `PocketTTSProvider.set_quantize()` is deprecated in favour of
  `configure(quantize=...)`. The old method is kept for backward
  compatibility in this release.

### New dependencies

None (all optional extras unchanged).

### Migration notes

- `check_status()` is kept for backward compatibility. New code should
  prefer `probe_status()` for lightweight checks and `load_status()`
  when models must be confirmed loaded.
- `ProviderBundle.check_statuses()` is kept as an alias for probe
  semantics.
- `build_websocket_voice_runtime()` and `handle_websocket_message()`
  in the WebSocket example remain importable. New apps should prefer
  `WebSocketSession` from `converse_framework.session`.
- See `MIGRATION.md` for the full v0.1 → v0.2 transition guide.

### What's next

- Refining the `WebSocketSession` hook API based on real-world usage.
- Extending `configure()` to ASR and LLM providers.
- Browser automation tests for the JS helpers.
- Formalising the provider event schema as typed subclasses.
