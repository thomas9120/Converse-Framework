# Converse Framework Improvement Plan

This plan turns the feedback in `Converse-Framework-feedback.md` into a staged set of framework improvements. The goal is to reduce repeated app glue while keeping the v0.1 boundary intact: the framework should own provider-agnostic speech/session mechanics, while apps still own profile persistence, product UI, HTTP routing, and deployment choices.

## Current Baseline

- Core package: `converse_framework/`
- WebSocket recipe: `converse_framework/examples/websocket_voice_chat.py`
- Browser playback helper: `converse_framework/js/tts-audio-player.js`
- Provider contracts: `converse_framework/protocols.py`
- Provider bundle/build/status helpers: `converse_framework/registry.py`
- Turn orchestration: `converse_framework/pipeline.py`
- VAD frame collection: `converse_framework/utterance_collector.py`
- Main docs: `README.md`, `MIGRATION.md`
- Test anchors: `tests/test_examples.py`, `tests/test_pipeline.py`, `tests/test_providers.py`, `tests/test_registry.py`, `tests/test_protocols.py`, `tests/test_transport.py`

## Design Principles

1. Preserve lightweight base imports. New helpers must not import FastAPI, browser-only dependencies, or heavy model libraries at `import converse_framework` time.
2. Prefer additive public APIs for v0.1 compatibility. Existing `status`, `check_status`, `ProviderBundle.check_statuses()`, and recipe functions should continue to work.
3. Separate cheap diagnostics from heavyweight load paths. App startup checks should be predictable and should not accidentally download or initialize models.
4. Standardize event payloads without removing existing event types. Add richer fields and provider lifecycle events; keep current `asr.progress`, `tts.progress`, `asr.error`, and `tts.error` consumers working.
5. Keep app policy injectable. Session helpers should provide hooks for settings, status, provider rebuilds, and custom message types without knowing profile formats or UI concepts.

## Phase 1: Error and Lazy-Load Reliability ✅

### 1.1 Fix faster-whisper first-use lazy loading ✅

Problem: `FasterWhisperASRProvider.transcribe_audio()` queues progress, then `_transcribe_blocking()` asserts `_model is not None` instead of loading it. Empty assertion messages become poor UI errors.

Implementation:

- [x] Update `converse_framework/providers/faster_whisper.py`.
- [x] In `_transcribe_blocking()`, call `self._ensure_model()` before asserting or using `_model`.
- [x] Emit progress in this order:
  - `asr.progress` stage `queued` from the async path.
  - `asr.progress` stage `loading` before `_ensure_model()`.
  - `asr.progress` stage `loaded` after `_ensure_model()`.
  - `asr.progress` stage `segment` and `complete` as today.
- [x] Remove the bare `assert self._model is not None` as the primary guard. If a model is still absent after `_ensure_model()`, raise `RuntimeError("faster-whisper model did not load")`.
- [x] Preserve `timeout_s` behavior by keeping the entire blocking load/transcribe work inside `asyncio.wait_for(...)`.

Tests:

- [x] Add a fake `WhisperModel` module/class test in `tests/test_providers.py` proving `transcribe_audio()` calls `_ensure_model()` on first use.
- [x] Add a test that injected `_model` still skips import/load.
- [x] Add a test for load failure that asserts `_load_error` is set and an actionable exception message is emitted.

### 1.2 Standardize exception serialization in pipeline events ✅

Problem: `SpeechPipeline.handle_audio_turn()` emits `{"message": str(exc)}` for `asr.error`; empty-message exceptions produce unhelpful UI output. `_stream_tts()` has the same pattern.

Implementation:

- [x] Add a small helper in `converse_framework/pipeline.py`: `exception_payload(exc, *, fallback: str) -> dict[str, str]`
  - Fields: `message`, `error_type`, `repr`.
- [x] Use it for:
  - `asr.error`
  - `tts.error`
  - `turn.error` in `_respond_to_transcript()`
  - `turn.error` in `handle_continue()`
- [x] Fallback message examples:
  - ASR: `"ASR provider failed with AssertionError."`
  - TTS: `"TTS provider failed with RuntimeError."`

Tests:

- [x] Add `tests/test_pipeline.py` coverage with a provider that raises `AssertionError()` and assert `message` is non-empty and `error_type == "AssertionError"`.
- [x] Add matching TTS error coverage.

## Phase 2: Provider Status Semantics and Lifecycle Events ✅

### 2.1 Split cheap probe status from heavyweight load status ✅

Problem: `ProviderBundle.check_statuses()` is convenient, but provider `check_status()` semantics are uneven. Some checks import dependencies or make HTTP calls; future implementations may load models.

Implementation:

- [x] Extend provider protocols in `converse_framework/protocols.py` with additive optional methods:
  - `async def probe_status(self) -> ProviderStatus`
  - `async def load_status(self) -> ProviderStatus`
- [x] Keep `check_status()` as compatibility alias for now.
- [x] Define semantics in docstrings:
  - `status`: current cached state, no I/O.
  - `probe_status()`: cheap readiness/import/network probe, no model load or download.
  - `load_status()`: may load or initialize heavy resources.
  - `check_status()`: deprecated compatibility method; should call `probe_status()` unless the provider already had stronger documented behavior.
- [x] Implement `probe_status()` and `load_status()` for built-in providers:
  - Mock providers: both return `status`.
  - `UnavailableProvider`: both return `status`.
  - `FasterWhisperASRProvider`: `probe_status()` imports `faster_whisper`; `load_status()` calls `load()`.
  - `PocketTTSProvider`: `probe_status()` imports `pocket_tts`; `load_status()` calls `load()`.
  - `SileroVADProvider`: `probe_status()` imports/checks availability; `load_status()` loads if supported.
  - HTTP providers (`LlamaCppProvider`, `WhisperCppASRProvider`): `probe_status()` checks httpx import; `load_status()` aliases to `probe_status()`.
  - `KokoroOnnxProvider`: mirror Pocket TTS import vs load semantics.
- [x] Add `ProviderBundle.probe_statuses()` and `ProviderBundle.load_statuses()`.
- [x] Keep `ProviderBundle.check_statuses()` for backward compatibility.
- [x] Update `status_only()` to use `probe_status()`.

Tests:

- [x] Extend `tests/test_registry.py` counting providers to prove `status_only()` and `probe_statuses()` do not call `load()`.
- [x] Add tests that `load_statuses()` does call `load()` for providers that expose it.
- [x] Update status-shape tests to include new serialized fields introduced below.

### 2.2 Add structured provider metadata to `ProviderStatus` ✅

Problem: settings UIs need voice lists and clearer management metadata, but provider-specific methods are currently ad hoc.

Implementation:

- [x] Add optional fields to `ProviderStatus` in `converse_framework/protocols.py`:
  - `voices: tuple[dict[str, str], ...] = ()`
  - `active_voice: str | None = None`
  - `models: tuple[dict[str, str], ...] = ()`
  - `active_model: str | None = None`
  - `status_level: str = "ready"` | `"configured"` | `"loading"` | `"error"` | `"unavailable"`
- [x] Update `_serialize_status()` in `converse_framework/registry.py`.
- [x] Populate obvious values:
  - Pocket TTS: `active_voice=self.voice`, `voices` with known voice list.
  - Kokoro: `active_voice=self.voice`.
  - Faster Whisper: `active_model=self.model_name`, `models` tuple.
  - Whisper CPP: `active_model=self.model`, `models` tuple.
  - Mock providers: empty metadata (defaults).

Tests:

- [x] Update `tests/test_protocols.py`, `tests/test_registry.py`, and `tests/test_providers.py` to assert the new keys serialize consistently.

### 2.3 Standardize provider lifecycle events ✅

Problem: provider setup failures arrive as generic stage events plus later errors. UIs need a consistent shape across VAD, ASR, LLM, and TTS.

Implementation:

- [x] Add helper functions in `converse_framework/provider_events.py`.
- [x] Standard event types:
  - `provider.loading`
  - `provider.loaded`
  - `provider.error`
- [x] Payload shape:
  - `kind`: `"vad" | "asr" | "llm" | "tts"`
  - `provider`: provider name or `ProviderStatus.name`
  - `stage`
  - `message`
  - `error_type` for failures
  - `loaded`
  - `latency_ms` when available
- [x] For Phase 2, emit these from:
  - Progress callback in `SpeechPipeline.handle_audio_turn()`: emits `provider.loading` and `provider.loaded` alongside existing `asr.progress`/`tts.progress`.
  - `SpeechPipeline` when catching ASR/TTS/LLM errors: emits `provider.error` alongside existing `asr.error`/`tts.error`/`turn.error`.
- [x] Keep existing `asr.progress` and `tts.progress` events for compatibility.

Tests:

- [x] Add tests for `provider_loading_event()`, `provider_loaded_event()`, `provider_error_event()` shape.
- [x] Verify failure payloads include `kind`, `provider`, `message`, and `error_type`.

## Phase 3: Runtime Provider Reload and Configuration Pattern ✅

### 3.1 Add public provider bundle update helpers ✅

Problem: apps currently rebuild a bundle and recreate the collector because the pipeline owns the provider bundle and the collector owns the VAD provider.

Implementation:

- [x] Add an immutable-ish update API to `ProviderBundle` in `converse_framework/registry.py`:
  - `replace(**providers) -> ProviderBundle`
  - `async unload_replaced(old_bundle, new_bundle) -> list[ProviderStatus]`
- [x] Add `SpeechPipeline.update_providers(providers: ProviderBundle, *, cancel_active_tts: bool = True, reason: str = "provider_reload")`.
  - Cancels active TTS by default.
  - Swaps `self.providers`.
  - Emits `providers.updated` with serialized statuses.
  - Does not clear conversation history.
  - Unloads replaced providers in background via `ensure_future`.
- [x] Add `AudioUtteranceCollector.update_vad_provider(vad_provider: VADProvider)`.
  - Rejects while recording.
  - Clears pre-speech buffer on swap.

Tests:

- [x] Pipeline test proving `update_providers()` swaps TTS and emits `providers.updated`.
- [x] Pipeline test proving `update_providers()` keeps conversation history.
- [x] Collector test proving `update_vad_provider()` swaps when idle.
- [x] Collector test proving `update_vad_provider()` rejects while recording.
- [x] Collector test `update_vad_provider()` clears pre-buffer.
- [x] Registry tests for `ProviderBundle.replace()` (single, multiple, no args).
- [x] Registry tests for `ProviderBundle.unload_replaced()` (replaced-only, identical bundles).

### 3.2 Document runtime reload recipes ✅

Implementation:

- [x] Add README section "Runtime Provider Updates".
- [x] Include recipes for:
  - `ProviderBundle.replace()` and `unload_replaced()`.
  - `SpeechPipeline.update_providers()`.
  - `AudioUtteranceCollector.update_vad_provider()`.
  - End-to-end settings-update flow.
- [x] Update `MIGRATION.md` boundary section: framework now owns safe swap mechanics.
- [x] Add `MIGRATION.md` section on `probe_status()` vs `load_status()`.

## Phase 4: First-Class TTS Voice Configuration ✅

### 4.1 Add optional provider configuration protocol ✅

Implementation:

- [x] Add dataclasses in `converse_framework/protocols.py`:
  - `VoiceInfo` with `id`, `label`, `language`, `description`, `gender`.
  - `ProviderConfigResult` with `status: ProviderStatus`, `changed: bool`, `requires_reload: bool`, `message: str`.
- [x] Add optional protocol methods to `TTSProvider`:
  - `async def configure(self, **options) -> ProviderConfigResult` with default no-op return.
  - `def list_voices(self) -> tuple[VoiceInfo, ...]` with default empty return.
- [x] Export `VoiceInfo` and `ProviderConfigResult` from `converse_framework.__init__`.

### 4.2 Implement Pocket TTS voice and config support ✅

Implementation:

- [x] Add `set_voice(voice: str) -> ProviderStatus`:
  - Same voice → keep loaded state.
  - Different voice → clear `_voice_state`, keep `_model`.
- [x] Add `configure(**options)` supporting `voice`, `quantize`, `language`, `temp`, `max_tokens`, `coalesce_ms`.
  - `voice` change → reload voice state only.
  - `quantize`, `language`, `temp` → unload model and voice state.
  - `max_tokens`, `coalesce_ms` → no unload.
- [x] Add `list_voices()` returning structured `VoiceInfo` from static known list.
- [x] `voices` and `active_voice` already populated from Phase 2.

Tests:

- [x] `set_voice()` clears only `_voice_state`.
- [x] `set_voice()` to same voice keeps both model and voice state.
- [x] `configure(max_tokens=...)` does not unload.
- [x] `configure(coalesce_ms=...)` does not unload.
- [x] `configure(voice=...)` clears voice state only.
- [x] `configure(quantize=...)` clears model and voice state.
- [x] `configure(temp=...)` clears model and voice state.
- [x] `configure()` with unchanged values returns `changed=False`.
- [x] `list_voices()` returns structured `VoiceInfo` objects.

## Phase 5: Reusable WebSocket Session Helper ✅

### 5.1 Add framework-owned session module ✅

Implementation:

- [x] Create `converse_framework/session.py`.
- [x] Transport-generic — depends only on framework `Transport` protocol, not FastAPI.
- [x] `WebSocketSessionConfig` with `provider_config`, `collector_config`, `pipeline_config`, `default_mode`, `auto_probe_status`.
- [x] `WebSocketSessionHooks` with `on_unknown_message`, `on_settings_update`, `on_status_request`, `on_before_provider_reload`, `on_after_provider_reload`, `on_event`.
- [x] `WebSocketSession` owning transport, sink, bundle, pipeline, collector, frame_stats.
- [x] `handle_message(message)` routes 7 built-in types: `audio.frame`, `text.turn`, `conversation.clear`, `tts.cancel`, `status.request`, `settings.update`, `providers.reload`.
- [x] Unknown messages call hook, fall back to `turn.error`.
- [x] `reload_providers(config, load=False)` — swaps bundle + VAD provider.
- [x] `emit_status(kind="probe")` — probe/check/load status dispatch.
- [x] No FastAPI import at module or construction time.

Tests:

- [x] `tests/test_session.py` — 24 tests covering:
  - Construction without FastAPI (sys.modules assert).
  - Default/custom config and hooks.
  - Unknown message→turn.error and hook routing.
  - `text.turn`, `conversation.clear`, `tts.cancel`, `audio.frame_error`.
  - `status.request` probe/check kinds + hook.
  - `settings.update` hook and silent-default.
  - `on_event` hook fires per event.
  - `providers.reload` swaps bundle, `load=True` no crash.
  - `before`/`after` reload hooks fire in order.
  - `emit_status()` probe/check/load.
  - `reload_providers()` public method.

### 5.2 Public exports ✅

Implementation:

- [x] Documented as `converse_framework.session` import path (not top-level export to keep lightweight imports).

Docs:

- [x] README will be updated in Phase 9 to reflect the session helper boundary.

## Phase 6: Browser Microphone and Frame Sender Helper ✅

### 6.1 Mic frame sender ✅

Implementation:

- [x] `converse_framework/js/mic-frame-sender.js`.
- [x] No npm/bundler dependency; vanilla script.
- [x] `MicFrameSender` class with full constructor options (`webSocket`, `sampleRate`, `channels`, `frameMs`, `mode`, `messageType`, `onLevel`, `onError`, `audioContext`, `shouldSendFrame`).
- [x] Methods: `start()`, `stop()`, `setMode()`, `setWebSocket()`, `close()`.
- [x] `getUserMedia` → `AudioWorkletNode` (via inline blob URL processor) → fallback `ScriptProcessorNode`.
- [x] Downsample with linear interpolation.
- [x] Float32 → PCM s16le with clamping.
- [x] Slice exact `frameMs` frames.
- [x] Framework-compatible payload (`audio.frame` with base64 data, sample_rate, channels, frame_ms, sequence, encoding, mode).
- [x] Optional `shouldSendFrame` gate for echo guard integration.
- [x] Level reporting via `onLevel` callback.
- [x] Docs: HTTPS/localhost/tunnel requirement for mobile.
- [x] UMD wrapper (AMD/CommonJS/global).

Tests:

- [x] `tests/js/test_helpers.mjs` — 25 Node-compatible tests:
  - `downsampleFloat32` (same rate, 2× down, up).
  - `float32ToPcmS16le` (zero, positive, negative, clamping, byte length).
  - `arrayBufferToBase64` (empty, known, null bytes).
  - `msToSamples` (1s/30ms/100ms/0ms).
- [x] `tests/js/manual-smoke-test.html` — full browser test page with mic, player, guard, status, log, level meter, and all controls.

### 6.2 Combined browser client ✅

Implementation:

- [x] `converse_framework/js/browser-voice-client.js`.
- [x] `BrowserVoiceClient` composes `MicFrameSender` + `TtsAudioPlayer` + optional `SpeakerEchoGuard`.
- [x] Automatic WebSocket event dispatch to player and guard.
- [x] `onEvent` observer for app-level monitoring.
- [x] Methods: `start()`, `stop()`, `close()`, plus `mic`/`player`/`guard` accessors.
- [x] Vanilla script, no dependencies on other JS modules.

### 6.3 Speaker echo guard ✅

Implementation:

- [x] `converse_framework/js/speaker-echo-guard.js`.
- [x] `SpeakerEchoGuard` class.
- [x] Constructor: `tailDelayMs` (350), `mode` (drop/pause), `onStateChange`, `clock`.
- [x] Methods: `onTtsEvent()`, `isSuppressed()`, `shouldSendFrame()`, `attachMicSender()`, `release()`.
- [x] Suppresses on `tts.first_chunk` / first `tts.audio`.
- [x] Stays suppressed across streaming `tts.audio` chunks.
- [x] Tail timer after final `tts.audio`, `tts.cancelled`, `tts.error`, `turn.finished`.
- [x] 15s fallback timeout for streams that never mark final.
- [x] `"drop"` mode: continue capture, skip send.
- [x] `"pause"` mode: stop capture, resume on tail expiry.
- [x] Integrates via `shouldSendFrame` option on `MicFrameSender`.
- [x] `attachMicSender()` wires the gate without app glue.
- [x] UMD wrapper.

Tests:

- [x] `tests/js/test_speaker_echo_guard.mjs` — 22 Node-compatible tests with mock clock:
  - Starts idling.
  - `tts.first_chunk` enters suppressed.
  - `tts.audio` enters suppressed.
  - Final `tts.audio` + tail delay → resume.
  - Multiple chunks keep suppression active.
  - `tts.cancelled` → tail → resume.
  - `tts.error` → tail → resume.
  - `turn.finished` → tail → resume.
  - `release()` clears state.
  - `onStateChange` callback fires suppressed/tail/idling.
- [x] `tests/js/manual-smoke-test.html` includes guard toggle with visual indcicator.

## Phase 7: CUDA DLL Discovery Helper and Docs ✅

### 7.1 Add Windows NVIDIA wheel DLL helper ✅

Implementation:

- [x] `converse_framework/cuda_utils.py`.
- [x] `discover_nvidia_dll_dirs() -> list[Path]` — searches site-packages for `nvidia/*/bin` folders with `.dll` files.
- [x] `add_nvidia_dll_directories() -> list[object]` — calls `os.add_dll_directory()` on discovered dirs, returns handles.
- [x] `format_nvidia_dll_diagnostic() -> str` — human-readable diagnostic string for log output.
- [x] Windows-only via `sys.platform != "win32"` guard.
- [x] Deduplicates identical search roots within the same process.
- [x] Searches: `nvidia/cublas/bin`, `nvidia/cudnn/bin`, `nvidia/cusparse/bin`, `nvidia/cusolver/bin`, `nvidia/curand/bin`.
- [x] Only returns directories that contain at least one `.dll` file.
- [x] Does not raise when no directories are found; returns empty list.
- [x] Integrated into `FasterWhisperASRProvider._ensure_model()`:
  - [x] Called before importing `WhisperModel`.
  - [x] Controlled by `auto_cuda_dll_dirs` config option (default `True`).
  - [x] Handles are cleared on `unload()`.
  - [x] Best-effort — failures are logged, not raised.

Tests:

- [x] `tests/test_cuda_utils.py` — **14 tests**:
  - `discover_nvidia_dll_dirs`:
    - non-Windows returns empty.
    - discovers 3 NVIDIA DLL dirs from fake tree (cublas, cudnn, cusparse).
    - empty when no site-packages.
    - deduplicates identical roots.
  - `add_nvidia_dll_directories`:
    - non-Windows no-op.
    - invokes `os.add_dll_directory` for each discovered dir.
    - handles `add_dll_directory` failure gracefully (returns empty).
  - `format_nvidia_dll_diagnostic`:
    - non-Windows output mentions Windows-only.
    - Windows without DLLs reports "none found".
    - Windows with DLLs lists filenames.
  - Provider integration:
    - `_ensure_model()` calls CUDA helper when enabled.
    - `_ensure_model()` skips helper when `auto_cuda_dll_dirs=False`.
  - `_get_search_roots`:
    - returns existing Path objects.
    - deduplicates.
- [x] 239 total Python tests (was 225).

## Phase 8: Mobile HTTPS and Tunnel Recipe ✅

Implementation:

- [x] README section "Mobile Browser Microphone Testing" added after browser playback section.
- [x] Recipes included: localhost desktop, same-LAN caveat, Cloudflare Tunnel, ngrok, local trusted cert (`mkcert`).
- [x] No launcher script — repo has no `scripts/` pattern; kept documented to avoid operational dependencies.
- [x] WebSocket URL forms table: `ws://localhost:8000/ws`, `ws://<lan-ip>:8000/ws`, `wss://<tunnel>/ws`, `wss://<lan-ip>:8000/ws`.
- [x] `converse_framework/examples/websocket_voice_chat.py` docstring updated with HTTPS/tunnel note.
- [x] Acceptance criteria met: doc explains why `http://<LAN-IP>` fails on mobile, includes both URL forms.
- [x] Also added Browser Microphone Capture section documenting `mic-frame-sender.js`, `speaker-echo-guard.js`, and `browser-voice-client.js` alongside the mobile testing section.

## Phase 9: Documentation and Migration Updates ✅

Implementation:

- [x] **README.md updates:**
  - [x] Quick Start: added "Provider status semantics" subsection with ``probe_status()`` / ``load_status()`` usage and ``status_level`` values.
  - [x] WebSocket session helper section: added before Examples section documenting all 7 message types, ``WebSocketSessionConfig``, ``WebSocketSessionHooks``, and usage sketch.
  - [x] Browser microphone helper section: done in Phase 8.
  - [x] Runtime provider update section: done in Phase 3.
  - [x] Pocket TTS voice listing/configuration recipe: added as recipe subsection with ``list_voices()``, ``configure(voice=...)``, ``configure(quantize=...)``, ``configure(max_tokens=...)`` examples.
  - [x] CUDA DLL helper section: added as recipe subsection with ``add_nvidia_dll_directories()``, ``discover_nvidia_dll_dirs()``, ``format_nvidia_dll_diagnostic()``, and FasterWhisperASRProvider auto-discovery usage.
  - [x] Mobile HTTPS/tunnel recipe: done in Phase 8.
  - [x] Framework/App Boundary updated to list ``WebSocketSession``, browser JS helpers, CUDA utils, and v0.2 features.
- [x] **MIGRATION.md updates:**
  - [x] Added WebSocket session routing note under "Behavior the framework does NOT own" — apps can keep existing handlers or opt into ``WebSocketSession``.
  - [x] "TTS hot-swap UX stays entirely in app" already replaced with "settings UX stays in app; provider configure/reload mechanics are framework-supported" (was done in Phase 3).
  - [x] ``probe_status()`` vs ``load_status()`` section already present at end of file (was added during Phase 2).
- [x] **Docstring updates:**
  - [x] ``converse_framework/examples/websocket_voice_chat.py`` docstring updated with ``.. seealso::`` pointing to ``WebSocketSession``.
  - [x] All public methods in ``cuda_utils.py``, ``session.py``, ``provider_events.py`` have docstrings. Inner callbacks in ``session.py`` (``utterance_callback``, ``cancel_callback``) are implementation details, not public API.
- [x] **239 Python tests pass** (unchanged).

## Phase 10: Compatibility and Release Strategy ✅

Backward compatibility:

- [x] `check_status()` kept through next minor version — present on all providers and `ProviderStatus` protocol.
- [x] `ProviderBundle.check_statuses()` kept as alias for probe semantics.
- [x] `build_websocket_voice_runtime()` and `handle_websocket_message()` kept as compatibility wrappers in the WebSocket example.
- [x] Existing progress/error event types kept intact; fields and events added rather than replaced.
- [x] `PocketTTSProvider.set_quantize()` kept alongside `configure()` and `set_voice()`.

Versioning:

- [x] Version bumped from `0.1.0` to `0.2.0` in `pyproject.toml`.
- [x] `CHANGELOG.md` created with full summary of all v0.2.0 features, breaking changes (ProviderStatus shape, set_quantize deprecation), migration notes, and what's next.
- [x] **239 Python tests pass** (unchanged).
- [x] **47 JS tests pass** (unchanged).

## Suggested Implementation Order

1. Phase 1: fix faster-whisper lazy load and richer exception payloads. This removes the most painful runtime failure with low API risk.
2. Phase 2: introduce status semantics and lifecycle events. This creates the vocabulary used by sessions and settings UIs.
3. Phase 4: add Pocket TTS voice/configuration support. This directly improves settings UX and exercises the new status metadata.
4. Phase 3: add provider bundle/pipeline/collector update APIs. This gives apps a safe reload path.
5. Phase 5: implement `WebSocketSession` on top of the stable reload/status/event APIs.
6. Phase 6: add browser microphone helper, optional composed browser client, and speaker echo guard.
7. Phase 7 and Phase 8: add CUDA and mobile testing support.
8. Phase 9 and Phase 10: finalize docs, compatibility notes, and release metadata.

## Cross-Cutting Test Plan

- Run `python -m pytest` after each phase.
- Add focused tests near existing coverage rather than one giant integration test:
  - `tests/test_pipeline.py`: richer error payloads and provider lifecycle events.
  - `tests/test_providers.py`: faster-whisper lazy load, Pocket TTS voice/configure/listing, CUDA helper invocation.
  - `tests/test_registry.py`: `probe_statuses()`, `load_statuses()`, serialized `ProviderStatus` shape.
  - `tests/test_utterance_collector.py`: VAD provider swap behavior.
  - `tests/test_examples.py` or `tests/test_session.py`: `WebSocketSession` routing/reload/status hooks.
  - JS helper tests if a lightweight Node path is introduced for pure conversion helpers.
- Manually smoke test:
  - text pipeline with mock providers;
  - WebSocket recipe with mock providers;
  - browser playback plus mic frame sender in a desktop browser;
  - browser speaker echo guard using phone/laptop speakers over a tunnel;
  - faster-whisper CPU first-use load path;
  - Windows CUDA discovery helper on a machine with NVIDIA wheels installed.

## Done Criteria

- Apps can use a reusable `WebSocketSession` without copying the recipe state machine.
- Apps can safely update provider config at runtime without manually coordinating pipeline and collector internals.
- Pocket TTS supports first-class voice changes and voice listing.
- Browser demos can use shipped JS for mic input, TTS output, and speaker echo suppression.
- Faster Whisper loads lazily on first audio transcription without assertion failures.
- Error events always include a non-empty message and exception class.
- Provider statuses clearly distinguish cached state, cheap probes, and heavyweight loads.
- Provider lifecycle events use a consistent cross-provider shape.
- Windows CUDA setup has a documented and testable DLL discovery helper.
- Mobile browser microphone testing has a documented HTTPS/tunnel path.
