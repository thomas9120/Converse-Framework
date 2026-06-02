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

## Phase 3: Runtime Provider Reload and Configuration Pattern

### 3.1 Add public provider bundle update helpers

Problem: apps currently rebuild a bundle and recreate the collector because the pipeline owns the provider bundle and the collector owns the VAD provider.

Implementation:

- Add an immutable-ish update API to `ProviderBundle` in `converse_framework/registry.py`:
  - `replace(**providers) -> ProviderBundle`
  - `async unload_replaced(old_bundle, new_bundle) -> list[ProviderStatus]` as a helper or method.
- Add `SpeechPipeline.update_providers(providers: ProviderBundle, *, cancel_active_tts: bool = True, reason: str = "provider_reload")`.
  - Cancels active TTS by default.
  - Swaps `self.providers`.
  - Emits `provider.loaded` or `providers.updated` with serialized statuses.
  - Does not clear conversation history unless caller asks separately.
- Add `AudioUtteranceCollector.update_vad_provider(vad_provider: VADProvider)`.
  - Reject while recording, just like `update_config()`.
  - Emit a clear error if called during an active utterance.
- Add a higher-level helper for common app code:
  - `build_runtime_components(config, transport, collector_config, pipeline_config)` can remain in the WebSocket recipe or move to the new session module in Phase 4.

Tests:

- Pipeline test proving `update_providers()` swaps ASR/LLM/TTS without clearing messages.
- Collector test proving `update_vad_provider()` rejects while recording and succeeds when idle.
- Registry tests for `ProviderBundle.replace()`.

### 3.2 Document runtime reload recipes

Implementation:

- Add README section "Runtime Provider Updates".
- Include recipes for:
  - Updating TTS voice on an existing provider when supported.
  - Rebuilding a bundle from config and swapping it into pipeline/collector.
  - Calling `probe_statuses()` before swap and `load_statuses()` only after user confirmation.
- Update `MIGRATION.md` boundary section: app still owns settings persistence, but framework now owns the safe swap mechanics.

## Phase 4: First-Class TTS Voice Configuration

### 4.1 Add optional provider configuration protocol

Problem: `PocketTTSProvider.set_quantize()` exists, but voice changes require rebuilding the provider. UIs also hard-code voice names.

Implementation:

- Add dataclasses in `converse_framework/protocols.py`:
  - `ProviderOption` or `VoiceInfo` with `id`, `label`, `language`, `description`.
  - `ProviderConfigResult` with `status: ProviderStatus`, `changed: bool`, `requires_reload: bool`, `message: str`.
- Add optional protocol methods:
  - `async def configure(self, **options) -> ProviderConfigResult`
  - `async def list_voices(self) -> tuple[VoiceInfo, ...]`
- Keep direct provider-specific setters where they already exist, but implement them through `configure()` where practical.

### 4.2 Implement Pocket TTS voice and config support

Implementation:

- Update `converse_framework/providers/pocket_tts.py`.
- Add `set_voice(voice: str) -> ProviderStatus`:
  - If same voice, keep loaded state.
  - If changed, clear `_voice_state`.
  - Keep `_model` if language/temp/quantize are unchanged.
  - Clear `_load_error`.
  - Return updated `status`.
- Add `configure(...)` supporting:
  - `voice`
  - `quantize`
  - `language`
  - `temp`
  - `max_tokens`
  - `coalesce_ms`
- Define reload behavior:
  - `voice` change reloads voice state only.
  - `quantize`, `language`, `temp` unload model and voice state.
  - `max_tokens`, `coalesce_ms` do not unload.
- Add `list_voices()`:
  - Prefer upstream `pocket_tts` metadata if available.
  - Fallback to a small documented default list including `azelma`, marked as best-effort.
  - Never import heavy model state just to list voices if the package exposes metadata separately.
- Populate `ProviderStatus.voices` and `active_voice`.

Tests:

- Add `tests/test_providers.py` cases for:
  - `set_voice()` clears only `_voice_state`.
  - `set_quantize()` still clears model and voice state.
  - `configure(max_tokens=...)` does not unload.
  - `list_voices()` returns structured voice info with current voice represented when possible.

## Phase 5: Reusable WebSocket Session Helper

### 5.1 Move recipe glue into a framework-owned session module

Problem: the WebSocket example is useful, but real apps still duplicate session glue for settings, status, and runtime provider rebuilds.

Implementation:

- Create `converse_framework/session.py` or `converse_framework/websocket_session.py`.
- Keep it transport-generic: it should depend on the framework `Transport` protocol, not FastAPI.
- Add `WebSocketSessionConfig`:
  - `provider_config`
  - `collector_config`
  - `pipeline_config`
  - `default_mode`
  - `auto_probe_status`
  - optional hooks
- Add `WebSocketSessionHooks`:
  - `on_unknown_message(session, message_type, payload)`
  - `on_settings_update(session, payload)`
  - `on_status_request(session, payload)`
  - `on_before_provider_reload(session, old_bundle, new_config)`
  - `on_after_provider_reload(session, old_bundle, new_bundle)`
  - `on_event(session, event)` if apps need monitoring.
- Add `WebSocketSession`:
  - Owns `transport`, `sink`, `ProviderBundle`, `SpeechPipeline`, `AudioUtteranceCollector`, and `AudioFrameStats`.
  - Method `async handle_message(message: dict[str, Any]) -> None`.
  - Method `async run_forever(receive: Callable[[], Awaitable[dict]])` or leave receive loop in the app to avoid committing too much.
  - Method `async reload_providers(config, *, load: bool = False)`.
  - Method `async emit_status(kind: str = "probe")`.
- Built-in message types:
  - `audio.frame`
  - `text.turn`
  - `conversation.clear`
  - `tts.cancel`
  - `status.request`
  - `settings.update` routed to hook by default.
  - `providers.reload`
- Unknown messages call hook first, then emit `turn.error` if unhandled.
- Refactor `converse_framework/examples/websocket_voice_chat.py` to use `WebSocketSession` while preserving existing imports/functions where possible:
  - `WebSocketVoiceRuntime` can become a compatibility wrapper or alias.
  - `build_websocket_voice_runtime()` can construct a session-backed runtime.
  - `handle_websocket_message()` delegates to `session.handle_message()`.

Tests:

- Extend `tests/test_examples.py` or add `tests/test_session.py`.
- Cover built-in message routing, unknown-message hook, status request, provider reload, and audio frame error.
- Assert no FastAPI import is required for `converse_framework.session`.

### 5.2 Public exports

Implementation:

- Export session classes from `converse_framework/__init__.py` if they are considered public.
- Otherwise document `converse_framework.session` as the import path and keep package-root exports modest.
- Update `__all__` tests if present.

Docs:

- README: replace "consumer owns all WebSocket glue" with a more nuanced boundary:
  - framework owns optional session helper and message routing;
  - app still owns actual web server, auth, routes, settings persistence, and deployment.

## Phase 6: Browser Microphone and Frame Sender Helper

### 6.1 Add browser audio capture module

Problem: browser clients hand-roll PCM capture, resampling, frame slicing, and WebSocket sending. Playback helper exists, but input helper does not.

Implementation:

- Add `converse_framework/js/mic-frame-sender.js`.
- No npm/bundler dependency; vanilla browser script like `tts-audio-player.js`.
- Public class: `MicFrameSender`.
- Constructor options:
  - `webSocket`
  - `sampleRate = 16000`
  - `channels = 1`
  - `frameMs = 30`
  - `mode = "chat"`
  - `messageType = "audio.frame"`
  - `onLevel`
  - `onError`
  - `audioContext`
- Methods:
  - `start()`
  - `stop()`
  - `setMode(mode)`
  - `setWebSocket(ws)`
  - `close()`
- Responsibilities:
  - Request microphone through `navigator.mediaDevices.getUserMedia`.
  - Capture via `AudioWorklet` when available; use `ScriptProcessorNode` fallback only if needed.
  - Downsample from device rate to target rate.
  - Convert float samples to PCM s16le.
  - Slice exact `frameMs` frames.
  - Send framework-compatible payload:
    - `type: "audio.frame"`
    - `payload.data`: base64 PCM s16le
    - `payload.sample_rate`
    - `payload.channels`
    - `payload.frame_ms`
    - `payload.sequence`
    - `payload.encoding: "pcm_s16le"`
    - `payload.mode`
- Include comments documenting that mobile microphone access requires HTTPS, localhost, or an approved tunnel.

Tests:

- Add JS unit tests only if the repo has a JS test harness. If not, add a small Node-compatible test for pure helpers:
  - base64 conversion
  - float-to-pcm conversion
  - frame slicing
- Add README manual smoke test with a tiny HTML page.

### 6.2 Optional combined browser client

Implementation:

- Add a small `converse_framework/js/browser-voice-client.js` that composes `MicFrameSender` and `TtsAudioPlayer`.
- Keep it optional; do not make `tts-audio-player.js` depend on it.

Docs:

- Add README section "Browser Voice Client".
- Show minimal HTML that:
  - opens a WebSocket,
  - starts `MicFrameSender`,
  - plays `tts.audio` events with `TtsAudioPlayer`.

## Phase 7: CUDA DLL Discovery Helper and Docs

### 7.1 Add Windows NVIDIA wheel DLL helper

Problem: on Windows, `nvidia-cublas-cu12` installs DLLs under `site-packages/nvidia/cublas/bin`, but CTranslate2 may not find them unless launchers add those folders to the DLL search path.

Implementation:

- Add `converse_framework/cuda_utils.py`.
- Public helpers:
  - `discover_nvidia_dll_dirs() -> list[Path]`
  - `add_nvidia_dll_directories() -> list[object]`
  - `format_nvidia_dll_diagnostic() -> str`
- Behavior:
  - Only active on Windows for `os.add_dll_directory`.
  - Search installed distributions or `site.getsitepackages()` / `sys.path` for:
    - `nvidia/cublas/bin`
    - `nvidia/cudnn/bin`
    - future `nvidia/*/bin` folders with `.dll` files.
  - Return handles from `os.add_dll_directory()` so callers can keep them alive.
  - Do not raise if no folders are found; return an empty list with a diagnostic string.
- Consider calling this helper inside `FasterWhisperASRProvider._ensure_model()` before importing/constructing `WhisperModel`, but make it conservative:
  - Windows only.
  - best effort.
  - can be disabled by config, e.g. `{"auto_cuda_dll_dirs": False}`.

Tests:

- Unit test discovery with monkeypatched paths/temp dirs.
- Unit test non-Windows no-op behavior with monkeypatch.
- Provider test that `_ensure_model()` invokes helper when enabled, without requiring CUDA.

Docs:

- README faster-whisper section:
  - Explain CPU vs CUDA install.
  - Explain Windows `nvidia-cublas-cu12` DLL layout.
  - Show launcher snippet:
    ```python
    from converse_framework.cuda_utils import add_nvidia_dll_directories
    _dll_handles = add_nvidia_dll_directories()
    ```

## Phase 8: Mobile HTTPS and Tunnel Recipe

Problem: browser WebSocket microphone access needs HTTPS or a trusted localhost context for mobile testing.

Implementation:

- Add README section "Mobile Browser Microphone Testing".
- Include recipes for:
  - local desktop browser on `localhost`;
  - same LAN with HTTPS caveat;
  - Cloudflare Tunnel;
  - ngrok;
  - local cert option for advanced users.
- Add an example launcher script only if the repo already has a scripts pattern. Otherwise keep this documented to avoid adding operational dependencies.
- In the WebSocket example docstring, mention that mobile microphone use requires HTTPS/tunnel even if the server recipe itself is plain WebSocket locally.

Acceptance criteria:

- A user can run the WebSocket recipe locally and know why mobile mic access fails over plain `http://<LAN-IP>`.
- The docs include the expected WebSocket URL forms:
  - `ws://localhost:8000/ws`
  - `wss://<tunnel-host>/ws`

## Phase 9: Documentation and Migration Updates

Implementation:

- Update `README.md`:
  - Quick Start additions for provider status semantics.
  - WebSocket session helper section.
  - Browser microphone helper section.
  - Runtime provider update recipe.
  - Pocket TTS voice listing/configuration recipe.
  - CUDA DLL helper section.
  - Mobile HTTPS/tunnel recipe.
- Update `MIGRATION.md`:
  - Clarify that v0.1 apps can still own WebSocket handlers, but the framework now offers optional reusable session routing.
  - Replace "TTS hot-swap UX stays entirely in app" with "settings UX stays in app; provider configure/reload mechanics are framework-supported."
  - Mention `probe_status()` vs `load_status()`.
- Update `converse_framework/examples/websocket_voice_chat.py` docstring to point to the new session helper.
- Add docstrings for every new public class/method.

## Phase 10: Compatibility and Release Strategy

Backward compatibility:

- Keep `check_status()` through at least the next minor version.
- Keep `ProviderBundle.check_statuses()` as an alias for probe semantics.
- Keep `build_websocket_voice_runtime()` and `handle_websocket_message()` as compatibility wrappers after introducing `WebSocketSession`.
- Keep existing progress/error event types and add fields/events rather than replacing them.
- Keep direct `PocketTTSProvider.set_quantize()` while adding `configure()` and `set_voice()`.

Versioning:

- This is a minor pre-release feature set. If `ProviderStatus` shape changes are considered public API sensitive, bump to `0.2.0`.
- Include a changelog entry summarizing:
  - session helper,
  - provider reload/configure/status APIs,
  - browser mic helper,
  - improved provider/error events,
  - CUDA DLL helper.

## Suggested Implementation Order

1. Phase 1: fix faster-whisper lazy load and richer exception payloads. This removes the most painful runtime failure with low API risk.
2. Phase 2: introduce status semantics and lifecycle events. This creates the vocabulary used by sessions and settings UIs.
3. Phase 4: add Pocket TTS voice/configuration support. This directly improves settings UX and exercises the new status metadata.
4. Phase 3: add provider bundle/pipeline/collector update APIs. This gives apps a safe reload path.
5. Phase 5: implement `WebSocketSession` on top of the stable reload/status/event APIs.
6. Phase 6: add browser microphone helper and optional composed browser client.
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
  - faster-whisper CPU first-use load path;
  - Windows CUDA discovery helper on a machine with NVIDIA wheels installed.

## Done Criteria

- Apps can use a reusable `WebSocketSession` without copying the recipe state machine.
- Apps can safely update provider config at runtime without manually coordinating pipeline and collector internals.
- Pocket TTS supports first-class voice changes and voice listing.
- Browser demos can use shipped JS for both mic input and TTS output.
- Faster Whisper loads lazily on first audio transcription without assertion failures.
- Error events always include a non-empty message and exception class.
- Provider statuses clearly distinguish cached state, cheap probes, and heavyweight loads.
- Provider lifecycle events use a consistent cross-provider shape.
- Windows CUDA setup has a documented and testable DLL discovery helper.
- Mobile browser microphone testing has a documented HTTPS/tunnel path.
