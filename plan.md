# Converse Framework Extraction Plan

## Summary

Extract the provider-agnostic speech stack from the reference Conversational AI Harness into a standalone Python package named `converse-framework`.

The framework should provide reusable contracts and mechanics for speech-to-speech applications:

- Provider protocols for VAD, ASR, LLM, and TTS.
- Audio frame parsing, PCM conversion, metering, and silence trimming.
- A stable event sink and event wire shape.
- A streaming speech pipeline for text/audio turns.
- A VAD-driven utterance collector.
- A lazy provider registry with optional concrete providers.

The framework must not own application policy. FastAPI, WebSockets, browser UI, profiles, runtime settings, character cards, companion memory, and TTS preset UX stay in the harness.

Source reference repository:

```text
C:\Users\pegas\Desktop\LLama\Test apps\backup\Converse-Framework\Reference-Repository-Conversational-AI-Harness
```

Target framework repository:

```text
C:\Users\pegas\Desktop\LLama\Test apps\backup\Converse-Framework
```

## Boundary Decisions

### Framework Owns

- Provider interfaces and shared provider dataclasses.
- Generic provider bundle and lazy provider registry.
- Audio utilities and frame/stat dataclasses.
- Event sink API and compatibility event envelope.
- Pipeline turn orchestration: ASR -> LLM -> TTS, streaming TTS chunks, cancellation, barge-in support.
- VAD utterance collection: pre-buffering, speech start/end state, max utterance length, energy gates, silence trimming.
- Generic test transports/utilities such as `QueueTransport`.
- Optional concrete providers that are dependency-isolated behind extras.

### Harness Owns

- FastAPI app, REST endpoints, WebSocket handler, browser UI.
- Profile loading and profile file layout.
- Runtime settings persistence.
- Character card parsing and first-message seeding.
- Companion mode policy and memory store.
- TTS preset manager and provider hot-swap UX.
- Launch scripts, doctor scripts, local operations.
- WebSocket-specific transport implementation.

### Important Refactor Rule

Do not extract `runtime_settings.py` into the framework. Instead, convert framework-facing behavior into injectable callables or plain config:

- `system_prompt_builder(mode, manual_prompt, messages) -> str`
- `sampler_builder(mode) -> dict[str, Any]`
- optional app hooks for seeding, memory, and companion behavior in the harness only

The framework may support arbitrary conversation modes as string keys, but it must not special-case `chat`, `companion`, character cards, or memory.

## Target Package Shape

```text
converse_framework/
|-- __init__.py
|-- protocols.py
|-- events.py
|-- audio_utils.py
|-- pipeline.py
|-- utterance_collector.py
|-- registry.py
|-- transport.py
`-- providers/
    |-- __init__.py
    |-- mock.py
    |-- silero.py
    |-- faster_whisper.py
    |-- llamacpp.py
    |-- kokoro_onnx.py
    |-- pocket_tts.py
    `-- unavailable.py
```

Base install dependency:

```toml
dependencies = ["numpy>=2.0"]
```

Optional provider extras:

```toml
[project.optional-dependencies]
silero = ["silero-vad>=6.0", "onnxruntime>=1.20"]
faster-whisper = ["faster-whisper>=1.2", "nvidia-cublas-cu12; platform_system == 'Windows'"]
llamacpp = ["httpx>=0.28"]
kokoro = ["kokoro-onnx>=0.5", "misaki>=0.7"]
pocket-tts = ["pocket-tts>=2.1"]
all-vad = ["converse-framework[silero]"]
all-asr = ["converse-framework[faster-whisper]"]
all-llm = ["converse-framework[llamacpp]"]
all-tts = ["converse-framework[kokoro,pocket-tts]"]
all = ["converse-framework[all-vad,all-asr,all-llm,all-tts]"]
```

If pip extra self-references are not accepted by the build backend, flatten `all` into the concrete dependency list before publishing.

## Public API

`converse_framework.__init__` should explicitly export:

```python
from converse_framework import (
    VADProvider,
    ASRProvider,
    LLMProvider,
    TTSProvider,
    AudioChunk,
    TranscriptEvent,
    VADEvent,
    ProviderCapabilities,
    ProviderStatus,
    AudioFrame,
    AudioFrameStats,
    EventSink,
    FrameworkEvent,
    QueueEventSink,
    ProviderBundle,
    SpeechPipeline,
    PipelineConfig,
    AudioUtteranceCollector,
    UtteranceCollectorConfig,
    register_provider,
    build_provider,
    build_provider_bundle,
    is_provider_available,
    Transport,
    QueueTransport,
)
```

Event compatibility rule:

- Keep the existing event wire shape for v0.1:

```python
{"type": event_type, "ts": timestamp, "payload": payload}
```

- Keep `EventSink.emit(event_type: str, **payload)`.
- Rename `HarnessEvent` to `FrameworkEvent`, but provide a temporary alias:

```python
HarnessEvent = FrameworkEvent
```

- Defer full typed event subclasses until after the harness runs on the extracted package.

## Phase 1: Package Skeleton, Core Types, and Lazy Registry

Goal: create a standalone framework package whose base import does not import FastAPI, provider backends, `httpx`, model libraries, or harness app modules.

Tasks:

- [x] Create `pyproject.toml`, `converse_framework/`, and explicit `__all__`.
- [x] Copy `providers/base.py` to `protocols.py`; remove harness wording from docstrings only.
- [x] Copy `events.py` to `events.py`; keep `EventSink.emit()` and event envelope compatibility.
- [x] Move `QueueEventSink` from `orchestrator.py` into `events.py`.
- [x] Merge `audio.py` and `audio_frames.py` into `audio_utils.py`.
- [x] Preserve existing function names:
  - `pcm_s16le_to_float32`
  - `float_audio_to_pcm_s16le_bytes`
  - `float_audio_to_wav_bytes`
  - `make_tone_wav`
  - `compute_pcm16_level`
  - `trim_pcm16_silence`
  - `parse_audio_frame`
- [x] Preserve `AudioFrame` and `AudioFrameStats`.
- [x] Implement `registry.py` early, before concrete providers move:
  - `ProviderBundle`
  - `register_provider(kind, name, *, availability_probe=None)`
  - `build_provider(kind, name, config)`
  - `build_provider_bundle(config: Mapping[str, Mapping[str, Any]], *, tts_provider=None)`
  - `is_provider_available(kind, name)`
  - status serialization helpers if still needed by the harness
- [x] Implement lazy provider loading with import strings or registration modules so importing `converse_framework` only imports Python stdlib plus `numpy`.
- [x] Add `providers/mock.py` and `providers/unavailable.py` first because they require no heavy dependencies.
- [x] Add initial framework tests for protocols, events, audio utilities, registry, and mock provider bundle.

Acceptance checks:

```powershell
python -c "import converse_framework; print(converse_framework.__all__)"
python -m pytest
```

The first command must not require FastAPI, `httpx`, `silero-vad`, `faster-whisper`, `kokoro-onnx`, or `pocket-tts`.

## Phase 2: Harness Compatibility Redirect

Goal: make the reference harness import the extracted core types without changing runtime behavior.

Tasks:

- [x] Add the framework as an editable local dependency for harness development:

```powershell
python -m pip install -e .
```

Run that from the `Converse-Framework` package root inside the environment used to test the harness.

- [x] Replace harness imports for copied core modules (via compatibility shims):
  - `conversational_harness.providers.base` -> `converse_framework.protocols` shim
  - `conversational_harness.events` -> `converse_framework.events` shim
  - `conversational_harness.audio` -> `converse_framework.audio_utils` shim
  - `conversational_harness.audio_frames` -> `converse_framework.audio_utils` shim
- [x] Keep small harness-local compatibility modules if needed to reduce churn during the transition.
- [x] Keep `config.py`, `runtime_settings.py`, and `tts_runtime.py` in the harness.
- [x] Keep concrete providers in the harness until Phase 4 unless a provider is already dependency-clean.
- [x] Ensure existing harness tests pass after import redirects.

Acceptance checks:

```powershell
python -m pytest  # 86 passed, 1 skipped ✓
```

Base import clean:

```powershell
python -c "import converse_framework; print(converse_framework.__all__)"  # ✓ no heavy deps
```

Manual smoke check:

- [ ] Start the harness.
- [ ] Confirm `/api/status` has the same provider summary shape as before.
- [ ] Confirm browser text turn still works with mock providers.

## Phase 3: Extract Pipeline Without App Policy

Goal: move turn orchestration into `SpeechPipeline` while leaving prompt policy, companion behavior, memory, and character seeding in the harness.

Tasks:

- [x] Copy `ConversationOrchestrator` to `SpeechPipeline`.
- [x] Replace the `RuntimeSettings` dependency with injected callables/config:

```python
@dataclass
class PipelineConfig:
    tts_chunk_chars: int = 120
    min_tts_chars: int = 0
    default_mode: str = "chat"

SystemPromptBuilder = Callable[[str, str, list[dict[str, str]]], str]
SamplerBuilder = Callable[[str], dict[str, Any]]
```

- [x] Constructor shape:

```python
class SpeechPipeline:
    def __init__(
        self,
        providers: ProviderBundle,
        sink: EventSink,
        config: PipelineConfig | None = None,
        system_prompt_builder: SystemPromptBuilder | None = None,
    ) -> None:
        ...
```

- [x] Keep these methods:
  - `handle_text_turn(text, mode="chat")`
  - `handle_audio_turn(pcm_s16le, sample_rate, mode="chat")`
  - `handle_continue(mode="chat")`
  - `set_system_prompt(prompt, mode="chat")`
  - `clear_conversation(mode="chat")`
  - `cancel_tts(reason)`
  - `messages_for_mode(mode)`
  - `update_turn_config(...)`
- [x] Remove from framework pipeline:
  - character first-message seeding
  - `MemoryStore`
  - direct `RuntimeSettings`
  - direct companion-specific prompt assembly
- [x] Let the harness implement seeding and memory by reading `pipeline.messages_for_mode()` and by supplying `system_prompt_builder`.
- [x] Keep emitted event names and payloads compatible with the current browser UI.
- [x] Add framework tests:
  - text turn with mock providers
  - audio turn with mock ASR
  - continue turn
  - TTS chunk flushing
  - cancellation and stale TTS task cleanup
  - separate conversation histories for arbitrary mode names

Harness adaptation:

- [x] Replace `ConversationOrchestrator` usage with `SpeechPipeline`.
- [x] Implement a harness-local `system_prompt_builder` that delegates to `RUNTIME_SETTINGS.effective_system_prompt(...)` and `MEMORY_STORE.read()` when the mode is companion.
- [x] Move character first-message seeding into harness code that appends/sets messages through an explicit pipeline helper or a narrow harness-side compatibility method.

Acceptance checks:

```powershell
python -m pytest  # framework: 66 passed; harness: 86 passed, 1 skipped
```

Manual smoke check:

- [ ] Text turn still streams LLM tokens and TTS audio.
- [ ] Character first message still appears in the harness.
- [ ] Companion memory summarization still uses companion conversation history.

## Phase 4: Extract VAD Utterance Collector

Goal: remove the nonlocal VAD state machine from the WebSocket receiver and make it reusable without FastAPI/WebSocket dependencies.

Tasks:

- [x] Create `UtteranceCollectorConfig`:

```python
@dataclass
class UtteranceCollectorConfig:
    sample_rate: int = 16000
    channels: int = 1
    frame_ms: int = 30
    pre_speech_ms: int = 450
    max_utterance_ms: int = 30000
    min_speech_duration_ms: int = 300
    reject_low_energy_rms: float = 0.003
    reject_low_energy_max_duration_ms: int = 900
    reject_utterance_rms: float = 0.002
    trim_silence_rms: float = 0.003
    trim_silence_frame_ms: int = 30
```

- [x] Compute derived values inside the config or collector:
  - `pre_speech_frames = pre_speech_ms // frame_ms`
  - `max_utterance_frames = max_utterance_ms // frame_ms`
  - `bytes_per_ms = sample_rate * channels * 2 // 1000`
  - `expected_frame_bytes = bytes_per_ms * frame_ms`
- [x] Preserve current behavior but fix duration math to include channels.
- [x] Create `AudioUtteranceCollector`:

```python
class AudioUtteranceCollector:
    def __init__(
        self,
        vad_provider: VADProvider,
        event_sink: EventSink,
        utterance_callback: Callable[[bytes, int, str], Awaitable[None]],
        config: UtteranceCollectorConfig | None = None,
        cancel_callback: Callable[[str], Awaitable[None]] | None = None,
        pre_speech_start_hook: Callable[[AudioFrame, str], Awaitable[None]] | None = None,
    ) -> None:
        ...

    async def ingest_frame(
        self,
        frame: AudioFrame,
        *,
        mode: str = "chat",
        pre_speech_start_hook: Callable[[AudioFrame, str], Awaitable[None]] | None = None,
    ) -> None:
        ...

    async def cancel_active_turn(self, reason: str) -> None:
        ...

    @property
    def is_recording(self) -> bool:
        ...

    @property
    def current_mode(self) -> str:
        ...
```

- [x] Move the following state into instance attributes:
  - pre-buffer
  - utterance buffer
  - recording flag
  - recording mode
  - audio stats
- [x] Keep this rejection order:
  - minimum duration
  - short low-energy utterance
  - whole-utterance RMS floor
  - silence edge trimming
- [x] Emit compatible event types from the collector:
  - `audio.input_level`
  - `vad.error`
  - `vad.probability`
  - `vad.speech_start`
  - `vad.speech_end`
  - `vad.speech_rejected`
  - `asr.audio_trimmed`
  - `asr.buffer_warning`
  - Note: `audio.frame_error` is emitted by the harness receiver (not the collector) because WebSocket payload parsing stays in the harness.
  - Note: `vad.speech_start` and `vad.speech_end` no longer carry a `source` field; the harness emits its own `vad.speech_start` with `source="browser"` from the browser-sent message.
- [x] Keep WebSocket JSON parsing in the harness. The framework collector accepts `AudioFrame`, not raw WebSocket payloads.
- [x] Refactor the harness receiver to:
  - parse message
  - call `parse_audio_frame(...)`
  - pass frames into `AudioUtteranceCollector` (with a per-frame `pre_speech_start_hook` that reads `system_prompt` from the WebSocket payload and calls `orchestrator.set_system_prompt(...)`)
  - route text/continue/control messages to `SpeechPipeline`

Framework tests (`tests/test_utterance_collector.py`):

- [x] speech start drains pre-buffer
- [x] speech end dispatches utterance bytes
- [x] minimum-duration rejection
- [x] low-energy rejection
- [x] utterance RMS rejection
- [x] silence trimming event
- [x] max utterance closes current utterance
- [x] VAD probability forwarding
- [x] barge-in cancellation callback fires on speech start
- [x] additional coverage: config derived values (mono/stereo), `pre_speech_start_hook` ordering vs cancel callback, per-frame hook override, VAD `ValueError` -> `vad.error`, `is_recording` / `current_mode` state, independent `AudioFrameStats` per collector, callback `sample_rate` propagation, zero-threshold pass-through.

Acceptance checks:

```powershell
python -m pytest  # framework: 91 passed (66 existing + 25 new); harness: 86 passed, 1 skipped ✓
python -c "import converse_framework; print(converse_framework.__all__)"  # ✓ no heavy deps, 33 exports
```

Manual smoke check:

- [ ] Browser mic flow still transcribes speech.
- [ ] Brief noise/keystrokes are still rejected.
- [ ] Barge-in still cancels active TTS.

## Phase 5: Move Concrete Providers Behind Extras

Goal: make concrete providers reusable while preserving base-package lightness.

Tasks:

- [x] Move provider implementations into `converse_framework.providers`.
- [x] Convert imports to framework modules only:
  - `converse_framework.protocols`
  - `converse_framework.audio_utils`
  - `converse_framework.registry`
- [x] Remove app path dependencies:
  - `KokoroOnnxProvider` must not import `PROJECT_ROOT`; default cache path should come from config or a platform cache directory.
  - `LlamaCppProvider` must not import or accept `RuntimeSettings`; sampler values come from config at construction or from an injected sampler callable if added to the LLM protocol.
- [x] Register providers lazily:

```python
register_provider("vad", "silero", "converse_framework.providers.silero:SileroVADProvider")
register_provider("asr", "faster-whisper", "converse_framework.providers.faster_whisper:FasterWhisperASRProvider")
register_provider("llm", "llamacpp", "converse_framework.providers.llamacpp:LlamaCppProvider")
register_provider("tts", "kokoro-onnx", "converse_framework.providers.kokoro_onnx:KokoroOnnxProvider")
register_provider("tts", "pocket-tts", "converse_framework.providers.pocket_tts:PocketTTSProvider")
```

- [x] Ensure each missing dependency error says which extra to install:
  - `pip install converse-framework[silero]`
  - `pip install converse-framework[faster-whisper]`
  - `pip install converse-framework[llamacpp]`
  - `pip install converse-framework[kokoro]`
  - `pip install converse-framework[pocket-tts]`
- [x] Keep harness-specific provider aliases or profile-name mapping in the harness if needed.
- [x] Update harness factory code to use framework registry and `ProviderBundle`.

Tests:

- [x] base import with no provider extras installed
- [x] each provider module imports when its dependencies are available
- [x] missing dependency produces friendly unavailable status or clear exception
- [x] `build_provider_bundle` builds mock-only bundle without heavy imports
- [x] provider status serialization remains compatible with `/api/status`

Harness adaptation:

- [x] Harness provider modules (`silero`, `faster_whisper`, `llamacpp`, `kokoro_onnx`, `pocket_tts`, `mock`, `unavailable`) are now thin shims that re-export from `converse_framework.providers`.
- [x] Harness `providers/factory.py` delegates `build_provider_bundle` to the framework registry while keeping the existing `ProviderBundle` import path and the `serialize_status` / `serialize_statuses` JSON shape.
- [x] Harness `main.py` wires the llama.cpp LLM through the new `set_sampler_provider` callable (passing `RUNTIME_SETTINGS.effective_sampler`) instead of `set_runtime_settings`.

Acceptance checks:

```powershell
python -m pytest  # framework: 116 passed (91 existing + 25 new); harness: 86 passed, 1 skipped
python -c "import converse_framework; print(converse_framework.__all__)"  # clean base import, 34 exports (added extra_hint_for)
```

Manual smoke check (still requires a real provider extra installed):

- [ ] Start the harness with a `silero` / `faster-whisper` / `kokoro-onnx` / `pocket-tts` profile and confirm the friendly `pip install converse-framework[<extra>]` hint appears when the extra is missing.

## Phase 6: Transport and Second Consumer

Goal: prove the framework is useful outside the browser harness.

Tasks:

- Define `Transport` protocol:

```python
class Transport(Protocol):
    async def send_event(self, event: FrameworkEvent) -> None:
        ...

    async def receive_event(self) -> FrameworkEvent:
        ...
```

- Implement `QueueTransport` for tests.
- Keep `WebSocketTransport` in the harness, not the framework.
- Add a small example or optional CLI consumer after the pipeline and collector are stable.
- The CLI can use mock providers by default and real providers only when extras are installed.
- Do not make the CLI the first proof of extraction; framework tests and harness compatibility come first.

Tests:

- `QueueTransport` round trip.
- CLI/example text conversation with mock providers.
- Optional manual voice example documented separately.

## Phase 7: Documentation, API Review, and Publish Prep

Goal: stabilize the v0.1 API before publishing.

Tasks:

- Audit `__all__` and public docstrings.
- Document the framework/app boundary clearly.
- Add README examples:
  - minimal mock text pipeline
  - audio frame -> utterance collector -> pipeline
  - custom provider registration
  - custom event sink
- Document provider extras and missing dependency behavior.
- Add migration notes for the harness.
- Run performance comparison against the original harness path:
  - first token latency
  - first TTS chunk latency
  - speech start to ASR start
  - barge-in cancellation latency
- Check package name availability before publishing.
- Build and inspect package metadata:

```powershell
python -m build
python -m twine check dist/*
```

Publish only after:

- framework tests pass independently
- harness tests pass against the framework package
- browser text and mic flows pass manually
- base import stays light
- at least one non-harness consumer/example works

## Testing Matrix

Framework unit tests:

- protocols instantiate through mock implementations
- event envelope compatibility
- queue event sink
- audio conversion and WAV generation
- audio frame validation and dropped-frame metrics
- silence trimming and level metering
- registry lazy loading and unavailable provider behavior
- mock provider bundle
- text pipeline turn
- audio pipeline turn
- continue turn
- TTS chunking
- cancellation and barge-in
- utterance collector VAD state and rejection gates
- queue transport

Harness regression tests:

- existing test suite unchanged in behavior
- `/api/status` shape
- runtime settings API
- character upload/import/delete
- companion memory endpoints
- TTS preset switching
- browser text turn
- browser mic turn
- manual TTS cancel
- VAD barge-in

Dependency tests:

- `pip install .` installs only base dependencies
- `pip install .[silero]`
- `pip install .[faster-whisper]`
- `pip install .[llamacpp]`
- `pip install .[kokoro]`
- `pip install .[pocket-tts]`
- `pip install .[all]`

## Key Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---:|---|
| App policy leaks into framework | High | Do not extract `runtime_settings.py`; use callables and harness-side adapters. |
| Event typing breaks browser UI | Medium | Keep dict event wire shape for v0.1; add typed events later. |
| Base package imports heavy providers | High | Implement lazy registry before moving concrete providers. |
| Llama sampler coupling blocks provider extraction | Medium | Replace `RuntimeSettings` access with config/callable before moving provider. |
| VAD collector misses current edge cases | Medium | Port current rejection order exactly and cover with canned-frame tests. |
| Duration math changes behavior | Medium | Include channels in framework math and add tests for mono/stereo duration. |
| CLI distracts from extraction | Low | Add CLI/example only after framework and harness compatibility are stable. |

## Success Criteria

1. `import converse_framework` works with only `numpy` installed.
2. Harness tests pass with core imports redirected to the framework.
3. Browser text and microphone flows behave the same as the reference harness.
4. Framework tests pass without FastAPI, WebSocket, profile files, browser UI, or harness runtime settings.
5. Provider selection is registry-based and lazy.
6. Changing providers remains a config change, not a pipeline code change.
7. A second consumer/example can run a text conversation using the framework without importing the harness.
