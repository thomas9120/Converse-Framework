# Migration guide: adopting `converse-framework` v0.1

The framework is in v0.1 pre-release. The public surface is the
explicit `__all__` in `converse_framework/__init__.py` — 34
symbols at the time of writing — governed by
`.harness/docs/standards.md` and the Boundary Decisions in
[`plan.md`](./plan.md). Anything not in `__all__` is internal and
may change without notice. The reference consumer in this repo is
`Reference-Repository-Conversational-AI-Harness/`.

## Who this is for

This guide is for application authors who previously copied or
re-implemented the speech stack (provider protocols, audio
utilities, event sink, speech pipeline, VAD utterance collector,
provider registry) inside their own repository and now want to
depend on `converse-framework` v0.1 as a separately installed
package. The payoff is a smaller app codebase, a stable public
API for the provider-agnostic core, and the ability to pick up
framework bug fixes and new providers without editing the app.
The app keeps ownership of its transport, profiles, settings,
character cards, companion-mode policy, and any other
application-level behavior — none of those move into the
framework.

## Install

```bash
pip install converse-framework
pip install converse-framework[silero]          # Silero VAD
pip install converse-framework[faster-whisper]  # faster-whisper ASR
pip install converse-framework[llamacpp]        # llama.cpp HTTP LLM
pip install converse-framework[kokoro]          # Kokoro ONNX TTS
pip install converse-framework[pocket-tts]      # Pocket TTS
pip install converse-framework[all]             # everything
```

For local development against an in-tree checkout, install the
package as editable from the framework root: `python -m pip install -e .`.
End-to-end install recipes (per-profile extras, CUDA
`cublas64_12.dll` workarounds, OS-specific `start.*` /
`install.*` scripts) live in `README.md` and in the consumer
app's own docs.

## Import changes

The framework's public surface is a single namespace
`converse_framework`. Replace your local copies of the speech
stack with imports from the package. The mapping below uses the
reference harness's old paths as concrete examples — replace the
`conversational_harness.*` prefix with your app's equivalent
module path.

| Old path (in the consumer app)                                | New path (in `converse_framework`)            |
|---------------------------------------------------------------|-----------------------------------------------|
| `conversational_harness.providers.base`                       | `converse_framework.protocols`                |
| `conversational_harness.events`                               | `converse_framework` (or `events`)            |
| `conversational_harness.audio`                                | `converse_framework.audio_utils`              |
| `conversational_harness.audio_frames`                         | `converse_framework.audio_utils`              |
| `conversational_harness.providers.{mock,silero,faster_whisper,llamacpp,kokoro_onnx,pocket_tts,unavailable}` | `converse_framework.providers.<x>` |
| `conversational_harness.transport.WebSocketTransport`         | **stays in the consumer app** (see below)     |
| `conversational_harness.orchestrator.ConversationOrchestrator` | `converse_framework.pipeline.SpeechPipeline` (with a harness-side subclass if you need hooks) |

For the symbols themselves, the canonical import site is the
package root:

```python
from converse_framework import (
    VADProvider, ASRProvider, LLMProvider, TTSProvider,
    AudioChunk, TranscriptEvent, VADEvent,
    ProviderCapabilities, ProviderStatus,
    AudioFrame, AudioFrameStats,
    pcm_s16le_to_float32, float_audio_to_pcm_s16le_bytes,
    float_audio_to_wav_bytes, make_tone_wav,
    compute_pcm16_level, trim_pcm16_silence, parse_audio_frame,
    EventSink, QueueEventSink, FrameworkEvent,
    SpeechPipeline, PipelineConfig,
    AudioUtteranceCollector, UtteranceCollectorConfig,
    ProviderBundle, register_provider, build_provider,
    build_provider_bundle, is_provider_available,
    Transport, QueueTransport,
    extra_hint_for,
)
```

The full symbol list is `converse_framework.__all__`. The
reference harness's local compatibility shims under
`app/conversational_harness/{events,audio,audio_frames,providers/*}.py`
are now thin re-exports pointing at the framework. New consumers
can skip the shims and import directly from `converse_framework`;
existing shims can stay to avoid churning call sites during the
transition. `HarnessEvent` is exported as a compatibility alias
of `FrameworkEvent`. Use `FrameworkEvent` in new code; the alias
is kept for one minor version.

## Behavior the framework does NOT own

The framework draws a hard line at provider-agnostic mechanics.
The following stay in the consumer app. For each item, the
framework provides an injection point or a clearly defined
contract the app uses instead of baking the behavior in.

- **FastAPI app, REST endpoints, WebSocket handler.** The
  framework has no FastAPI dependency. The app owns the HTTP
  surface and the browser-facing endpoints. The only public
  contract for moving events is the `Transport` protocol
  (`send_event` / `receive_event` on `FrameworkEvent`).
- **WebSocket transport implementation.** Keep your
  `WebSocketTransport` (or equivalent) in the consumer app; it
  implements the framework's `Transport` protocol. The reference
  harness exposes
  `conversational_harness.transport.WebSocketTransport`.

  As of v0.2 the framework also offers an optional reusable
  `WebSocketSession` (``converse_framework.session``) that owns
  the message-dispatch loop for browser-based voice apps,
  including provider reload, status requests, settings updates,
  and event dispatch.  Apps that previously copied this loop
  from the WebSocket recipe can switch to the session helper;
  apps that already own their own handler can keep it unchanged.
  The session class is **not** in the top-level
  ``__init__.py`` — apps opt in via an explicit import.
- **Profile file loading and layout.** The framework reads no
  profile files. Pass the relevant sections into
  `build_provider_bundle(config={...})` as a plain mapping.
- **Runtime settings persistence.** `user_settings.json`,
  `RuntimeSettings`, and the three-tier sampler merge (server
  `/props` → profile → user overrides) stay in the app. The
  pipeline accepts an injected `system_prompt_builder` callable
  that pulls settings into the prompt without the framework
  importing `RuntimeSettings`.
- **Character card parsing and first-message seeding.** TavernAI
  V2 PNG/JSON parsing, `{{user}}`/`{{char}}` substitution, and
  `first_mes` seeding are app policy. The pipeline exposes
  `messages_for_mode(mode)` so the app can read and prepend the
  seed; the app emits `conversation.seeded` and skips TTS for
  it.
- **Companion mode policy and memory store.** `memory.md` save /
  summarize / clear, the Companion tab, and Companion sampler
  overrides stay in the app. The framework treats modes as
  opaque string keys; the pipeline keeps a separate history per
  mode key. The app injects the companion-specific prompt
  assembly and memory read through `system_prompt_builder`.
- **TTS preset manager and provider hot-swap UX.** The runtime
  TTS selector, preset switching, and the
  `/api/tts/{select,load,unload,voice}` endpoints are app UX.
  The framework exposes `build_provider_bundle(..., tts_provider=...)`
  so the app can inject a harness-managed TTS instance instead
  of the registry-built one, and `provider.unload()` for the
  lifecycle. As of v0.2 the framework also provides safe swap
  mechanics (`ProviderBundle.replace()`, `pipeline.update_providers()`,
  `collector.update_vad_provider()`) — the app still owns the
  settings UX that triggers the swap, but the low-level
  coordination (cancelling in-flight TTS, emitting lifecycle
  events, unloading old providers) is now handled by the
  framework.
- **`WebSocketTransport`**, **`config.py`**, **`runtime_settings.py`**, **`tts_runtime.py`**,
  character card parser, memory store, and doctor / start /
  install scripts all stay in the consumer. The framework never
  imports them.

## Provider registration

The framework ships a built-in registry that knows the mock
providers and the optional concrete providers behind extras. The
consumer does not need to register the built-in providers — they
are already wired in `converse_framework.registry` (see the
`register_provider(...)` calls at the bottom of that module). The
consumer's job is to (1) read each kind's `provider` field from
the active profile, (2) call `build_provider_bundle({...})` with
sections shaped like `{"vad": {...}, "asr": {...}, "llm": {...},
"tts": {...}, "audio": {...}}`, (3) optionally pass a
harness-managed TTS via `tts_provider=` to override the
registry-built one, and (4) surface friendly missing-extra
messages built by `extra_hint_for(kind, name)` (e.g.
`pip install converse-framework[silero]`) in the status
endpoint and doctor script.

```python
from converse_framework import (
    build_provider_bundle, extra_hint_for, is_provider_available,
)

profile_sections = {
    "vad":   {"provider": "silero",         "speech_threshold": 0.6},
    "asr":   {"provider": "faster-whisper", "model": "large-v3-turbo"},
    "llm":   {"provider": "llamacpp",       "base_url": "http://127.0.0.1:8080"},
    "tts":   {"provider": "pocket-tts",     "voice": "azelma"},
    "audio": {"sample_rate": 16000, "channels": 1, "frame_ms": 30},
}

if not is_provider_available("vad", "silero"):
    print(f"Install silero VAD with: pip install {extra_hint_for('vad', 'silero')}")

bundle = build_provider_bundle(profile_sections)
```

**Registering your own provider.** If you ship a custom provider
that does not live under `converse_framework.providers`, register
it with an import string and an optional `availability_probe`
(returns `True` only when the provider is genuinely ready to
use — heavy dep installed, model loaded, network reachable,
etc.; without it the registry falls back to a best-effort module
import):

```python
from converse_framework import register_provider

register_provider(
    "tts", "my-cloud-tts",
    "myapp.providers.my_cloud_tts:MyCloudTTSProvider",
    availability_probe=lambda: _my_cloud_tts_sdk_importable(),
)
```

## Configuration surface that moved out of the framework

The framework does not own any application-level configuration.
The following items **used to be** either on the framework side
or duplicated in the consumer; after extraction they belong
solely in the consumer. The column on the right names the module
where the reference harness keeps it.

| Item                                       | Lives in the consumer (reference harness path)               |
|--------------------------------------------|--------------------------------------------------------------|
| Profile loading, `HarnessConfig`, `PROJECT_ROOT`, `DEFAULT_PROFILE` | `conversational_harness.config`                              |
| `RuntimeSettings`, `user_settings.json` persistence, sampler merge (`effective_sampler`, `sampler_display`) | `conversational_harness.runtime_settings` |
| TTS runtime / preset manager (long-lived TTS instance, model load/unload, voice selection) | `conversational_harness.tts_runtime`            |
| `WebSocketTransport` (FastAPI adapter)     | `conversational_harness.transport`                            |
| Doctor / readiness / pre-flight checks     | `conversational_harness.doctor` and `doctor.{ps1,sh}`        |
| Launch entry point                         | `conversational_harness.launch`                              |
| Character card parser (TavernAI V2 PNG / JSON) | app-local `character_cards` module (within `runtime_settings`) |
| Companion memory store (`memory.md`)       | app-local `MemoryStore` (within `runtime_settings`)          |
| `start.*` / `install.*` / `stop.*` / `update.*` / `tunnel.*` scripts | the harness repo root                 |

What moved **into** the framework, in case the consumer used to
carry its own copies:

- Provider protocols and shared dataclasses (`VADProvider`,
  `ASRProvider`, `LLMProvider`, `TTSProvider`, `ProviderStatus`,
  `ProviderCapabilities`, `AudioChunk`, `TranscriptEvent`,
  `VADEvent`).
- Audio utilities (`AudioFrame`, `AudioFrameStats`,
  `pcm_s16le_to_float32`, `float_audio_to_pcm_s16le_bytes`,
  `float_audio_to_wav_bytes`, `make_tone_wav`,
  `compute_pcm16_level`, `trim_pcm16_silence`,
  `parse_audio_frame`).
- Event sink API and wire shape (`EventSink`, `QueueEventSink`,
  `FrameworkEvent`).
- `SpeechPipeline` and `PipelineConfig` (turn orchestration).
- `AudioUtteranceCollector` and `UtteranceCollectorConfig` (VAD
  state machine, pre-buffer, rejection gates, silence trimming).
- Lazy provider registry (`register_provider`, `build_provider`,
  `build_provider_bundle`, `is_provider_available`,
  `ProviderBundle`).
- `Transport` protocol and `QueueTransport` test double.
- `extra_hint_for` helper for friendly missing-dep messages.

## Event wire compatibility

The on-the-wire shape of a framework event is unchanged from the
pre-extraction harness: `{"type": event_type, "ts": timestamp,
"payload": payload}`. `FrameworkEvent.to_json()` produces exactly
that dict, and `QueueEventSink.emit` writes the same shape into
its queue, so any existing browser client that reads
`event.type`, `event.ts`, and `event.payload` keeps working. The
framework does not introduce typed event subclasses in v0.1; that
work is deferred to a later minor version. `HarnessEvent` is
exported as a temporary alias of `FrameworkEvent` for one minor
version; new code should import `FrameworkEvent` directly.

## Testing migration

Redirect the consumer's existing test suite to import from
`converse_framework` instead of the local copy: replace
`conversational_harness.providers.base` with
`converse_framework.protocols`, the local `audio` / `audio_frames`
shims with `converse_framework.audio_utils`, and the local
`events` shim with `converse_framework` (or `events`). Reuse
`QueueEventSink` and `QueueTransport` as test doubles and use
`build_provider_bundle({"vad": {"provider": "mock"}, ...})` as
the default fixture for unit tests. The framework's tests under
`tests/` are the canonical example of how to use the public API.

```bash
# Framework (from the converse-framework repo root)
python -m pytest

# Consumer (from inside the consumer repo)
python -m pytest
```

Both should pass against the same `converse_framework` version.
Pin the framework dependency in the consumer's
`requirements.txt` (or `pyproject.toml`) so test reproducibility
matches the framework release you validated against.

## Rollback plan

If a v0.1 release regresses the consumer, pin the consumer to the
last known-good in-tree version of the speech stack. Restore the
consumer's old local copies plus the small compatibility shims
under
`app/conversational_harness/{events,audio,audio_frames,providers/base,providers/silero,providers/faster_whisper,providers/llamacpp,providers/kokoro_onnx,providers/pocket_tts,providers/mock,providers/unavailable}.py`,
remove the editable install (`pip uninstall converse-framework`
and drop `-e .` from the consumer's `requirements.txt`), and the
shims fall back to importing the consumer's own local modules. No
data migration is involved — the rollback is purely an
import-source change.

## Probe vs Load Status

As of v0.2, provider status has three tiers:

- ``status`` (property): cached state, no I/O.
- ``probe_status()``: cheap check (import probe, HTTP reachability)
  that does not load models. Used by ``status_only()`` and
  ``ProviderBundle.probe_statuses()``.
- ``load_status()``: may load or initialize heavy resources before
  returning the status. Used by ``ProviderBundle.load_statuses()``.

The old ``check_status()`` is kept for backward compatibility.
Callers that only need a quick readiness check should prefer
``probe_status()``.

## Reference: from the harness

The reference harness in
`Reference-Repository-Conversational-AI-Harness/` is the
concrete worked example. `conversational_harness/providers/base.py`
is a 14-line shim that re-exports `ASRProvider`, `LLMProvider`,
`TTSProvider`, `VADProvider`, `ProviderStatus`,
`ProviderCapabilities`, `AudioChunk`, `TranscriptEvent`,
`VADEvent`, and `ProgressCallback` from
`converse_framework.protocols`. `conversational_harness/audio.py`
and `conversational_harness/audio_frames.py` are equally thin;
both pull their public names from
`converse_framework.audio_utils` so the rest of the harness can
keep importing `from conversational_harness.audio import
make_tone_wav` without edits, and `audio_frames.py` re-exports
`AudioFrame`, `AudioFrameStats`, `parse_audio_frame`,
`trim_pcm16_silence`, `compute_pcm16_level`, and
`SUPPORTED_ENCODING`. The harness's `orchestrator.py` is a thin
subclass of the framework `SpeechPipeline` that supplies the
app's `system_prompt_builder` (delegating to
`RuntimeSettings.effective_system_prompt`) and adds a
`seed_character_first_message` method that pokes the
character-card seed into `pipeline.state.messages`; turn
orchestration, streaming, cancellation, barge-in, and per-mode
history all come from the framework unchanged. The harness's
`transport.py` keeps the `WebSocketTransport` that implements
the framework's `Transport` protocol over a FastAPI `WebSocket`;
the framework never imports FastAPI, and `main.py` still owns
the `/ws/events` handler, the send/receive tasks, the pre-send
event queue, and ASR warmup — only the boundary object was
extracted.
