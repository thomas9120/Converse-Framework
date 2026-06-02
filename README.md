# Converse Framework

Provider-agnostic speech stack for speech-to-speech applications.

## Install

```bash
pip install converse-framework
```

The base install pulls in only `numpy`. Real VAD / ASR / LLM / TTS
providers live behind optional extras:

```bash
pip install converse-framework[silero]          # Silero VAD
pip install converse-framework[faster-whisper]  # faster-whisper ASR
pip install converse-framework[whisper-cpp]     # whisper.cpp HTTP ASR
pip install converse-framework[llamacpp]        # llama.cpp HTTP LLM
pip install converse-framework[kokoro]          # Kokoro ONNX TTS
pip install converse-framework[pocket-tts]      # Pocket TTS
pip install converse-framework[all]             # everything
```

### Missing dependency behavior

If a config requests a provider whose heavy backend is not installed,
`build_provider` (and therefore `build_provider_bundle`) returns an
`UnavailableProvider` sentinel for that slot instead of raising a bare
`ImportError`. The sentinel's `status.message` always names the provider
that was missing and includes the `pip install` extra to fix it. The
mapping is owned by `converse_framework.providers.unavailable.EXTRA_HINTS`
and exposed as `extra_hint_for(kind, name)`, which returns the extra
name (e.g. `"converse-framework[silero]"`) when one is known and `None`
otherwise.

```python
from converse_framework import extra_hint_for
from converse_framework.providers.unavailable import UnavailableProvider

print(extra_hint_for("vad", "silero"))          # converse-framework[silero]
print(extra_hint_for("asr", "faster-whisper"))  # converse-framework[faster-whisper]
print(extra_hint_for("vad", "made-up"))         # None

p = UnavailableProvider("vad", "silero")
print(p.status.message)
# Provider 'silero' (vad) is not available. Install the required extra
# with `pip install converse-framework[silero]`.
```

`is_provider_available(kind, name)` is the companion check: it returns
`True` only when the provider's heavy dependency is importable, so you
can fail fast before handing the config to a pipeline. `UnavailableProvider`
is a real implementation of all four provider protocols, so the rest of
the pipeline keeps running (turns fail with a clear `RuntimeError` when
the broken provider is actually invoked) and the consumer can decide
whether to prompt for the install or fall back to a different provider.

### Python version compatibility

The base package supports Python 3.11 and newer. Each extra has its
own constraints (the table below mirrors the markers in
`pyproject.toml`):

| Extra | Python | Notes |
|---|---|---|
| base | 3.11+ | `numpy>=2.0` is the only required runtime dependency. |
| `silero` | 3.11+ | `silero-vad` + `onnxruntime`. No known upper bound. |
| `faster-whisper` | 3.11+ | The `nvidia-cublas-cu12` wheel pins Windows. |
| `llamacpp` | 3.11+ | `httpx` itself supports 3.9+, so 3.11+ is the only constraint. |
| `whisper-cpp` | 3.11+ | Only needs `httpx`, which supports 3.9+. |
| `kokoro` | 3.11 to <3.14 | `kokoro-onnx` 0.5.0 requires Python <3.14. The wheel build fails fast on 3.14+. |
| `pocket-tts` | 3.11+ | No known upper bound. |

The `kokoro` extra is the only one with an upper-bound marker today.
If you are on Python 3.14+ and need a TTS provider, use `pocket-tts`
or a mock provider. New providers should add their own
`python_version` markers in `pyproject.toml` when their backend has a
known limit.

## Quick Start

```python
from converse_framework import (
    build_provider_bundle,
    QueueEventSink,
)
import asyncio

config = {
    "vad": {"provider": "mock"},
    "asr": {"provider": "mock"},
    "llm": {"provider": "mock"},
    "tts": {"provider": "mock"},
}

bundle = build_provider_bundle(config)
print(bundle.statuses())
```

`import converse_framework` only needs `numpy` to be installed — heavy
provider backends are loaded lazily through the registry.

## Recipes

The recipes below are short, self-contained scripts that exercise the
public API. They all run with the base install (`numpy` + the framework)
unless a snippet is explicitly fenced as `requires the \`<extra>\` extra`.

### Minimal mock text pipeline

`build_provider_bundle` returns a fully-mock provider bundle and
`SpeechPipeline` runs an end-to-end text turn against it. `QueueEventSink`
captures every event the pipeline emits so the script can assert or
print them.

```python
import asyncio

from converse_framework import (
    PipelineConfig,
    QueueEventSink,
    SpeechPipeline,
    build_provider_bundle,
)


async def main():
    queue: asyncio.Queue = asyncio.Queue()
    sink = QueueEventSink(queue)
    pipeline = SpeechPipeline(
        providers=build_provider_bundle(
            {
                "vad": {"provider": "mock"},
                "asr": {"provider": "mock"},
                "llm": {"provider": "mock"},
                "tts": {"provider": "mock"},
            }
        ),
        sink=sink,
        config=PipelineConfig(tts_chunk_chars=80),
    )

    await pipeline.handle_text_turn("Hello, mock pipeline.")
    # Let the TTS streaming task finish, then drain the captured events.
    await asyncio.sleep(0.5)
    types = [queue.get_nowait()["type"] for _ in range(queue.qsize())]
    print(types)


asyncio.run(main())
```

### Audio frame to utterance collector to pipeline

`parse_audio_frame` validates a wire payload and turns it into an
`AudioFrame`. `AudioUtteranceCollector` runs VAD on the frame, applies
the rejection gates, and on `vad.speech_end` hands the assembled PCM
bytes to its `utterance_callback`. The recipe wires that callback into
`SpeechPipeline.handle_audio_turn`. The in-process VAD below fires
`vad.speech_start` on the first frame and `vad.speech_end` on the third
so the collector has something to dispatch — the framework's own
`MockVADProvider` returns no events and is not useful for this path.

```python
import asyncio
import base64

from converse_framework.audio_utils import AudioFrameStats, parse_audio_frame
from converse_framework.events import QueueEventSink
from converse_framework.pipeline import PipelineConfig, SpeechPipeline
from converse_framework.protocols import (
    ProviderCapabilities,
    ProviderStatus,
    VADEvent,
)
from converse_framework.registry import build_provider_bundle
from converse_framework.utterance_collector import (
    AudioUtteranceCollector,
    UtteranceCollectorConfig,
)


class ScriptedVAD:
    """A tiny in-process VAD: start on frame 0, end on frame 2."""

    def __init__(self) -> None:
        self._count = 0

    @property
    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name="scripted",
            kind="vad",
            ready=True,
            message="Scripted VAD fires start at frame 0 and end at frame 2.",
            capabilities=ProviderCapabilities(),
        )

    async def check_status(self) -> ProviderStatus:
        return self.status

    async def process_frame(self, frame):
        self._count += 1
        events: list[VADEvent] = []
        if self._count == 1:
            events.append(VADEvent(type="vad.speech_start", probability=1.0, audio_ms=30))
        if self._count == 3:
            events.append(VADEvent(type="vad.speech_end", probability=1.0, audio_ms=90))
        return events


async def main():
    queue: asyncio.Queue = asyncio.Queue()
    sink = QueueEventSink(queue)
    bundle = build_provider_bundle(
        {
            "vad": {"provider": "mock"},
            "asr": {"provider": "mock"},
            "llm": {"provider": "mock"},
            "tts": {"provider": "mock"},
        }
    )
    pipeline = SpeechPipeline(providers=bundle, sink=sink, config=PipelineConfig(tts_chunk_chars=80))

    cfg = UtteranceCollectorConfig(
        sample_rate=16000,
        channels=1,
        frame_ms=30,
        # Disable the rejection gates -- this recipe shows the wiring
        # from frame to pipeline, not the collector's silence handling.
        min_speech_duration_ms=0,
        reject_low_energy_rms=0,
        reject_utterance_rms=0,
        trim_silence_rms=0,
    )
    stats = AudioFrameStats(
        expected_sample_rate=16000,
        expected_channels=1,
        expected_frame_ms=30,
    )

    async def on_utterance(pcm: bytes, sample_rate: int, mode: str) -> None:
        await pipeline.handle_audio_turn(pcm, sample_rate, mode=mode)

    collector = AudioUtteranceCollector(
        vad_provider=ScriptedVAD(),
        event_sink=sink,
        utterance_callback=on_utterance,
        config=cfg,
    )

    # Three 30 ms frames of silence (16 kHz mono -> 480 samples -> 960 bytes).
    silence = base64.b64encode(b"\x00\x00" * 480).decode("ascii")
    for seq in range(3):
        frame = parse_audio_frame(
            {
                "data": silence,
                "sample_rate": 16000,
                "channels": 1,
                "frame_ms": 30,
                "sequence": seq,
                "encoding": "pcm_s16le",
            },
            stats,
        )
        await collector.ingest_frame(frame)

    await pipeline.cancel_tts("done")
    await asyncio.sleep(0.3)
    types = [queue.get_nowait()["type"] for _ in range(queue.qsize())]
    print(types)


asyncio.run(main())
```

### Custom provider registration

`register_provider` adds a new (kind, name) pair to the registry by
import string. `build_provider_bundle` then resolves the name on demand
and instantiates the class. `is_provider_available` is the companion
probe — it returns `True` only when the underlying module can be
imported, which is the safe check before handing the config to a
pipeline. The recipe points the new name at the framework's own mock
VAD so it runs against the base install; replace the import string
with your own `my_pkg.providers:MyVADProvider` to register a real
implementation.

```python
from converse_framework.registry import (
    build_provider_bundle,
    is_provider_available,
    register_provider,
)

# Register a custom VAD name. Replace the import string with your own
# `my_pkg.providers:MyVADProvider` to wire up a real implementation.
register_provider(
    "vad",
    "my-vad",
    "converse_framework.providers.mock:MockVADProvider",
)

bundle = build_provider_bundle(
    {
        "vad": {"provider": "my-vad"},
        "asr": {"provider": "mock"},
        "llm": {"provider": "mock"},
        "tts": {"provider": "mock"},
    }
)
print(bundle.vad.status.provider_id)        # "mock" (the registered class)
print(is_provider_available("vad", "my-vad"))  # True
```

### Custom event sink

`SpeechPipeline` accepts any `EventSink` subclass. The recipe prints
each event as it fires, which is handy when you are wiring up a new
transport and want to see the wire shape without standing up a queue.

```python
import asyncio

from converse_framework import (
    EventSink,
    PipelineConfig,
    SpeechPipeline,
    build_provider_bundle,
)


class PrintSink(EventSink):
    """Minimal sink that prints each event as it fires."""

    async def emit(self, event_type, **payload):
        keys = ", ".join(payload) or "-"
        print(f"[event] {event_type} ({keys})")


async def main():
    sink = PrintSink()
    pipeline = SpeechPipeline(
        providers=build_provider_bundle(
            {
                "vad": {"provider": "mock"},
                "asr": {"provider": "mock"},
                "llm": {"provider": "mock"},
                "tts": {"provider": "mock"},
            }
        ),
        sink=sink,
        config=PipelineConfig(tts_chunk_chars=80),
    )
    await pipeline.handle_text_turn("Hello, custom sink.")
    # Let the TTS streaming task finish before the loop exits.
    await asyncio.sleep(0.5)


asyncio.run(main())
```

#### Browser playback (JS reference client)

The framework ships a vanilla JavaScript / Web Audio reference client at
`converse_framework/js/tts-audio-player.js` that turns the framework's
`tts.audio` events into sound without bundling a build step. It builds
`AudioBuffer`s directly from PCM s16le bytes (avoiding
`decodeAudioData` on tiny chunks) and coalesces consecutive events
within a short window before scheduling, which is the same fix that
resolved Pocket TTS choppiness in the reference harness.

```html
<script src="converse_framework/js/tts-audio-player.js"></script>
<script>
  const player = new TtsAudioPlayer({ coalesceMs: 80 });
  ws.addEventListener('message', (ev) => {
    const event = JSON.parse(ev.data);
    if (event.type === 'tts.audio') player.onEvent(event);
  });
  // when the conversation ends:
  player.close();
</script>
```

The reference client handles the most common case (mono / stereo PCM
s16le with explicit sample rate, channels, and `final` flag) and
ignores anything that is not `pcm_s16le` with a console warning. Drop
the file into your static assets directory; no npm / bundler required.

#### Wrap an external CLI as a provider

When the engine you want to use is only available as a CLI binary
(`whisper-cli`, `whisper.cpp/main`, the Vosk CLI, …), the framework's
`converse_framework.examples.subprocess_provider` shows the pattern.
The class shells out to a configured binary, writes a WAV header
followed by the caller's PCM s16le body to the subprocess's stdin,
and yields the subprocess's stdout as a single final transcript
event.

```python
from converse_framework.examples.subprocess_provider import (
    SubprocessASRProvider,
)

provider = SubprocessASRProvider({
    "binary": "whisper-cli",
    "model": "ggml-small.en.bin",
    "command_template": ["-m", "{model}", "-f", "-"],
    "timeout_s": 120,
})
# Then plug it into a ProviderBundle:
from converse_framework.registry import build_provider_bundle
bundle = build_provider_bundle(
    {
        "vad": {"provider": "mock"},
        "asr": {"provider": "subprocess"},   # see note below
        "llm": {"provider": "mock"},
        "tts": {"provider": "mock"},
    },
)
```

`SubprocessASRProvider` is shipped as a recipe (not a registered
provider) because it is generic: copy the class, point it at your
binary of choice, and register it with `register_provider("asr",
"my-name", "my.module:MySubprocessProvider")`. The example also
ships a fake-echo script (`--use-fake-echo`) that lets the driver
run end-to-end in CI without installing any real ASR.

## Examples

The framework ships two opt-in example consumers. They live under
`converse_framework/examples/` and are not imported by the framework
package, so the base install stays light.

### Text chat (automated-test covered)

Run a real text conversation against `SpeechPipeline` using only the
framework's public API. No FastAPI, no WebSocket, no profile files.

```bash
python -m converse_framework.examples.text_chat
```

Try a real provider by passing overrides (the matching extra must be
installed):

```bash
python -m converse_framework.examples.text_chat \
    --provider asr=faster-whisper \
    --provider llm=llamacpp \
    --provider tts=kokoro
```

The driver behind the CLI is `converse_framework.examples.text_chat.run_text_chat`,
which is what the test suite exercises.

### Voice chat (manual)

The voice example wires an `AudioUtteranceCollector` to the pipeline
and feeds it PCM frames. It is a **manual** example — you supply a
WAV file (or replace the source with a microphone capture) and the
script drives the conversation. It is intentionally not covered by
the automated tests because it depends on platform audio I/O.

```bash
# With real providers installed
python -m converse_framework.examples.voice_chat --input path/to/16k_mono.wav

# Or run the same flow with mock providers to validate the path
python -m converse_framework.examples.voice_chat --mock --input path/to/16k_mono.wav
```

## Framework / App Boundary

The framework owns the **provider-agnostic speech stack**:

* Provider protocols (`VADProvider`, `ASRProvider`, `LLMProvider`, `TTSProvider`).
* Audio frame parsing, PCM conversion, metering, and silence trimming.
* Event sink API and the wire shape used by the browser UI.
* `SpeechPipeline` turn orchestration (ASR → LLM → TTS, streaming
  chunks, cancellation, barge-in).
* `AudioUtteranceCollector` (VAD-driven utterance collection).
* A lazy provider registry and the optional concrete providers
  behind extras.

The framework does **not** own the application. The following stay in
the consumer app (e.g. the reference harness):

* FastAPI app, REST endpoints, WebSocket handler.
* Profile files and runtime settings persistence.
* Character card parsing and first-message seeding.
* Companion mode policy and memory store.
* TTS preset manager and provider hot-swap UX.
* The WebSocket transport itself.

### Transport boundary

The framework defines a generic `Transport` protocol and ships a
`QueueTransport` for tests. The consumer app owns the real
WebSocket transport — `WebSocketTransport` (or equivalent) lives in
the app, not in the framework, so the framework never takes a hard
dependency on FastAPI. The reference harness exposes
`conversational_harness.transport.WebSocketTransport` for that
purpose.

## Status

The package is in v0.1 pre-release. The test matrix below is the
current contract:

| Surface | Tests |
|---|---:|
| `converse_framework` (base) | 126 |
| Reference harness (`Reference-Repository-Conversational-AI-Harness`) | 91 passed, 1 skipped |

Run them locally:

```bash
# Framework (run from the package root)
python -m pytest

# Harness (run from inside the harness directory)
python -m pytest
```
