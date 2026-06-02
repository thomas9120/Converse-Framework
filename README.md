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
pip install converse-framework[llamacpp]        # llama.cpp HTTP LLM
pip install converse-framework[kokoro]          # Kokoro ONNX TTS
pip install converse-framework[pocket-tts]      # Pocket TTS
pip install converse-framework[all]             # everything
```

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
