# Improvements

Working checklist of low-hanging improvements identified in the July 2026 review.
Ranked by impact-per-effort. Mark items off as they land.

## Tier 1 — cheap, disproportionate payoff

- [x] **1. Generalize llama.cpp provider into an "OpenAI-compatible" LLM provider**
  (done: `converse_framework/providers/openai_compat.py`, `openai-compat` extra,
  `api_key` support on both providers, tests in `tests/test_openai_compat.py`)
  - `converse_framework/providers/llamacpp.py` already speaks the OpenAI
    chat-completions protocol with a configurable `base_url`.
  - Add an optional `api_key` config option → `Authorization: Bearer` header.
  - Register an `"openai-compatible"` alias in the registry (keep `"llamacpp"`
    working as-is).
  - The `/health` endpoint is llama.cpp-specific; the generalized status probe
    should fall back to `/v1/models` alone when `/health` is absent.
  - Payoff: one provider covers Ollama, LM Studio, vLLM, Groq, OpenRouter,
    Together, and actual OpenAI.

- [ ] **2. OpenAI-compatible ASR + TTS providers**
  - ASR: `/v1/audio/transcriptions` (multipart WAV upload). Covers OpenAI
    Whisper, Groq hosted Whisper, local servers like `speaches` /
    faster-whisper-server. Copy the httpx provider pattern from
    `converse_framework/providers/whisper_cpp.py`.
  - TTS: `/v1/audio/speech`. Covers OpenAI TTS, Kokoro-FastAPI,
    openedai-speech. Copy the pattern from
    `converse_framework/providers/audio_cpp.py`.
  - Payoff: local-first stays the default; cloud becomes possible. Removes the
    "local-only catalog" adoption objection.

- [ ] **3. Eager first TTS chunk (time-to-first-audio)**
  - `should_flush_tts` (`converse_framework/pipeline.py`) uses one threshold
    (`tts_chunk_chars=120` or sentence punctuation) for every chunk.
  - Add a `first_chunk_chars` field to `PipelineConfig` with a much lower
    threshold for the first chunk of a turn (~30-40 chars or first clause
    boundary / comma), then revert to normal chunking.
  - Needs a "have I flushed yet this turn" flag in `_stream_llm_and_tts`.
  - Payoff: voice starts noticeably sooner; steady-state audio unchanged.

- [ ] **4. `turn.metrics` summary event**
  - Individual events already carry `latency_ms`, but reconstructing a turn
    timeline from the event stream is painful.
  - Emit one event alongside `turn.finished` with `asr_ms`,
    `llm_first_token_ms`, `tts_first_chunk_ms`, and `total_ms`.
  - All timestamps already exist inside the pipeline; this is bookkeeping.
  - Payoff: makes the latency story measurable — prerequisite to improving it.

- [ ] **5. CI test workflow**
  - `.github/workflows/publish.yml` exists but nothing runs pytest on push.
  - Add a test workflow with a Python 3.11 / 3.12 / 3.13 matrix (~25 lines of
    YAML). Optionally add a status badge to the README.
  - Payoff: cheapest possible answer to the bus-factor/maturity concern.

## Tier 2 — moderate effort, real wins

- [ ] **6. Binary WebSocket frames for audio**
  - Base64-JSON per 30 ms mic frame costs ~33% bandwidth plus per-frame JSON
    encode/parse on both sides.
  - Accept raw binary WS messages for mic frames (few header bytes for
    sequence / sample rate, then PCM); optionally send `tts.audio` payloads as
    binary too. Keep JSON for control messages.
  - Touches `converse_framework/js/mic-frame-sender.js`,
    `parse_audio_frame` (`converse_framework/audio_utils.py`), and the session
    dispatch (`converse_framework/session.py`).
  - Needs a versioned format so existing JSON clients keep working.

## Explicitly out of scope (not low-hanging)

These are the remaining architectural gaps versus capital-backed frameworks
(Pipecat, LiveKit Agents). Documented so we don't re-litigate them:

- **Streaming ASR partials feeding the LLM early** — requires incremental
  Whisper windowing, hypothesis revision, and speculative LLM cancellation.
  A rewrite of the turn model, not a patch.
- **WebRTC transport** — aiortc, ICE/STUN, jitter buffers. Weeks of work and a
  heavy new dependency.
- **Semantic turn detection / real AEC** — model-training territory. The JS
  mic sender already requests browser `echoCancellation: true`, which is the
  right call at this scale.
