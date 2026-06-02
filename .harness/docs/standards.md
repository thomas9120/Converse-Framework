# Converse Framework — Coding & Test Standards

This file is shared across all reins. Link to it from each rein's
`agent.md` instead of inlining the rules. If a rule needs to change,
edit this file once.

## Python & packaging

- Python 3.11+ is the floor. `pyproject.toml` declares it; do not
  add syntax that requires a newer interpreter.
- Build backend is `hatchling` (already configured). Do not switch
  build systems.
- Base install dependencies must stay at the minimum. The only
  required runtime dep is `numpy>=2.0`. Heavy provider backends
  (`silero-vad`, `faster-whisper`, `httpx`, `kokoro-onnx`, `misaki`,
  `pocket-tts`) live under `[project.optional-dependencies]`.
- License is MIT. Keep `LICENSE` at the repo root unchanged.

## Public API contract

- The framework's public API is the explicit `__all__` in
  `converse_framework/__init__.py`. Anything not in `__all__` is
  internal and may change without notice.
- Adding a public symbol: update `__all__`, add a public docstring,
  add a test in `tests/`.
- Removing or renaming a public symbol: bump the version in
  `pyproject.toml` AND `converse_framework/__init__.py.__version__`,
  note the change in the changelog, and (if past v0.1) write a
  deprecation shim.

## Lightweight base import

`python -c "import converse_framework; print(converse_framework.__all__)"`
must succeed with only `numpy` installed. No `import` statement
anywhere in `converse_framework/__init__.py` or its eagerly-loaded
submodules may pull in `fastapi`, `httpx`, `silero_vad`,
`faster_whisper`, `kokoro_onnx`, `misaki`, or `pocket_tts`.

Heavy deps load lazily through the registry's import-string mechanism
(see `converse_framework/registry.py`).

## Framework / app boundary

The framework does NOT own:

- FastAPI app, REST endpoints, WebSocket handler, browser UI
- Profile file loading, runtime settings persistence
- Character card parsing, first-message seeding
- Companion mode policy, memory store
- TTS preset manager, provider hot-swap UX
- `runtime_settings.py`, `config.py`, `tts_runtime.py`
- The `WebSocketTransport` implementation
- The `Reference-Repository-Conversational-AI-Harness/` consumer
  app itself

App behavior reaches the framework through injected callables or
plain config only. The framework defines callables such as
`system_prompt_builder` and `sampler_builder` — the app supplies
them; the framework never imports app code.

## Event wire format

- Event wire shape is `{"type": str, "ts": float, "payload": dict}`
  for v0.1. Do not break this.
- `EventSink.emit(event_type: str, **payload)` is the sink API.
- `FrameworkEvent` is the canonical event class. `HarnessEvent`
  remains a compatibility alias for one minor version.

## Provider protocol rules

- Concrete providers implement protocols from
  `converse_framework.protocols` (`VADProvider`, `ASRProvider`,
  `LLMProvider`, `TTSProvider`).
- They are registered lazily via import strings in `registry.py`:
  `register_provider("vad", "silero", "converse_framework.providers.silero:SileroVADProvider")`.
- A missing dependency must surface as a friendly message produced
  by `extra_hint_for("silero")` (or whichever extra), not a bare
  `ImportError`.
- Provider constructors accept a config mapping. They must not
  reach back into harness `RuntimeSettings` or any app module.

## Testing

- Test runner is `pytest`. Test paths: `tests/`.
- Coverage style: deterministic, no real I/O. Use `QueueEventSink`
  and `QueueTransport` from the framework itself to capture events.
- Use mock providers for unit tests by default. Real provider
  tests must be guarded by a dep-presence skip.
- Add a test for every public symbol change. Existing nine test
  files: `test_audio_utils.py`, `test_events.py`, `test_examples.py`,
  `test_pipeline.py`, `test_protocols.py`, `test_providers.py`,
  `test_registry.py`, `test_transport.py`, `test_utterance_collector.py`.
- Lint with `ruff` (cache directories `.ruff_cache` are already
  present). Match the existing rule set; do not introduce new
  rule categories without discussion.

## Release prep (Phase 7 of plan.md)

- `python -m build` must succeed. `python -m twine check dist/*`
  must pass.
- README must include: minimal mock text pipeline, audio frame →
  utterance collector → pipeline, custom provider registration,
  custom event sink.
- Document the extras and the missing-dependency behavior.
- Add a migration note for the reference harness consumer.
- At least one non-harness consumer must run end-to-end (the
  `converse_framework.examples.text_chat` driver is the canonical
  second consumer).

## Git workflow

- One task = one commit. Multi-phase work should land in separate
  commits so the project history stays readable.
- Do not amend commits. Do not force-push.
- The `.harness/` directory IS the team definition and should be
  committed to git.
- Reference `plan.md` checkboxes by `[x]` when a phase item closes.
