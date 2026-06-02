## Initial implementation notes

- The framework makes the app/framework boundary easy to follow: `SpeechPipeline`,
  `AudioUtteranceCollector`, provider protocols, and event sinks were enough to
  build a browser-based consumer without modifying framework code.
- The plan requested whisper.cpp, but the registry currently ships
  `faster-whisper` only. The test app uses an app-local `WhisperCppASRProvider`
  that shells out to `whisper-cli`/`main`; a first-class whisper.cpp provider or
  recipe would make this path much smoother.
- The voice example is useful, but it reads WAV bytes as hex even though
  `parse_audio_frame` documents and enforces base64. The README recipe uses
  base64 correctly.
- The framework forwards Kokoro `pcm_s16le` chunks cleanly. Browser consumers
  still need a small PCM playback helper because there is no packaged JS
  transport/client reference in this repo.
- Provider status is helpful for missing dependencies, but app code still has to
  construct the bundle before checking statuses. A lightweight "status-only"
  helper that accepts raw config could simplify settings screens.
- The framework/package docs should call out Python version compatibility for
  concrete providers. `kokoro-onnx==0.5.0` currently requires Python
  `>=3.10,<3.14`, so running this stack under Python 3.14 fails at install time.
