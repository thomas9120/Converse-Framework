# converse-framework feedback (found while building this POC, 2026-07-10)

Issues found in the released `converse-framework` wheel while building the
converse-concept voice POC on Windows. File paths refer to the installed
package (`.venv/Lib/site-packages/converse_framework/`). C:\Users\pegas\Desktop\LLama\Projects 2026\converse-concept

## 1. `BrowserVoiceClient.close()` calls a method that doesn't exist

`js/browser-voice-client.js` line ~123:

```js
close: function () {
    this.stop();
    if (this._player) {
        this._player.clear();   // <-- TtsAudioPlayer has no clear()
    }
    ...
}
```

`TtsAudioPlayer` only exposes `onEvent()`, `flush()`, and `close()`, so any
app that calls `BrowserVoiceClient.close()` gets a TypeError. Probably
meant `this._player.close()`.

## 2. `TtsAudioPlayer` has no way to cancel scheduled audio (barge-in gap)

Related to the above: once PCM has been flushed, the
`AudioBufferSourceNode`s are `start()`ed and not tracked
(`js/tts-audio-player.js`, `_scheduleAudioBuffer`), so there is no public
way to stop playback that is already scheduled. `close()` only stops
*accepting* new events — audio already handed to the AudioContext keeps
playing.

This matters for barge-in and for a "stop speaking" button: when the
server emits `tts.cancelled`, the client should be able to silence
immediately. The POC works around it by discarding the whole player and
closing its AudioContext (touching the private `_ctx`):

```js
function resetPlayer() {
  const old = player;
  player = new TtsAudioPlayer({ coalesceMs: 80 });
  old.close();
  if (old._ctx) old._ctx.close().catch(() => {});
}
```

Suggested fix: track live source nodes and add a public `cancel()`/`clear()`
that stops them and resets `_nextStartTime`.

## 3. `cuda_utils.add_nvidia_dll_directories()` is not enough for ctranslate2 on Windows

`cuda_utils.py` registers the pip-installed NVIDIA wheel dirs
(`nvidia/cublas/bin`, `nvidia/cudnn/bin`, ...) with
`os.add_dll_directory()`. That works for DLLs loaded through Python, but
ctranslate2 (faster-whisper backend) resolves `cublas64_12.dll` at
*inference time* with a plain `LoadLibrary` call that only searches the
application dir and `PATH` — the `add_dll_directory` entries are ignored.

Observed: `faster-whisper` provider on `device=auto/cuda` loads the model
fine, then every transcription fails with
`Library cublas64_12.dll is not found or cannot be loaded`, even though
the log shows `Added DLL directory: ...nvidia\cublas\bin`.

Suggested fix in `add_nvidia_dll_directories()`: also prepend the
discovered dirs to `os.environ["PATH"]`. Working app-side shim (see
`app/main.py:_extend_path_with_nvidia_dlls` in this repo):

```python
from converse_framework.cuda_utils import discover_nvidia_dll_dirs
dirs = [str(d) for d in discover_nvidia_dll_dirs()]
if dirs:
    os.environ["PATH"] = os.pathsep.join(dirs + [os.environ.get("PATH", "")])
```

## 4. `SpeakerEchoGuard` resumes on event arrival, not playback completion

`speaker-echo-guard.js` schedules its resume `tailDelayMs` (default 350 ms)
after the last `tts.audio` event — but events arrive as fast as synthesis
streams, while `TtsAudioPlayer` schedules that PCM *ahead* on the
AudioContext. For any reply longer than a couple of seconds, playback
outlives the final event by seconds, so the mic unsuppresses while the
speaker is still talking and the echo loop returns. The docstring example
(player + guard fed from the same `ws.onmessage`) has this problem as-is.

App-side workaround used here (`static/app.js`): swallow the `final`
marker, compute the real remaining playback from the player
(`(_nextStartTime - ctx.currentTime) * 1000`), and only trigger the guard's
tail once that has drained. Suggested framework fix: let the guard accept a
`TtsAudioPlayer` reference (or a `remainingMs()` callback) and base the
resume on scheduled-playback drain. `turn.finished` as a resume trigger has
the same flaw — it can even arrive *before* the last `tts.audio` chunk.

## 5. (Minor) PyPI long-description looks stale

The PyPI page doesn't mention the `audio-cpp` (TTS + ASR) or `pocket-tts`
providers even though the wheel registers them in `registry.py` — I
initially planned to write a custom TTS provider before discovering the
built-in one. A registry/README refresh on the next release would save
integrators the same detour.
