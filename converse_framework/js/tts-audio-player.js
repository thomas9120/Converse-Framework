/**
 * tts-audio-player.js — browser reference client for converse_framework `tts.audio` events.
 *
 * The framework emits TTS audio on `tts.audio` events with this wire shape:
 *
 *   {
 *     "type": "tts.audio",
 *     "ts": 1234567890.123,
 *     "payload": {
 *       "data": "<base64 PCM s16le bytes>",
 *       "encoding": "pcm_s16le",
 *       "sample_rate": 24000,
 *       "channels": 1,
 *       "duration_ms": 240,
 *       "final": false
 *     }
 *   }
 *
 * Why this file exists:
 *
 *   1. The framework only ships the Python side. Browser consumers have
 *      to write their own glue to turn `tts.audio` events into sound.
 *
 *   2. Calling `AudioContext.decodeAudioData` on a stream of tiny chunks
 *      (e.g. raw WAV blobs the model emits per phrase) is the classic
 *      cause of choppy / stuttering TTS playback. The fix is to build
 *      `AudioBuffer`s directly from PCM s16le bytes and coalesce
 *      consecutive chunks before scheduling them.
 *
 *   3. The same fix that resolved Pocket TTS choppiness in the harness
 *      (per the harness AGENTS.md) generalises: always carry explicit
 *      audio metadata, never decode tiny chunks, always coalesce.
 *
 * Public surface:
 *
 *   const player = new TtsAudioPlayer();
 *   ws.addEventListener('message', (ev) => {
 *     const event = JSON.parse(ev.data);
 *     if (event.type === 'tts.audio') player.onEvent(event);
 *     if (event.type === 'tts.cancelled') player.cancel(); // barge-in
 *   });
 *   player.remainingMs();  // scheduled audio still to play
 *   // when the conversation ends (also silences scheduled audio):
 *   player.close();
 *
 * The class is exported as `window.TtsAudioPlayer` in the browser and
 * as a CommonJS module export under Node (for unit tests). No build
 * step is required; copy the file into your static assets and load it
 * with a plain <script> tag.
 */
(function (root, factory) {
  const exported = factory();
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exported;
  }
  if (typeof root !== 'undefined') {
    root.TtsAudioPlayer = exported.TtsAudioPlayer;
  }
})(typeof window !== 'undefined' ? window : globalThis, function () {
  'use strict';

  /**
   * Browser reference client for converse_framework `tts.audio` events.
   *
   * @param {object} [opts]
   * @param {AudioContext} [opts.audioContext] Reuse an existing context.
   *   A new context is created from the first chunk's sample rate when omitted.
   * @param {number} [opts.coalesceMs=80] Maximum time to wait before
   *   flushing the coalescing buffer with whatever chunks are queued.
   * @param {number} [opts.maxCoalesceBytes=32768] Maximum bytes to
   *   coalesce before forcing a flush. Avoids building a single huge
   *   AudioBuffer when audio is dense.
   */
  class TtsAudioPlayer {
    constructor(opts) {
      opts = opts || {};
      this._ctx = opts.audioContext || null;
      this._coalesceMs = (typeof opts.coalesceMs === 'number') ? opts.coalesceMs : 80;
      this._maxCoalesceBytes = (typeof opts.maxCoalesceBytes === 'number')
        ? opts.maxCoalesceBytes
        : 32768;
      this._channels = 1;
      this._buffer = [];
      this._bufferBytes = 0;
      this._flushTimer = null;
      this._closed = false;
      this._nextStartTime = 0;
      this._sources = [];
    }

    /**
     * Handle a `tts.audio` event from the framework. Decodes the
     * base64 PCM s16le payload, appends it to the coalescing buffer,
     * and schedules a flush when the buffer is full or the time
     * window expires.
     *
     * @param {object} event The event envelope as emitted by the framework.
     */
    onEvent(event) {
      if (this._closed) return;
      if (!event) return;
      if (event.type === 'tts.cancelled') {
        // Barge-in: the server abandoned this synthesis, so audio that
        // is already scheduled must be silenced, not played out.
        this.cancel();
        return;
      }
      if (event.type !== 'tts.audio') return;
      const payload = event.payload || {};
      if (payload.encoding && payload.encoding !== 'pcm_s16le') {
        // The framework only ships pcm_s16le today. Any other encoding
        // would need a different decoder; surface it loudly.
        console.warn('tts-audio-player: unsupported encoding', payload.encoding);
        return;
      }
      const sampleRate = payload.sample_rate || 24000;
      const channels = payload.channels || 1;
      this._ensureContext(sampleRate, channels);
      if (!payload.data) {
        return;
      }
      const bytes = _base64ToBytes(payload.data);
      this._buffer.push(bytes);
      this._bufferBytes += bytes.byteLength;
      const isFinal = !!payload.final;
      if (isFinal || this._bufferBytes >= this._maxCoalesceBytes) {
        this._flush();
      } else {
        this._scheduleFlush();
      }
    }

    /** Flush any pending coalesced audio immediately. */
    flush() {
      this._flush();
    }

    /**
     * Stop all playback immediately: silences audio that is already
     * scheduled on the AudioContext, discards any pending coalesced
     * chunks, and resets the schedule clock. The player keeps accepting
     * new `tts.audio` events afterwards, so this is the right call for
     * barge-in and for handling `tts.cancelled`.
     */
    cancel() {
      if (this._flushTimer) {
        clearTimeout(this._flushTimer);
        this._flushTimer = null;
      }
      this._buffer = [];
      this._bufferBytes = 0;
      const sources = this._sources;
      this._sources = [];
      for (let i = 0; i < sources.length; i++) {
        try {
          sources[i].onended = null;
          sources[i].stop();
        } catch (_) {
          // Already stopped / never started — nothing to silence.
        }
      }
      this._nextStartTime = 0;
    }

    /** Alias for {@link cancel}. */
    clear() {
      this.cancel();
    }

    /**
     * Milliseconds of scheduled audio still to play, based on the
     * AudioContext clock. Returns 0 when nothing is scheduled. Lets
     * consumers (e.g. SpeakerEchoGuard) wait for real playback drain
     * instead of the last `tts.audio` event, which can arrive seconds
     * before the speaker goes quiet.
     *
     * @returns {number}
     */
    remainingMs() {
      if (!this._ctx || !this._nextStartTime) return 0;
      const remaining = (this._nextStartTime - this._ctx.currentTime) * 1000;
      return remaining > 0 ? remaining : 0;
    }

    /**
     * Stop playback and stop accepting events. Releases the coalescing
     * timer and silences any audio still scheduled on the AudioContext.
     */
    close() {
      this.cancel();
      this._closed = true;
    }

    _ensureContext(sampleRate, channels) {
      if (!this._ctx) {
        const Ctor = (typeof window !== 'undefined'
          ? (window.AudioContext || window.webkitAudioContext)
          : null);
        if (!Ctor) {
          throw new Error('tts-audio-player: no AudioContext constructor available');
        }
        this._ctx = new Ctor({ sampleRate: sampleRate });
      }
      if (this._ctx.sampleRate !== sampleRate || this._channels !== channels) {
        // The browser cannot resample through createBuffer, so the
        // consumer must match the TTS provider's output rate. A
        // mismatch here usually means the conversation crossed a
        // profile switch; the right fix is to recreate the player.
        console.warn(
          'tts-audio-player: sample rate / channel count changed; recreating context',
          { from: this._ctx.sampleRate, to: sampleRate, fromCh: this._channels, toCh: channels }
        );
        this._ctx = new (typeof window !== 'undefined'
          ? (window.AudioContext || window.webkitAudioContext)
          : globalThis.AudioContext)({ sampleRate: sampleRate });
      }
      this._channels = channels;
    }

    _scheduleFlush() {
      if (this._flushTimer) return;
      this._flushTimer = setTimeout(() => this._flush(), this._coalesceMs);
    }

    _flush() {
      if (this._flushTimer) {
        clearTimeout(this._flushTimer);
        this._flushTimer = null;
      }
      if (!this._buffer.length || !this._ctx) {
        this._buffer = [];
        this._bufferBytes = 0;
        return;
      }
      const merged = _concatBytes(this._buffer);
      this._buffer = [];
      this._bufferBytes = 0;
      this._scheduleAudioBuffer(merged);
    }

    _scheduleAudioBuffer(bytes) {
      const ctx = this._ctx;
      const channels = this._channels;
      // 16-bit signed little-endian = 2 bytes per sample per channel.
      const totalSamples = Math.floor(bytes.byteLength / 2);
      if (totalSamples === 0) return;
      const audioBuffer = ctx.createBuffer(channels, totalSamples, ctx.sampleRate);
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      for (let ch = 0; ch < channels; ch++) {
        const channelData = audioBuffer.getChannelData(ch);
        for (let i = 0; i < totalSamples; i++) {
          const sample = view.getInt16(i * 2, true); // little-endian
          // Map -32768..32767 to -1.0..1.0; both endpoints preserved.
          channelData[i] = sample < 0 ? sample / 32768 : sample / 32767;
        }
      }
      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(ctx.destination);
      const now = ctx.currentTime;
      const startAt = Math.max(now, this._nextStartTime);
      this._sources.push(source);
      source.onended = () => {
        const idx = this._sources.indexOf(source);
        if (idx !== -1) this._sources.splice(idx, 1);
      };
      source.start(startAt);
      this._nextStartTime = startAt + audioBuffer.duration;
    }
  }

  function _base64ToBytes(b64) {
    const binary = (typeof atob !== 'undefined') ? atob(b64) : Buffer.from(b64, 'base64').toString('binary');
    const len = binary.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
  }

  function _concatBytes(chunks) {
    let total = 0;
    for (let i = 0; i < chunks.length; i++) {
      total += chunks[i].byteLength;
    }
    const out = new Uint8Array(total);
    let offset = 0;
    for (let i = 0; i < chunks.length; i++) {
      out.set(chunks[i], offset);
      offset += chunks[i].byteLength;
    }
    return out;
  }

  return { TtsAudioPlayer: TtsAudioPlayer };
});
