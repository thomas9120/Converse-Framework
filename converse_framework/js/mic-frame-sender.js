/**
 * mic-frame-sender.js — browser microphone capture and frame sender for
 * converse_framework ``audio.frame`` events.
 *
 * Captures microphone input via the Web Audio API, downsamples to the
 * target sample rate, slices into fixed-size PCM s16le frames, and
 * sends them as JSON/base64 or opt-in binary packets over a WebSocket.
 *
 * ```html
 * <script src="mic-frame-sender.js"></script>
 * <script>
 *   const ws = new WebSocket("ws://localhost:8000/ws");
 *   const mic = new MicFrameSender({ webSocket: ws });
 *   document.getElementById("start-btn").onclick = () => mic.start();
 *   document.getElementById("stop-btn").onclick = () => mic.stop();
 * </script>
 * ```
 *
 * Mobile microphone access requires HTTPS, localhost, or a secure tunnel
 * (Cloudflare Tunnel, ngrok, etc.).  Plain ``http://<lan-ip>`` will be
 * rejected by ``getUserMedia`` on most mobile browsers.
 *
 * @module
 */

((root, factory) => {
	if (typeof define === "function" && define.amd) {
		define([], factory);
	} else if (typeof module === "object" && module.exports) {
		module.exports = factory();
	} else {
		root.MicFrameSender = factory();
	}
})(this, () => {
	// -----------------------------------------------------------------------
	// Constants
	// -----------------------------------------------------------------------

	const ENCODING = "pcm_s16le";
	const BINARY_AUDIO_MAGIC_0 = 0x43; // C
	const BINARY_AUDIO_MAGIC_1 = 0x46; // F
	const BINARY_AUDIO_VERSION = 1;
	const BINARY_AUDIO_KIND_MIC = 1;
	const BINARY_AUDIO_HEADER_SIZE = 16;

	// -----------------------------------------------------------------------
	// Utilities
	// -----------------------------------------------------------------------

	/**
	 * Downsample an interleaved float32 array from deviceRate to targetRate.
	 * Simple linear interpolation.
	 */
	function downsampleFloat32(samples, deviceRate, targetRate) {
		if (deviceRate === targetRate) return samples;
		const ratio = deviceRate / targetRate;
		const outLen = Math.round(samples.length / ratio);
		const out = new Float32Array(outLen);
		for (let i = 0; i < outLen; i++) {
			const srcIdx = i * ratio;
			const lo = Math.floor(srcIdx);
			const hi = Math.min(lo + 1, samples.length - 1);
			const frac = srcIdx - lo;
			// Clamp to [-1, 1] to avoid overflow on conversion
			out[i] = Math.max(
				-1,
				Math.min(1, samples[lo] * (1 - frac) + samples[hi] * frac),
			);
		}
		return out;
	}

	/**
	 * Convert a Float32Array of samples to a PCM s16le ArrayBuffer.
	 * Assumes samples in [-1, 1] range.
	 */
	function float32ToPcmS16le(samples) {
		const len = samples.length;
		const buf = new ArrayBuffer(len * 2);
		const view = new DataView(buf);
		for (let i = 0; i < len; i++) {
			// Clamp to [-1, 1] then scale to int16
			const s = Math.max(-1, Math.min(1, samples[i]));
			const val = s < 0 ? s * 0x8000 : s * 0x7fff;
			view.setInt16(i * 2, Math.round(val), true); // little-endian
		}
		return buf;
	}

	/**
	 * Convert an ArrayBuffer to a base64-encoded string.
	 */
	function arrayBufferToBase64(buf) {
		const bytes = new Uint8Array(buf);
		let binary = "";
		for (let i = 0; i < bytes.length; i++) {
			binary += String.fromCharCode(bytes[i]);
		}
		return btoa(binary);
	}

	/**
	 * Encode one raw PCM frame using the converse_framework binary audio v1
	 * packet format. Header integers use network byte order; PCM remains s16le.
	 */
	function encodeBinaryAudioFrameV1(pcmBuf, metadata) {
		if (!(pcmBuf instanceof ArrayBuffer) || pcmBuf.byteLength === 0) {
			throw new Error("binary audio PCM must be a non-empty ArrayBuffer");
		}
		metadata = metadata || {};
		const sequence = Number(metadata.sequence);
		const sampleRate = Number(metadata.sample_rate);
		const channels = Number(metadata.channels);
		const frameMs = Number(metadata.frame_ms);
		if (!Number.isInteger(sequence) || sequence < 0 || sequence > 0xffffffff) {
			throw new Error("binary audio sequence must be a uint32");
		}
		if (!Number.isInteger(sampleRate) || sampleRate <= 0 || sampleRate > 0xffffffff) {
			throw new Error("binary audio sample_rate must be a uint32");
		}
		if (!Number.isInteger(channels) || channels <= 0 || channels > 0xff) {
			throw new Error("binary audio channels must be a uint8");
		}
		if (!Number.isInteger(frameMs) || frameMs <= 0 || frameMs > 0xffff) {
			throw new Error("binary audio frame_ms must be a uint16");
		}

		const modeBytes = new TextEncoder().encode(String(metadata.mode || ""));
		if (modeBytes.byteLength > 0xff) {
			throw new Error("binary audio mode must be at most 255 UTF-8 bytes");
		}

		const packet = new ArrayBuffer(
			BINARY_AUDIO_HEADER_SIZE + modeBytes.byteLength + pcmBuf.byteLength,
		);
		const view = new DataView(packet);
		view.setUint8(0, BINARY_AUDIO_MAGIC_0);
		view.setUint8(1, BINARY_AUDIO_MAGIC_1);
		view.setUint8(2, BINARY_AUDIO_VERSION);
		view.setUint8(3, BINARY_AUDIO_KIND_MIC);
		view.setUint32(4, sequence, false);
		view.setUint32(8, sampleRate, false);
		view.setUint8(12, channels);
		view.setUint16(13, frameMs, false);
		view.setUint8(15, modeBytes.byteLength);
		const bytes = new Uint8Array(packet);
		bytes.set(modeBytes, BINARY_AUDIO_HEADER_SIZE);
		bytes.set(
			new Uint8Array(pcmBuf),
			BINARY_AUDIO_HEADER_SIZE + modeBytes.byteLength,
		);
		return packet;
	}

	/**
	 * Number of samples for N ms at a given sample rate.
	 */
	function msToSamples(ms, rate) {
		return Math.round((ms / 1000) * rate);
	}

	// -----------------------------------------------------------------------
	// AudioWorklet processor (inline via blob URL)
	// -----------------------------------------------------------------------

	let _processorUrl = null;

	function getProcessorUrl() {
		if (_processorUrl) return _processorUrl;
		const code = [
			"class CaptureProcessor extends AudioWorkletProcessor {",
			"  process(inputs, outputs, params) {",
			"    const input = inputs[0];",
			"    if (input && input[0] && input[0].length > 0) {",
			"      this.port.postMessage(input[0]);",
			"    }",
			"    return true;",
			"  }",
			"}",
			"registerProcessor('mic-capture-processor', CaptureProcessor);",
		].join("\n");
		const blob = new Blob([code], { type: "application/javascript" });
		_processorUrl = URL.createObjectURL(blob);
		return _processorUrl;
	}

	// -----------------------------------------------------------------------
	// MicFrameSender
	// -----------------------------------------------------------------------

	/**
	 * Create a microphone frame sender.
	 *
	 * @param {Object} options
	 * @param {WebSocket} options.webSocket  - target WebSocket.
	 * @param {number}  [options.sampleRate=16000] - target sample rate.
	 * @param {number}  [options.channels=1]       - output channel count.
	 * @param {number}  [options.frameMs=30]       - frame duration in ms.
	 * @param {string}  [options.mode="chat"]      - conversation mode tag.
	 * @param {string}  [options.messageType="audio.frame"] - WebSocket message type.
	 * @param {string}  [options.frameFormat="json"] - "json" or opt-in "binary-v1".
	 * @param {function(number):void} [options.onLevel] - level callback (0-1).
	 * @param {function(Error):void}  [options.onError]  - error callback.
	 * @param {AudioContext} [options.audioContext]  - shared AudioContext (created if omitted).
	 * @param {function(Object):boolean} [options.shouldSendFrame] - optional gate; called with payload, return false to drop.
	 */
	function MicFrameSender(options) {
		options = options || {};
		this._ws = options.webSocket || null;
		this._targetRate = options.sampleRate || 16000;
		this._channels = options.channels || 1;
		this._frameMs = options.frameMs || 30;
		this._mode = options.mode || "chat";
		this._messageType = options.messageType || "audio.frame";
		this._frameFormat = options.frameFormat || "json";
		if (this._frameFormat !== "json" && this._frameFormat !== "binary-v1") {
			throw new Error('frameFormat must be "json" or "binary-v1"');
		}
		this._onLevel = options.onLevel || null;
		this._onError = options.onError || null;
		this._audioContext = options.audioContext || null;
		this._shouldSendFrame = options.shouldSendFrame || null;

		// Owned AudioContext (if none provided)
		this._ownedCtx = null;

		// Active stream / nodes / worklet
		this._stream = null;
		this._source = null;
		this._workletNode = null;
		this._scriptProcessor = null;

		// Frame sequencing
		this._sequence = 0;
		this._accumulator = new Float32Array(0);
		this._frameSamples = msToSamples(this._frameMs, this._targetRate);
		this._running = false;
		this._paused = false;

		// Worklet availability
		this._workletSupported =
			typeof AudioWorkletNode !== "undefined" &&
			typeof AudioContext !== "undefined";
	}

	MicFrameSender.prototype = {
		constructor: MicFrameSender,

		// -----------------------------------------------------------------------
		// Public API
		// -----------------------------------------------------------------------

		/**
		 * Start capturing and sending mic frames.
		 * @returns {Promise<void>}
		 */
		start: async function () {
			if (this._running) return;
			this._paused = false;

			try {
				const stream = await navigator.mediaDevices.getUserMedia({
					audio: {
						sampleRate: { ideal: this._targetRate },
						channelCount: { ideal: this._channels },
						echoCancellation: true,
						noiseSuppression: true,
					},
				});
				this._stream = stream;

				const ctx =
					this._audioContext ||
					new (window.AudioContext || window.webkitAudioContext)();
				if (!this._audioContext) {
					this._ownedCtx = ctx;
				}

				// Determine device sample rate from AudioContext
				const deviceRate = ctx.sampleRate;

				// Ensure context is running (needed after autoplay policy)
				if (ctx.state === "suspended") {
					await ctx.resume();
				}

				const source = ctx.createMediaStreamSource(stream);
				this._source = source;

				// Try AudioWorklet first, fall back to ScriptProcessorNode
				try {
					await this._setupWorkletNode(ctx, source, deviceRate);
				} catch (_) {
					this._setupScriptProcessor(ctx, source, deviceRate);
				}

				this._running = true;
			} catch (err) {
				this._safeError(err);
				throw err;
			}
		},

		/**
		 * Stop capturing and release resources.
		 */
		stop: function () {
			this._running = false;
			this._paused = false;
			this._sequence = 0;

			this._teardownNodes();
			this._teardownStream();
			this._teardownContext();
		},

		/**
		 * Update the conversation mode tag sent with each frame.
		 * @param {string} mode
		 */
		setMode: function (mode) {
			this._mode = String(mode);
		},

		/**
		 * Replace the target WebSocket.  Can be called while running.
		 * @param {WebSocket|null} ws
		 */
		setWebSocket: function (ws) {
			this._ws = ws;
		},

		/**
		 * Release all resources (alias for stop).
		 */
		close: function () {
			this.stop();
		},

		// -----------------------------------------------------------------------
		// Internal: AudioWorklet branch
		// -----------------------------------------------------------------------

		_setupWorkletNode: async function (ctx, source, deviceRate) {
			const url = getProcessorUrl();
			await ctx.audioWorklet.addModule(url);
			const workletNode = new AudioWorkletNode(ctx, "mic-capture-processor");
			this._workletNode = workletNode;

			source.connect(workletNode);

			workletNode.port.onmessage = (evt) => {
				if (!this._running || this._paused) return;
				const floatSamples = evt.data;
				this._processAudio(floatSamples, deviceRate);
			};

			workletNode.connect(ctx.destination);
		},

		// -----------------------------------------------------------------------
		// Internal: ScriptProcessorNode fallback
		// -----------------------------------------------------------------------

		_setupScriptProcessor: function (ctx, source, deviceRate) {
			// Buffer size = one frame at device rate, rounded to power-of-2
			const bufSize = this._nextPow2(msToSamples(this._frameMs, deviceRate));
			const processor = ctx.createScriptProcessor(bufSize, 1, 1);
			this._scriptProcessor = processor;

			source.connect(processor);
			processor.connect(ctx.destination);

			processor.onaudioprocess = (evt) => {
				if (!this._running || this._paused) return;
				const input = evt.inputBuffer.getChannelData(0);
				this._processAudio(input, deviceRate);
			};
		},

		// -----------------------------------------------------------------------
		// Internal: audio processing pipeline
		// -----------------------------------------------------------------------

		_processAudio: function (floatSamples, deviceRate) {
			// Downsample to target rate
			const downsampled = downsampleFloat32(
				floatSamples,
				deviceRate,
				this._targetRate,
			);

			// Append to accumulator
			const acc = this._accumulator;
			const combined = new Float32Array(acc.length + downsampled.length);
			combined.set(acc);
			combined.set(downsampled, acc.length);
			this._accumulator = combined;

			// Slice and send complete frames
			const frameSize = this._frameSamples;
			while (this._accumulator.length >= frameSize) {
				const frame = this._accumulator.slice(0, frameSize);
				this._accumulator = this._accumulator.slice(frameSize);
				this._sendFrame(frame);
			}
		},

		_sendFrame: function (frame) {
			// Report level
			if (this._onLevel) {
				let sum = 0;
				for (let i = 0; i < frame.length; i++) {
					sum += frame[i] * frame[i];
				}
				const rms = Math.sqrt(sum / frame.length);
				this._onLevel(Math.min(1, rms * 3));
			}

			const pcmBuf = float32ToPcmS16le(frame);
			const jsonData =
				this._frameFormat === "json" ? arrayBufferToBase64(pcmBuf) : null;
			const payload = {
				type: this._messageType,
				payload: {
					data: jsonData,
					encoding: ENCODING,
					sample_rate: this._targetRate,
					channels: this._channels,
					frame_ms: this._frameMs,
					sequence: this._sequence++,
					mode: this._mode,
				},
			};

			// Optional gate (speaker echo guard)
			if (this._shouldSendFrame && !this._shouldSendFrame(payload)) {
				return;
			}

			if (this._ws && this._ws.readyState === WebSocket.OPEN) {
				if (this._frameFormat === "binary-v1") {
					this._ws.send(encodeBinaryAudioFrameV1(pcmBuf, payload.payload));
				} else {
					this._ws.send(JSON.stringify(payload));
				}
			}
		},

		// -----------------------------------------------------------------------
		// Internal: teardown helpers
		// -----------------------------------------------------------------------

		_teardownNodes: function () {
			if (this._workletNode) {
				this._workletNode.disconnect();
				this._workletNode = null;
			}
			if (this._scriptProcessor) {
				this._scriptProcessor.disconnect();
				this._scriptProcessor = null;
			}
			if (this._source) {
				this._source.disconnect();
				this._source = null;
			}
		},

		_teardownStream: function () {
			if (this._stream) {
				this._stream.getTracks().forEach((t) => {
					t.stop();
				});
				this._stream = null;
			}
		},

		_teardownContext: function () {
			if (this._ownedCtx) {
				this._ownedCtx.close();
				this._ownedCtx = null;
			}
		},

		// -----------------------------------------------------------------------
		// Internal: utilities
		// -----------------------------------------------------------------------

		_safeError: function (err) {
			if (this._onError) {
				this._onError(err instanceof Error ? err : new Error(String(err)));
			}
		},

		_nextPow2: (n) => {
			let v = 1;
			while (v < n) v <<= 1;
			return v;
		},
	};

	// -----------------------------------------------------------------------
	// Exports (pure helper functions for testing)
	// -----------------------------------------------------------------------

	MicFrameSender.downsampleFloat32 = downsampleFloat32;
	MicFrameSender.float32ToPcmS16le = float32ToPcmS16le;
	MicFrameSender.arrayBufferToBase64 = arrayBufferToBase64;
	MicFrameSender.encodeBinaryAudioFrameV1 = encodeBinaryAudioFrameV1;
	MicFrameSender.msToSamples = msToSamples;

	return MicFrameSender;
});
