/**
 * Node-compatible tests for TtsAudioPlayer cancellation and drain
 * reporting, plus the BrowserVoiceClient close() wiring.
 *
 * Uses a fake AudioContext so no audio hardware is needed.
 *
 * Run with: node tests/js/test_tts_audio_player.mjs
 */

import { createRequire } from "module";
const require = createRequire(import.meta.url);

const { TtsAudioPlayer } = require("../../converse_framework/js/tts-audio-player.js");

// ---------------------------------------------------------------------------
// Fake AudioContext
// ---------------------------------------------------------------------------

function createFakeAudioContext(sampleRate) {
	const ctx = {
		sampleRate: sampleRate || 24000,
		currentTime: 0,
		destination: {},
		sources: [],
		createBuffer(channels, length, rate) {
			const data = [];
			for (let ch = 0; ch < channels; ch++) {
				data.push(new Float32Array(length));
			}
			return {
				duration: length / rate,
				numberOfChannels: channels,
				getChannelData: (ch) => data[ch],
			};
		},
		createBufferSource() {
			const source = {
				buffer: null,
				onended: null,
				started: false,
				stopped: false,
				connect() {},
				start(at) {
					this.started = true;
					this.startAt = at;
				},
				stop() {
					this.stopped = true;
				},
			};
			ctx.sources.push(source);
			return source;
		},
	};
	return ctx;
}

function makeAudioEvent(sampleCount, opts) {
	opts = opts || {};
	const pcm = Buffer.alloc(sampleCount * 2); // silence, s16le
	return {
		type: "tts.audio",
		payload: {
			data: pcm.toString("base64"),
			encoding: "pcm_s16le",
			sample_rate: opts.sample_rate || 24000,
			channels: 1,
			final: opts.final !== false,
		},
	};
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

let passed = 0;
let failed = 0;

function assert(condition, label) {
	if (condition) {
		passed++;
	} else {
		failed++;
		console.error(`  FAIL: ${label}`);
	}
}

// ---------------------------------------------------------------------------
// Test: final chunk schedules a source and remainingMs reports drain
// ---------------------------------------------------------------------------

{
	const ctx = createFakeAudioContext(24000);
	const player = new TtsAudioPlayer({ audioContext: ctx });

	player.onEvent(makeAudioEvent(4800)); // 0.2s at 24kHz, final → flush now
	assert(ctx.sources.length === 1, "final chunk schedules one source");
	assert(ctx.sources[0].started === true, "source started");

	const remaining = player.remainingMs();
	assert(Math.abs(remaining - 200) < 1, `remainingMs ≈ 200 (got ${remaining})`);

	// Simulate playback progress
	ctx.currentTime = 0.15;
	assert(
		Math.abs(player.remainingMs() - 50) < 1,
		"remainingMs tracks the context clock",
	);

	// Simulate playback completion
	ctx.currentTime = 0.3;
	assert(player.remainingMs() === 0, "remainingMs clamps to 0 after drain");
}

// ---------------------------------------------------------------------------
// Test: cancel() stops scheduled sources and resets the schedule clock
// ---------------------------------------------------------------------------

{
	const ctx = createFakeAudioContext(24000);
	const player = new TtsAudioPlayer({ audioContext: ctx });

	player.onEvent(makeAudioEvent(4800));
	player.onEvent(makeAudioEvent(4800));
	assert(ctx.sources.length === 2, "two sources scheduled");

	player.cancel();
	assert(
		ctx.sources.every((s) => s.stopped === true),
		"cancel stops every live source",
	);
	assert(player.remainingMs() === 0, "remainingMs is 0 after cancel");

	// Player must keep working after cancel (barge-in, not shutdown)
	player.onEvent(makeAudioEvent(2400));
	assert(ctx.sources.length === 3, "player still accepts audio after cancel");
	assert(ctx.sources[2].started === true, "post-cancel source scheduled");
}

// ---------------------------------------------------------------------------
// Test: clear() is an alias for cancel()
// ---------------------------------------------------------------------------

{
	const ctx = createFakeAudioContext(24000);
	const player = new TtsAudioPlayer({ audioContext: ctx });
	player.onEvent(makeAudioEvent(4800));

	assert(typeof player.clear === "function", "clear() exists");
	player.clear();
	assert(ctx.sources[0].stopped === true, "clear silences scheduled audio");
}

// ---------------------------------------------------------------------------
// Test: tts.cancelled event silences playback (barge-in)
// ---------------------------------------------------------------------------

{
	const ctx = createFakeAudioContext(24000);
	const player = new TtsAudioPlayer({ audioContext: ctx });
	player.onEvent(makeAudioEvent(4800));

	player.onEvent({ type: "tts.cancelled", payload: {} });
	assert(ctx.sources[0].stopped === true, "tts.cancelled stops playback");
	assert(player.remainingMs() === 0, "tts.cancelled resets drain");
}

// ---------------------------------------------------------------------------
// Test: close() silences scheduled audio and stops accepting events
// ---------------------------------------------------------------------------

{
	const ctx = createFakeAudioContext(24000);
	const player = new TtsAudioPlayer({ audioContext: ctx });
	player.onEvent(makeAudioEvent(4800));

	player.close();
	assert(ctx.sources[0].stopped === true, "close silences scheduled audio");

	player.onEvent(makeAudioEvent(4800));
	assert(ctx.sources.length === 1, "closed player ignores new events");
}

// ---------------------------------------------------------------------------
// Test: onended bookkeeping removes finished sources
// ---------------------------------------------------------------------------

{
	const ctx = createFakeAudioContext(24000);
	const player = new TtsAudioPlayer({ audioContext: ctx });
	player.onEvent(makeAudioEvent(4800));

	assert(player._sources.length === 1, "live source tracked");
	ctx.sources[0].onended();
	assert(player._sources.length === 0, "finished source untracked");
}

// ---------------------------------------------------------------------------
// Test: BrowserVoiceClient.close() works (regression: called ._clear())
// ---------------------------------------------------------------------------

{
	globalThis.WebSocket = { OPEN: 1 };
	globalThis.TtsAudioPlayer = TtsAudioPlayer;
	globalThis.MicFrameSender = require("../../converse_framework/js/mic-frame-sender.js");
	globalThis.SpeakerEchoGuard = require("../../converse_framework/js/speaker-echo-guard.js");
	const BrowserVoiceClient = require("../../converse_framework/js/browser-voice-client.js");

	const ws = {
		readyState: 1,
		listeners: [],
		send() {},
		addEventListener(type, fn) {
			this.listeners.push(fn);
		},
		removeEventListener(type, fn) {
			this.listeners = this.listeners.filter((f) => f !== fn);
		},
	};

	const client = new BrowserVoiceClient({
		webSocket: ws,
		playerOptions: { audioContext: createFakeAudioContext(24000) },
		guardOptions: { tailDelayMs: 100 },
	});

	assert(
		client.guard && client.guard._player === client.player,
		"guard is wired to the player for drain-aware resume",
	);

	let threw = false;
	try {
		client.close();
	} catch (e) {
		threw = true;
	}
	assert(threw === false, "close() does not throw");
	assert(client.player._closed === true, "close() closes the player");
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

console.log(
	`TtsAudioPlayer: ${passed} passed, ${failed} failed out of ${passed + failed}`,
);
if (failed > 0) process.exit(1);
