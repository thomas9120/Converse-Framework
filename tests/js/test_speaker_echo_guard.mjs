/**
 * Node-compatible tests for SpeakerEchoGuard.
 *
 * Uses a mock clock to control time without real delays.
 *
 * Run with: node tests/js/test_speaker_echo_guard.mjs
 */

import { createRequire } from "module";
const require = createRequire(import.meta.url);

const SpeakerEchoGuard = require("../../converse_framework/js/speaker-echo-guard.js");

// ---------------------------------------------------------------------------
// Mock clock
// ---------------------------------------------------------------------------

function createMockClock() {
	let now = 0;
	const timers = new Map();
	let nextId = 1;

	return {
		Date: {
			now: () => now,
		},
		setTimeout: (fn, ms) => {
			const id = nextId++;
			const fireAt = now + ms;
			timers.set(id, { fn, fireAt });
			return id;
		},
		clearTimeout: (id) => {
			timers.delete(id);
		},
		advance(ms) {
			now += ms;
			// Fire expired timers in order
			const expired = [...timers.entries()]
				.filter(([, t]) => t.fireAt <= now)
				.sort(([, a], [, b]) => a.fireAt - b.fireAt);
			for (const [id, t] of expired) {
				timers.delete(id);
				t.fn();
			}
		},
		tick(ms) {
			this.advance(ms);
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
// Test: starts idling
// ---------------------------------------------------------------------------

{
	const guard = new SpeakerEchoGuard({ clock: createMockClock() });
	assert(guard.isSuppressed() === false, "initially idling");
	assert(guard.shouldSendFrame({}) === true, "initially should send");
}

// ---------------------------------------------------------------------------
// Test: tts.first_chunk enters suppressed
// ---------------------------------------------------------------------------

{
	const guard = new SpeakerEchoGuard({ clock: createMockClock() });
	guard.onTtsEvent({ type: "tts.first_chunk" });
	assert(guard.isSuppressed() === true, "first_chunk → suppressed");
	assert(guard.shouldSendFrame({}) === false, "suppressed → drop frame");
}

// ---------------------------------------------------------------------------
// Test: tts.audio enters suppressed
// ---------------------------------------------------------------------------

{
	const guard = new SpeakerEchoGuard({ clock: createMockClock() });
	guard.onTtsEvent({ type: "tts.audio", payload: { final: false } });
	assert(guard.isSuppressed() === true, "audio.chunk → suppressed");
}

// ---------------------------------------------------------------------------
// Test: final tts.audio schedules resume after tail delay
// ---------------------------------------------------------------------------

{
	const clock = createMockClock();
	const guard = new SpeakerEchoGuard({ tailDelayMs: 200, clock });
	guard.onTtsEvent({ type: "tts.first_chunk" });
	assert(guard.isSuppressed() === true, "suppressed before final");

	// Final audio chunk
	guard.onTtsEvent({ type: "tts.audio", payload: { final: true } });
	assert(
		guard.isSuppressed() === true,
		"still suppressed immediately after final",
	);

	// Advance less than tail delay
	clock.advance(100);
	assert(guard.isSuppressed() === true, "still suppressed during tail");

	// Advance past tail delay
	clock.advance(150); // total 250ms > 200ms tail
	assert(guard.isSuppressed() === false, "resumed after tail delay");
	assert(guard.shouldSendFrame({}) === true, "sending after resume");
}

// ---------------------------------------------------------------------------
// Test: multiple tts.audio chunks keep suppression active
// ---------------------------------------------------------------------------

{
	const clock = createMockClock();
	const guard = new SpeakerEchoGuard({ tailDelayMs: 300, clock });
	guard.onTtsEvent({ type: "tts.first_chunk" });

	// Stream of audio chunks
	for (let i = 0; i < 5; i++) {
		guard.onTtsEvent({ type: "tts.audio", payload: { final: false } });
	}
	assert(guard.isSuppressed() === true, "active during stream");

	// Send final
	guard.onTtsEvent({ type: "tts.audio", payload: { final: true } });
	clock.advance(400); // past tail
	assert(guard.isSuppressed() === false, "resumed after stream");
}

// ---------------------------------------------------------------------------
// Test: tts.cancelled schedules resume
// ---------------------------------------------------------------------------

{
	const clock = createMockClock();
	const guard = new SpeakerEchoGuard({ tailDelayMs: 150, clock });
	guard.onTtsEvent({ type: "tts.first_chunk" });
	guard.onTtsEvent({ type: "tts.cancelled" });
	clock.advance(50);
	assert(
		guard.isSuppressed() === true,
		"still suppressed after cancel (before tail)",
	);
	clock.advance(200); // past tail
	assert(guard.isSuppressed() === false, "resumed after cancel tail");
}

// ---------------------------------------------------------------------------
// Test: tts.error schedules resume
// ---------------------------------------------------------------------------

{
	const clock = createMockClock();
	const guard = new SpeakerEchoGuard({ tailDelayMs: 100, clock });
	guard.onTtsEvent({ type: "tts.first_chunk" });
	guard.onTtsEvent({ type: "tts.error", payload: { message: "oops" } });
	clock.advance(250);
	assert(guard.isSuppressed() === false, "resumed after error");
}

// ---------------------------------------------------------------------------
// Test: turn.finished schedules resume when suppressed
// ---------------------------------------------------------------------------

{
	const clock = createMockClock();
	const guard = new SpeakerEchoGuard({ tailDelayMs: 100, clock });
	guard.onTtsEvent({ type: "tts.first_chunk" });
	guard.onTtsEvent({ type: "turn.finished" });
	clock.advance(250);
	assert(guard.isSuppressed() === false, "resumed after turn.finished");
}

// ---------------------------------------------------------------------------
// Test: release clears state
// ---------------------------------------------------------------------------

{
	const guard = new SpeakerEchoGuard({ clock: createMockClock() });
	guard.onTtsEvent({ type: "tts.first_chunk" });
	assert(guard.isSuppressed() === true, "suppressed before release");
	guard.release();
	assert(guard.isSuppressed() === false, "idling after release");
	assert(guard.shouldSendFrame({}) === true, "sending after release");
}

// ---------------------------------------------------------------------------
// Test: state change callback fires
// ---------------------------------------------------------------------------

{
	const states = [];
	const guard = new SpeakerEchoGuard({
		tailDelayMs: 50,
		clock: createMockClock(),
		onStateChange: (s) => states.push(s),
	});

	guard.onTtsEvent({ type: "tts.first_chunk" });
	guard.onTtsEvent({ type: "tts.audio", payload: { final: true } });
	guard._clock.advance(100);

	assert(states.includes("suppressed"), "state: suppressed fired");
	assert(states.includes("tail"), "state: tail fired");
	assert(states.includes("idling"), "state: idling fired");
}

// ---------------------------------------------------------------------------
// Test: resume waits for scheduled playback drain (remainingMs callback)
// ---------------------------------------------------------------------------

{
	const clock = createMockClock();
	let remaining = 1000; // 1s of audio still scheduled on the AudioContext
	const guard = new SpeakerEchoGuard({
		tailDelayMs: 200,
		clock,
		remainingMs: () => remaining,
	});

	guard.onTtsEvent({ type: "tts.first_chunk" });
	guard.onTtsEvent({ type: "tts.audio", payload: { final: true } });

	// Regression (feedback #4): with only the tail delay, the guard would
	// resume at 200ms while the speaker still has 1s of audio to play.
	clock.advance(300);
	assert(
		guard.isSuppressed() === true,
		"still suppressed while playback drains (past plain tail delay)",
	);

	// Playback drains; timer fires at remaining + tail = 1200ms.
	remaining = 0;
	clock.advance(900); // total 1200
	assert(guard.isSuppressed() === false, "resumed after drain + tail delay");
}

// ---------------------------------------------------------------------------
// Test: tail timer re-arms if audio is still scheduled when it fires
// ---------------------------------------------------------------------------

{
	const clock = createMockClock();
	let remaining = 1000;
	const guard = new SpeakerEchoGuard({
		tailDelayMs: 200,
		clock,
		remainingMs: () => remaining,
	});

	guard.onTtsEvent({ type: "tts.first_chunk" });
	guard.onTtsEvent({ type: "tts.audio", payload: { final: true } });

	// More audio got scheduled while waiting: still 500ms left at 1200ms.
	remaining = 500;
	clock.advance(1200);
	assert(guard.isSuppressed() === true, "re-armed while audio remains");

	// Second timer fires at 1200 + 500 + 200 = 1900ms with drain complete.
	remaining = 0;
	clock.advance(700);
	assert(guard.isSuppressed() === false, "resumed after re-armed drain");
}

// ---------------------------------------------------------------------------
// Test: attachPlayer() uses the player's remainingMs()
// ---------------------------------------------------------------------------

{
	const clock = createMockClock();
	const guard = new SpeakerEchoGuard({ tailDelayMs: 100, clock });
	const fakePlayer = { remaining: 400, remainingMs() { return this.remaining; } };
	guard.attachPlayer(fakePlayer);

	guard.onTtsEvent({ type: "tts.first_chunk" });
	guard.onTtsEvent({ type: "tts.audio", payload: { final: true } });

	clock.advance(200); // past plain tail, playback still live
	assert(guard.isSuppressed() === true, "player drain delays resume");

	fakePlayer.remaining = 0;
	clock.advance(300); // timer fires at 400 + 100 = 500ms
	assert(guard.isSuppressed() === false, "resumed after player drained");
}

// ---------------------------------------------------------------------------
// Test: broken remainingMs callback never wedges the mic
// ---------------------------------------------------------------------------

{
	const clock = createMockClock();
	const guard = new SpeakerEchoGuard({
		tailDelayMs: 100,
		clock,
		remainingMs: () => {
			throw new Error("boom");
		},
	});

	guard.onTtsEvent({ type: "tts.first_chunk" });
	guard.onTtsEvent({ type: "tts.audio", payload: { final: true } });
	clock.advance(150);
	assert(
		guard.isSuppressed() === false,
		"throwing drain estimate degrades to plain tail delay",
	);
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

console.log(
	`SpeakerEchoGuard: ${passed} passed, ${failed} failed out of ${passed + failed}`,
);
if (failed > 0) process.exit(1);
