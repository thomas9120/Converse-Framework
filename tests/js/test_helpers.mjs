/**
 * Node-compatible tests for pure JS helper functions extracted from
 * ``mic-frame-sender.js``.
 *
 * Run with: node tests/js/test_helpers.mjs
 */

import { createRequire } from "module";
const require = createRequire(import.meta.url);

// Load the mic-frame-sender module in CJS mode
const MicFrameSender = require("../../converse_framework/js/mic-frame-sender.js");

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
// downsampleFloat32
// ---------------------------------------------------------------------------

{
	const fn = MicFrameSender.downsampleFloat32;

	// Same rate = identity
	const input = new Float32Array([0.0, 0.5, -0.5, 1.0, -1.0, 0.25]);
	const same = fn(input, 16000, 16000);
	assert(
		same.length === input.length,
		"downsample: same rate preserves length",
	);
	for (let i = 0; i < input.length; i++) {
		assert(
			Math.abs(same[i] - input[i]) < 0.001,
			`downsample: same rate value ${i}`,
		);
	}

	// Downsample 2x (32000 → 16000)
	const down = fn(input, 32000, 16000);
	assert(down.length < input.length, "downsample: reduces length");
	assert(
		down.length === Math.round(input.length / 2),
		"downsample: correct length",
	);

	// Upsample 2x (8000 → 16000)
	const up = fn(input, 8000, 16000);
	assert(up.length > input.length, "upsample: increases length");
}

// ---------------------------------------------------------------------------
// float32ToPcmS16le
// ---------------------------------------------------------------------------

{
	const fn = MicFrameSender.float32ToPcmS16le;

	// Zero input
	const zeros = new Float32Array([0, 0, 0]);
	const zBuf = fn(zeros);
	const zView = new DataView(zBuf);
	for (let i = 0; i < 3; i++) {
		assert(zView.getInt16(i * 2, true) === 0, `pcm: zero at ${i}`);
	}

	// Positive sample
	const pos = new Float32Array([1.0]);
	const pBuf = fn(pos);
	const pView = new DataView(pBuf);
	assert(pView.getInt16(0, true) === 32767, "pcm: 1.0 → 32767");

	// Negative sample
	const neg = new Float32Array([-1.0]);
	const nBuf = fn(neg);
	const nView = new DataView(nBuf);
	assert(nView.getInt16(0, true) === -32768, "pcm: -1.0 → -32768");

	// Clamping
	const over = new Float32Array([1.5, -1.5]);
	const oBuf = fn(over);
	const oView = new DataView(oBuf);
	assert(oView.getInt16(0, true) === 32767, "pcm: 1.5 clamped to 32767");
	assert(oView.getInt16(2, true) === -32768, "pcm: -1.5 clamped to -32768");

	// Correct byte length
	const four = new Float32Array([0, 0, 0, 0]);
	assert(fn(four).byteLength === 8, "pcm: 4 samples = 8 bytes");
}

// ---------------------------------------------------------------------------
// arrayBufferToBase64
// ---------------------------------------------------------------------------

{
	const fn = MicFrameSender.arrayBufferToBase64;

	// Simple known values
	const buf1 = new ArrayBuffer(0);
	assert(fn(buf1) === "", "base64: empty");

	const buf2 = new Uint8Array([0x48, 0x65, 0x6c, 0x6c, 0x6f]).buffer; // "Hello"
	assert(fn(buf2) === "SGVsbG8=", "base64: Hello");

	const buf3 = new Uint8Array([0, 0, 0]).buffer; // three null bytes
	assert(fn(buf3) === "AAAA", "base64: three nulls");
}

// ---------------------------------------------------------------------------
// msToSamples
// ---------------------------------------------------------------------------

{
	const fn = MicFrameSender.msToSamples;

	assert(fn(1000, 16000) === 16000, "msToSamples: 1s at 16kHz");
	assert(fn(30, 16000) === 480, "msToSamples: 30ms at 16kHz");
	assert(fn(100, 44100) === 4410, "msToSamples: 100ms at 44.1kHz");
	assert(fn(0, 16000) === 0, "msToSamples: 0ms");
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

console.log(
	`\nResults: ${passed} passed, ${failed} failed out of ${passed + failed}`,
);
if (failed > 0) process.exit(1);
