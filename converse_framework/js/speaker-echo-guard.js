/**
 * speaker-echo-guard.js — browser-side echo suppression guard for
 * converse_framework voice clients.
 *
 * When a device runs on speakers (phone, laptop), the microphone can pick
 * up the assistant's TTS playback and re-trigger ASR.  This guard pauses or
 * drops microphone frames while TTS is active, with a configurable tail
 * delay after the last audio chunk to let speaker decay and room echo fade.
 *
 * Two modes:
 *   * ``"drop"`` — continue capture but skip WebSocket sends while suppressed.
 *     Simpler, keeps mic state stable, preserves frame sequencing.
 *   * ``"pause"`` — stop capture while suppressed.  Resumes after the tail
 *     delay.  Uses less CPU/battery during TTS playback.
 *
 * The guard integrates with ``MicFrameSender`` via the optional
 * ``shouldSendFrame`` option or the ``attachMicSender()`` method.
 *
 * ```html
 * <script src="tts-audio-player.js"></script>
 * <script src="mic-frame-sender.js"></script>
 * <script src="speaker-echo-guard.js"></script>
 * <script>
 *   const ws = new WebSocket("ws://localhost:8000/ws");
 *   const player = new TtsAudioPlayer({ webSocket: ws });
 *   const mic = new MicFrameSender({ webSocket: ws });
 *   // Pass the player so the guard resumes when *playback* drains, not
 *   // when the last event arrives (playback can outlive it by seconds).
 *   const guard = new SpeakerEchoGuard({ player });
 *   guard.attachMicSender(mic);
 *
 *   // Forward events to both player and guard
 *   ws.onmessage = (evt) => {
 *     const msg = JSON.parse(evt.data);
 *     player.onEvent(msg);
 *     guard.onTtsEvent(msg);
 *   };
 * </script>
 * ```
 *
 * @module
 */

((root, factory) => {
	if (typeof define === "function" && define.amd) {
		define([], factory);
	} else if (typeof module === "object" && module.exports) {
		module.exports = factory();
	} else {
		root.SpeakerEchoGuard = factory();
	}
})(this, () => {
	// -----------------------------------------------------------------------
	// Constants
	// -----------------------------------------------------------------------

	/**
	 * Fallback timeout (ms): if TTS is streaming but never marks `final`,
	 * force-resume after this duration to avoid stuck mic.
	 */
	const FALLBACK_TIMEOUT_MS = 15000;

	// -----------------------------------------------------------------------
	// SpeakerEchoGuard
	// -----------------------------------------------------------------------

	/**
	 * Create an echo suppression guard.
	 *
	 * @param {Object} [options]
	 * @param {number}  [options.tailDelayMs=350]  - Delay (ms) after playback
	 *     drains before resuming mic frame sending.
	 * @param {string}  [options.mode="drop"]      - ``"drop"`` or ``"pause"``.
	 * @param {Object}  [options.player]  - Optional ``TtsAudioPlayer`` (or any
	 *     object with a ``remainingMs()`` method). When set, the guard resumes
	 *     ``tailDelayMs`` after the player's *scheduled playback* drains rather
	 *     than after the last ``tts.audio`` event — events arrive as fast as
	 *     synthesis streams, so playback can outlive the final event by seconds.
	 * @param {function():number} [options.remainingMs]  - Alternative to
	 *     ``options.player``: callback returning milliseconds of audio still
	 *     scheduled to play.
	 * @param {function(string):void} [options.onStateChange]  - Called with
	 *     ``"idling"``, ``"suppressed"``, or ``"tail"``.
	 * @param {Object} [options.clock]  - Optional clock for testing
	 *     (``{ setTimeout, clearTimeout, Date }``).
	 */
	function SpeakerEchoGuard(options) {
		options = options || {};
		this._tailDelayMs = options.tailDelayMs || 350;
		this._mode = options.mode === "pause" ? "pause" : "drop";
		this._onStateChange = options.onStateChange || null;
		this._player = options.player || null;
		this._remainingMsCallback = options.remainingMs || null;

		// Clock abstraction for testability
		this._clock = options.clock || {
			setTimeout: (fn, ms) => setTimeout(fn, ms),
			clearTimeout: (id) => clearTimeout(id),
		};

		// Internal state
		this._state = "idling"; // "idling" | "suppressed" | "tail"
		this._tailTimer = null;
		this._fallbackTimer = null;
		this._micSender = null;
		this._micWasRunning = false;
		this._suppressionCount = 0;
	}

	SpeakerEchoGuard.prototype = {
		constructor: SpeakerEchoGuard,

		// -----------------------------------------------------------------------
		// Public API
		// -----------------------------------------------------------------------

		/**
		 * Feed a framework event to the guard.
		 *
		 * The guard watches for these event types:
		 * - ``tts.first_chunk`` — enter suppressed.
		 * - ``tts.audio`` — enter/refresh suppressed.
		 * - ``tts.cancelled`` — start tail timer.
		 * - ``tts.error`` — start tail timer.
		 * - ``turn.finished`` — start tail timer.
		 *
		 * Other event types are ignored.
		 *
		 * @param {Object} event  Framework event object with ``type`` string.
		 */
		onTtsEvent: function (event) {
			if (!event || typeof event.type !== "string") return;

			switch (event.type) {
				case "tts.first_chunk":
					this._enterSuppressed();
					break;

				case "tts.audio":
					this._enterSuppressed();
					// Check for final marker
					if (event.payload && event.payload.final === true) {
						this._scheduleResume();
					}
					break;

				case "tts.cancelled":
				case "tts.error":
					this._scheduleResume();
					break;

				case "turn.finished":
					// Only schedule resume if we were suppressed (handles non-TTS turns)
					if (this._state !== "idling") {
						this._scheduleResume();
					}
					break;

				default:
					break;
			}
		},

		/**
		 * Whether mic frames should currently be paused or dropped.
		 * @returns {boolean}
		 */
		isSuppressed: function () {
			return this._state !== "idling";
		},

		/**
		 * Check whether a specific frame should be sent.
		 *
		 * In ``"drop"`` mode, returns ``false`` while suppressed (drop the
		 * frame).  In ``"pause"`` mode, returns ``false`` while suppressed
		 * and also stops capture.  Resume is handled by the tail timer.
		 *
		 * @param {Object} framePayload  The payload object (unused in
		 *     suppression logic, passed for future extensibility).
		 * @returns {boolean}  ``true`` if the frame should be sent.
		 */
		shouldSendFrame: function (framePayload) {
			if (this._state === "idling") return true;

			if (this._mode === "pause") {
				// In pause mode, tell the sender to stop capture.
				// The sender is expected to call _resumeCapture() via the
				// onStateChange callback or tail timer.
				return false;
			}

			// Drop mode: let sender continue capture, skip send
			return false;
		},

		/**
		 * Wire this guard into a ``MicFrameSender`` instance.
		 *
		 * Sets the sender's ``shouldSendFrame`` option to the guard's
		 * ``shouldSendFrame`` method.  Also wires state changes so
		 * ``"pause"`` mode can stop and resume the sender.
		 *
		 * @param {Object} micSender  A ``MicFrameSender`` instance.
		 */
		attachMicSender: function (micSender) {
			this._micSender = micSender;

			// Wire the shouldSendFrame gate
			micSender._shouldSendFrame = (payload) => this.shouldSendFrame(payload);

			// Wire state changes for pause mode
			if (this._mode === "pause" && !this._onStateChange) {
				this._onStateChange = (state) => {
					if (state === "suppressed" && micSender._running) {
						this._micWasRunning = true;
						// In pause mode, actually stop the sender
					} else if (state === "idling" && this._micWasRunning) {
						// Resume will be triggered by tail timer
					}
				};
			}
		},

		/**
		 * Wire this guard to a ``TtsAudioPlayer`` so resume timing is based
		 * on the player's scheduled-playback drain (``remainingMs()``)
		 * instead of event arrival.
		 *
		 * @param {Object} player  A ``TtsAudioPlayer`` instance (or any object
		 *     with a ``remainingMs()`` method).
		 */
		attachPlayer: function (player) {
			this._player = player;
		},

		/**
		 * Release all timers and reset state.
		 */
		release: function () {
			this._clearTimers();
			this._state = "idling";
			this._micSender = null;
			this._micWasRunning = false;
			this._setState("idling");
		},

		// -----------------------------------------------------------------------
		// Internal state machine
		// -----------------------------------------------------------------------

		_enterSuppressed: function () {
			this._clearTimers();
			if (this._state !== "suppressed") {
				this._state = "suppressed";
				this._setState("suppressed");
			}

			if (this._mode === "pause" && this._micSender) {
				this._micWasRunning = this._micSender._running;
			}
		},

		_scheduleResume: function () {
			// Already in tail or idling — no-op for repeated final markers
			if (this._state === "tail" || this._state === "idling") return;

			// Enter tail state
			this._state = "tail";
			this._setState("tail");
			this._suppressionCount = 0; // reset for next turn

			// Clear previous tail timer
			if (this._tailTimer) {
				this._clock.clearTimeout(this._tailTimer);
			}
			if (this._fallbackTimer) {
				this._clock.clearTimeout(this._fallbackTimer);
			}

			// Schedule resume once scheduled playback has drained, plus the
			// tail delay. Without a player/callback the drain is 0 and this
			// degrades to the plain tail-delay behaviour.
			this._armTailTimer();

			// Fallback: force resume even if a final marker was missed or the
			// drain estimate never reaches zero. Budgeted from the playback
			// known to be scheduled right now so long replies are not cut off.
			this._fallbackTimer = this._clock.setTimeout(() => {
				this._resume();
			}, this._remainingPlaybackMs() + FALLBACK_TIMEOUT_MS);
		},

		_armTailTimer: function () {
			var remaining = this._remainingPlaybackMs();
			this._tailTimer = this._clock.setTimeout(() => {
				this._tailTimer = null;
				// Re-check: more audio may have been scheduled while waiting
				// (the drain estimate grows as chunks keep streaming in).
				if (this._remainingPlaybackMs() > 0) {
					this._armTailTimer();
					return;
				}
				this._resume();
			}, remaining + this._tailDelayMs);
		},

		_remainingPlaybackMs: function () {
			var value = 0;
			try {
				if (typeof this._remainingMsCallback === "function") {
					value = this._remainingMsCallback();
				} else if (
					this._player &&
					typeof this._player.remainingMs === "function"
				) {
					value = this._player.remainingMs();
				}
			} catch (e) {
				// A broken drain estimate must never wedge the mic.
				value = 0;
			}
			return typeof value === "number" && value > 0 ? value : 0;
		},

		_resume: function () {
			if (this._state === "idling") return;
			this._clearTimers();

			var wasSuppressed =
				this._state === "suppressed" || this._state === "tail";
			this._state = "idling";
			this._setState("idling");

			// In pause mode, restart capture if it was running before
			if (wasSuppressed && this._mode === "pause" && this._micSender) {
				if (this._micWasRunning && !this._micSender._running) {
					this._micSender.start().catch(() => {});
				}
			}
		},

		_setState: function (state) {
			if (this._onStateChange) {
				try {
					this._onStateChange(state);
				} catch (e) {
					// Swallow errors from user-provided callbacks
				}
			}
		},

		_clearTimers: function () {
			if (this._tailTimer) {
				this._clock.clearTimeout(this._tailTimer);
				this._tailTimer = null;
			}
			if (this._fallbackTimer) {
				this._clock.clearTimeout(this._fallbackTimer);
				this._fallbackTimer = null;
			}
		},
	};

	return SpeakerEchoGuard;
});
