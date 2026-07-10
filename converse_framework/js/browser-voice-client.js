/**
 * browser-voice-client.js — optional composed browser voice client for
 * converse_framework.
 *
 * Combines ``MicFrameSender`` (mic capture) and ``TtsAudioPlayer`` (playback)
 * into a single helper, with optional ``SpeakerEchoGuard`` integration.
 *
 * This module is optional — apps can use ``MicFrameSender`` and
 * ``TtsAudioPlayer`` independently.  It does **not** modify or depend on
 * either module's internal API beyond the documented public interface.
 *
 * ```html
 * <script src="tts-audio-player.js"></script>
 * <script src="mic-frame-sender.js"></script>
 * <script src="speaker-echo-guard.js"></script>
 * <script src="browser-voice-client.js"></script>
 * <script>
 *   const client = new BrowserVoiceClient({
 *     webSocket: new WebSocket("ws://localhost:8000/ws"),
 *   });
 *   client.start();
 *
 *   // Later:
 *   client.stop();
 *   client.close();
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
		root.BrowserVoiceClient = factory();
	}
})(this, () => {
	// -----------------------------------------------------------------------
	// BrowserVoiceClient
	// -----------------------------------------------------------------------

	/**
	 * Create a composed browser voice client.
	 *
	 * @param {Object} options
	 * @param {WebSocket} options.webSocket  - Required: target WebSocket.
	 * @param {Object}  [options.micOptions]  - Options passed to ``MicFrameSender`` constructor.
	 * @param {Object}  [options.playerOptions]  - Options passed to ``TtsAudioPlayer`` constructor.
	 * @param {Object}  [options.guardOptions]  - Options passed to ``SpeakerEchoGuard`` constructor
	 *     (omit to disable echo guard).
	 * @param {boolean} [options.autoStart=false]  - If true, call ``start()`` in constructor.
	 * @param {function(Object):void} [options.onEvent]  - Receive all framework events (from
	 *     the WebSocket) before they are dispatched.
	 */
	function BrowserVoiceClient(options) {
		if (!options || !options.webSocket) {
			throw new Error("BrowserVoiceClient requires a webSocket");
		}

		var ws = options.webSocket;
		this._ws = ws;
		this._micOptions = options.micOptions || {};
		this._playerOptions = options.playerOptions || {};
		this._guardOptions = options.guardOptions || null;
		this._onEvent = options.onEvent || null;

		// WebSocket message handler bound so we can removeEventListener
		this._boundOnMessage = null;

		// Build sub-components
		this._buildMic();
		this._buildPlayer();
		this._buildGuard();

		if (options.autoStart) {
			this.start();
		}
	}

	BrowserVoiceClient.prototype = {
		constructor: BrowserVoiceClient,

		// -----------------------------------------------------------------------
		// Public API
		// -----------------------------------------------------------------------

		/**
		 * Start mic capture and connect WebSocket event handler.
		 * @returns {Promise<void>}
		 */
		start: async function () {
			this._startWebSocketHandler();
			if (this._mic) {
				try {
					await this._mic.start();
				} catch (e) {
					// Mic start failures are surfaced via the mic's onError callback
					// and the caller's catch block.
					throw e;
				}
			}
		},

		/**
		 * Stop mic capture and disconnect WebSocket handler.
		 */
		stop: function () {
			if (this._mic) {
				this._mic.stop();
			}
			this._stopWebSocketHandler();
		},

		/**
		 * Release all resources.
		 */
		close: function () {
			this.stop();
			if (this._player) {
				this._player.close();
			}
			if (this._guard) {
				this._guard.release();
				this._guard = null;
			}
		},

		/**
		 * Current ``MicFrameSender`` instance.
		 * @type {Object|null}
		 */
		get mic() {
			return this._mic;
		},

		/**
		 * Current ``TtsAudioPlayer`` instance.
		 * @type {Object|null}
		 */
		get player() {
			return this._player;
		},

		/**
		 * Current ``SpeakerEchoGuard`` instance (may be null).
		 * @type {Object|null}
		 */
		get guard() {
			return this._guard;
		},

		// -----------------------------------------------------------------------
		// Internal: sub-component construction
		// -----------------------------------------------------------------------

		_buildMic: function () {
			var micOpts = Object.assign({}, this._micOptions);
			if (!micOpts.webSocket) {
				micOpts.webSocket = this._ws;
			}
			// Guard will attach via attachMicSender after construction
			if (
				typeof MicFrameSender !== "undefined" ||
				typeof root.MicFrameSender !== "undefined"
			) {
				var Sender = MicFrameSender || root.MicFrameSender;
				this._mic = new Sender(micOpts);
			}
		},

		_buildPlayer: function () {
			var playerOpts = Object.assign({}, this._playerOptions);
			if (!playerOpts.webSocket) {
				playerOpts.webSocket = this._ws;
			}
			if (
				typeof TtsAudioPlayer !== "undefined" ||
				typeof root.TtsAudioPlayer !== "undefined"
			) {
				var Player = TtsAudioPlayer || root.TtsAudioPlayer;
				this._player = new Player(playerOpts);
			}
		},

		_buildGuard: function () {
			if (
				!this._guardOptions ||
				(typeof SpeakerEchoGuard === "undefined" &&
					typeof root.SpeakerEchoGuard === "undefined")
			) {
				return;
			}

			var Guard = SpeakerEchoGuard || root.SpeakerEchoGuard;
			this._guard = new Guard(this._guardOptions);

			if (this._mic) {
				this._guard.attachMicSender(this._mic);
			}
			if (this._player && typeof this._guard.attachPlayer === "function") {
				// Base the guard's resume on real playback drain rather than
				// event arrival (scheduled audio outlives the final event).
				this._guard.attachPlayer(this._player);
			}
		},

		// -----------------------------------------------------------------------
		// Internal: WebSocket event dispatch
		// -----------------------------------------------------------------------

		_startWebSocketHandler: function () {
			if (this._boundOnMessage) return;

			this._boundOnMessage = (evt) => {
				var msg;
				try {
					msg = JSON.parse(evt.data);
				} catch (_) {
					return; // Not a JSON event
				}

				// Global event observer
				if (this._onEvent) {
					this._onEvent(msg);
				}

				// Dispatch to player
				if (this._player && typeof this._player.onEvent === "function") {
					this._player.onEvent(msg);
				}

				// Dispatch to guard
				if (this._guard && typeof this._guard.onTtsEvent === "function") {
					this._guard.onTtsEvent(msg);
				}
			};

			this._ws.addEventListener("message", this._boundOnMessage);
		},

		_stopWebSocketHandler: function () {
			if (this._boundOnMessage) {
				this._ws.removeEventListener("message", this._boundOnMessage);
				this._boundOnMessage = null;
			}
		},
	};

	return BrowserVoiceClient;
});
