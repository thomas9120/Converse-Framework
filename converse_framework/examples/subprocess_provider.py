"""Subprocess-based ASR provider recipe.

This module shows the pattern for wrapping any external CLI binary
(``whisper-cli`` from whisper.cpp, ``whisper.cpp/main``,
``vosk-transcriber``, etc.) as a framework :class:`ASRProvider`. The
recipe is intentionally minimal so a consumer can copy the class and
adapt it to any tool that reads audio on stdin and writes a transcript
on stdout.

Pattern:

* accept a config with ``binary`` (path or name on PATH) and a
  ``command_template`` (list of argument tokens; ``{model}`` is
  substituted at construction time)
* in :meth:`transcribe_audio`, build a 44-byte WAV header from the
  caller's sample rate and channels, append the PCM s16le body, and
  pipe the result into the subprocess via ``stdin``
* read the subprocess's stdout as the transcript and yield a single
  final :class:`TranscriptEvent`

The example is testable end-to-end with the bundled fake echo script
(``--use-fake-echo``), which writes its stdin back to stdout. That
makes the recipe runnable on a clean machine without installing
``whisper-cli`` or any other ASR binary — the same trick the
``run_subprocess_demo`` driver uses for CI smoke tests.

Usage::

    # with a real whisper-cli install on PATH:
    python -m converse_framework.examples.subprocess_provider \\
        --binary whisper-cli \\
        --model ggml-small.en.bin \\
        --input path/to/16k_mono.wav

    # to validate the wiring without a real ASR:
    python -m converse_framework.examples.subprocess_provider \\
        --binary "$(which python)" \\
        --use-fake-echo \\
        --input path/to/16k_mono.wav
"""

from __future__ import annotations

import argparse
import asyncio
import io
import math
import os
import shutil
import struct
import sys
import tempfile
import wave
from collections.abc import AsyncIterator
from typing import Any

from converse_framework.protocols import (
    ASRProvider,
    ProviderCapabilities,
    ProviderStatus,
    TranscriptEvent,
)


# A tiny executable that emits a deterministic transcript on stdout.
# Used by the CLI driver to validate the subprocess wiring
# end-to-end without requiring whisper-cli (or any real ASR binary)
# on PATH. We deliberately do not echo stdin to stdout: raw PCM
# bytes are not valid UTF-8, which would corrupt the captured
# transcript and trip up the test that asserts on it. Instead the
# script prints a known string plus the number of bytes it
# consumed, so the test can confirm both the wiring and the
# round-trip count.
FAKE_ECHO_SCRIPT = (
    "#!/usr/bin/env python\n"
    "import sys\n"
    "consumed = len(sys.stdin.read())\n"
    "sys.stdout.write(f\"fake transcript: consumed={consumed} bytes\")\n"
)


class SubprocessASRProvider(ASRProvider):
    """ASRProvider that shells out to an external CLI binary.

    The provider wraps a 44-byte WAV header around the caller's PCM
    s16le body and pipes the result into the subprocess's stdin. The
    subprocess's stdout is decoded as UTF-8 and yielded as a single
    final :class:`TranscriptEvent`.

    This is the standard convention for CLI-based ASR tools and is
    the recommended starting point when wrapping any external engine
    that has not yet shipped a first-class provider in the framework
    registry. Consumers can subclass and override
    :meth:`_build_wav_bytes` to swap the WAV-wrapping for a different
    on-disk format the target binary expects.
    """

    def __init__(self, config: dict[str, Any]):
        self.binary: str = str(config.get("binary", ""))
        self.command_template: list[str] = list(
            config.get("command_template", ["-m", "{model}", "-f", "-"])
        )
        self.model: str = str(config.get("model", ""))
        self.timeout_s: float = float(config.get("timeout_s", 120))
        self.language: str | None = (
            str(config["language"]) if config.get("language") else None
        )
        self.channels: int = int(config.get("channels", 1))
        self._ready: bool | None = None
        self._reason: str = ""

    # ------------------------------------------------------------------
    # Protocol surface
    # ------------------------------------------------------------------

    @property
    def status(self) -> ProviderStatus:
        if self._ready is True:
            return ProviderStatus(
                name="subprocess",
                kind="asr",
                ready=True,
                message=(
                    f"Subprocess ASR ready: '{self.binary}' "
                    f"with model '{self.model}'."
                ),
                capabilities=ProviderCapabilities(
                    supports_partials=False,
                    languages=(self.language,) if self.language else ("auto",),
                ),
                provider_id="subprocess",
            )
        if not self.binary:
            return ProviderStatus(
                name="subprocess",
                kind="asr",
                ready=False,
                message=(
                    "Subprocess ASR is not configured: 'binary' is empty. "
                    "Set 'binary' in the provider config to the CLI tool "
                    "to wrap (e.g. 'whisper-cli' or an absolute path)."
                ),
                capabilities=ProviderCapabilities(),
                provider_id="subprocess",
            )
        if not self._is_on_path():
            return ProviderStatus(
                name="subprocess",
                kind="asr",
                ready=False,
                message=(
                    f"Subprocess ASR cannot find binary '{self.binary}' "
                    "on PATH. Install the CLI tool or pass an absolute "
                    "path via 'binary'."
                ),
                capabilities=ProviderCapabilities(),
                provider_id="subprocess",
            )
        return ProviderStatus(
            name="subprocess",
            kind="asr",
            ready=False,
            message=self._reason or "Subprocess ASR not initialised.",
            capabilities=ProviderCapabilities(),
            provider_id="subprocess",
        )

    async def check_status(self) -> ProviderStatus:
        if self._ready is None:
            self._ready = bool(self.binary) and self._is_on_path()
            if not self._ready:
                self._reason = (
                    f"Binary '{self.binary}' not on PATH."
                    if self.binary
                    else "No 'binary' configured."
                )
        return self.status

    async def load(self) -> ProviderStatus:
        # Subprocess ASR providers do not preload anything; the binary
        # is launched on demand per transcription. We just verify the
        # binary is on PATH and remember the result.
        if not self.binary:
            self._ready = False
            self._reason = "No 'binary' configured."
        elif not self._is_on_path():
            self._ready = False
            self._reason = f"Binary '{self.binary}' not on PATH."
        else:
            self._ready = True
            self._reason = ""
        return self.status

    async def transcribe_text_input(
        self, text: str
    ) -> AsyncIterator[TranscriptEvent]:
        stripped = text.strip()
        if stripped:
            yield TranscriptEvent(text=stripped, final=True)

    async def transcribe_audio(
        self,
        pcm_s16le: bytes,
        sample_rate: int,
        progress=None,
    ) -> AsyncIterator[TranscriptEvent]:
        if not self._ready:
            await self.load()
        if not self._ready:
            raise RuntimeError(self._reason or self.status.message)

        wav_bytes = self._build_wav_bytes(pcm_s16le, sample_rate=sample_rate)
        cmd = [self.binary, *self._render_command()]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Subprocess ASR failed to launch '{self.binary}': {exc}"
            ) from exc
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(wav_bytes), timeout=self.timeout_s
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(
                f"Subprocess ASR '{self.binary}' timed out after {self.timeout_s}s."
            )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Subprocess ASR '{self.binary}' failed with exit code "
                f"{proc.returncode}: "
                f"{stderr.decode('utf-8', errors='replace').strip() or '(no stderr)'}"
            )
        text = stdout.decode("utf-8", errors="replace").strip()
        if text:
            yield TranscriptEvent(text=text, final=True)

    async def unload(self) -> ProviderStatus:
        # No persistent state to release; the per-call subprocess already exits.
        return self.status

    # ------------------------------------------------------------------
    # Helpers (subclass-friendly)
    # ------------------------------------------------------------------

    def _render_command(self) -> list[str]:
        rendered: list[str] = []
        for token in self.command_template:
            rendered.append(token.replace("{model}", self.model))
        return rendered

    def _is_on_path(self) -> bool:
        return bool(self.binary) and shutil.which(self.binary) is not None

    def _build_wav_bytes(
        self, pcm_s16le: bytes, *, sample_rate: int
    ) -> bytes:
        """Wrap raw PCM s16le bytes in a minimal 44-byte WAV header.

        Most CLI-based ASR engines (whisper-cli, whisper.cpp/main, the
        Vosk CLI, etc.) read a WAV header from stdin and decode the
        body accordingly. This helper is the recommended default;
        override it in a subclass if the target binary expects a
        different on-the-wire format.
        """
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_s16le)
        return buf.getvalue()


# ----------------------------------------------------------------------
# Driver + CLI
# ----------------------------------------------------------------------


def _synthesize_tone_wav(path: str, *, duration_s: float = 1.0, frequency: float = 440.0) -> None:
    """Write a short mono 16 kHz tone WAV to *path*.

    Used as a default input for the CLI smoke test, so the example
    always has something to feed the subprocess.
    """
    sample_rate = 16000
    total = int(sample_rate * duration_s)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(total):
            sample = int(0.3 * 32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
            frames.extend(struct.pack("<h", sample))
        wf.writeframes(bytes(frames))


async def run_subprocess_demo(
    *,
    binary: str,
    command_template: list[str],
    input_path: str,
    model: str = "",
    timeout_s: float = 30.0,
) -> str | None:
    """Drive :class:`SubprocessASRProvider` against a WAV file.

    Returns the transcript text or ``None`` when the subprocess
    produced no output. The driver is what the CLI ``__main__`` and
    the integration test both call.
    """
    provider = SubprocessASRProvider(
        {
            "binary": binary,
            "command_template": command_template,
            "model": model,
            "timeout_s": timeout_s,
        }
    )
    await provider.load()
    with wave.open(input_path, "rb") as wf:
        assert wf.getnchannels() == 1, "demo expects mono input"
        assert wf.getsampwidth() == 2, "demo expects 16-bit PCM"
        sample_rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
    transcript: str | None = None
    async for event in provider.transcribe_audio(pcm, sample_rate):
        transcript = event.text
    return transcript


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m converse_framework.examples.subprocess_provider",
        description=(
            "Subprocess-based ASR provider recipe. Wraps an external CLI "
            "(e.g. whisper-cli) as an ASRProvider. Use --use-fake-echo to "
            "validate the round-trip without installing a real ASR."
        ),
    )
    parser.add_argument(
        "--binary",
        required=True,
        help="Path to the CLI binary (or a name on PATH).",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model identifier substituted into {model} in the command template.",
    )
    parser.add_argument(
        "--command-template",
        nargs="*",
        default=["-m", "{model}", "-f", "-"],
        help="Tokens to pass after the binary; {model} is replaced with --model.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Path to a 16 kHz mono WAV file. If omitted, a synthetic 1-second "
            "tone is generated and fed to the subprocess."
        ),
    )
    parser.add_argument(
        "--use-fake-echo",
        action="store_true",
        help=(
            "Write the bundled FAKE_ECHO_SCRIPT to a temp file and invoke it "
            "via --binary's interpreter. Useful for CI smoke tests."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    cleanup_paths: list[str] = []
    try:
        binary = args.binary
        if args.use_fake_echo:
            # Write the fake echo script to a temp file and use the
            # configured binary as its interpreter. The command
            # template starts with the script path, so the fake echo
            # receives the WAV bytes on stdin and echoes them to
            # stdout.
            fd, script_path = tempfile.mkstemp(suffix=".py", prefix="fake_echo_")
            os.close(fd)
            with open(script_path, "w", encoding="utf-8") as handle:
                handle.write(FAKE_ECHO_SCRIPT)
            cleanup_paths.append(script_path)
            command_template = [script_path, *args.command_template]
        else:
            command_template = args.command_template

        if args.input is None:
            fd, tone_path = tempfile.mkstemp(suffix=".wav", prefix="subproc_tone_")
            os.close(fd)
            _synthesize_tone_wav(tone_path)
            cleanup_paths.append(tone_path)
            input_path = tone_path
        else:
            input_path = args.input

        transcript = asyncio.run(
            run_subprocess_demo(
                binary=binary,
                command_template=command_template,
                input_path=input_path,
                model=args.model,
            )
        )

        if transcript is None:
            print("(no transcript returned)")
        else:
            print(f"transcript: {transcript!r}")
        return 0
    finally:
        for path in cleanup_paths:
            try:
                os.unlink(path)
            except OSError:
                pass


if __name__ == "__main__":  # pragma: no cover - manual example
    sys.exit(main())
