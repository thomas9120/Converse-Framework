"""Standalone example consumers for Converse Framework.

These examples prove the framework is useful outside the browser
harness. They live outside the framework's core import path so that
``import converse_framework`` stays lightweight and the examples are
opt-in.

Run the text example from the repository root::

    python -m converse_framework.examples.text_chat

The CLI uses the mock provider bundle by default. To try a real
provider, pass the relevant names and ensure the matching extra is
installed::

    python -m converse_framework.examples.text_chat \\
        --asr faster-whisper \\
        --llm llamacpp \\
        --tts kokoro
"""
