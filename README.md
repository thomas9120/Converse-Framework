# Converse Framework

Provider-agnostic speech stack for speech-to-speech applications.

## Install

```bash
pip install converse-framework
```

## Quick Start

```python
from converse_framework import (
    build_provider_bundle,
    QueueEventSink,
)
import asyncio

config = {
    "vad": {"provider": "mock"},
    "asr": {"provider": "mock"},
    "llm": {"provider": "mock"},
    "tts": {"provider": "mock"},
}

bundle = build_provider_bundle(config)
print(bundle.statuses())
```
