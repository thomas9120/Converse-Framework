"""Generic OpenAI-compatible chat-completions LLM provider.

Works with any server that implements the OpenAI ``/v1/chat/completions``
and ``/v1/models`` endpoints: Ollama, LM Studio, vLLM, llama.cpp, Groq,
OpenRouter, Together, and OpenAI itself. Install with::

    pip install 'converse-framework[openai-compat]'

Configuration::

    {
        "llm": {
            "provider": "openai-compatible",
            "base_url": "https://api.openai.com",  # no /v1 suffix
            "model": "gpt-4.1-mini",               # "auto" = first listed
            "api_key": "sk-...",                   # optional Bearer token
        }
    }

``base_url`` must not include the ``/v1`` path segment -- the provider
appends ``/v1/chat/completions`` and ``/v1/models`` itself. ``model``
defaults to ``"auto"``, which resolves to the first entry reported by
``/v1/models``; hosted services list many models, so set it explicitly
for anything other than a single-model local server.

Unlike :class:`~converse_framework.providers.llamacpp.LlamaCppProvider`
(which this class extends), ``check_status`` does not probe the
llama.cpp-specific ``/health`` endpoint -- it goes straight to
``/v1/models``, which every OpenAI-compatible server exposes.
"""

from __future__ import annotations

from converse_framework.providers.llamacpp import LlamaCppProvider


class OpenAICompatLLMProvider(LlamaCppProvider):
    display_name = "openai-compatible"
    default_provider_id = "openai-compatible"
    install_extra = "openai-compat"
    use_health_endpoint = False


__all__ = ["OpenAICompatLLMProvider"]
