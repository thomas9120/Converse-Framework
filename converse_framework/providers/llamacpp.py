"""llama.cpp HTTP provider (OpenAI-compatible API).

The ``httpx`` package is imported lazily inside async methods so the base
:mod:`converse_framework` package stays light. Install with::

    pip install 'converse-framework[llamacpp]'

Sampler values are provided either through the ``sampler`` config key at
construction time, or at runtime through :meth:`set_sampler_provider`, which
takes a zero-argument callable returning a dict of sampler overrides. The
framework never imports the harness ``RuntimeSettings`` -- the harness is
responsible for wiring the callable that resolves effective sampler values.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable

from converse_framework.protocols import (
    LLMProvider,
    ProviderCapabilities,
    ProviderStatus,
)

SamplerProvider = Callable[[], dict]


class LlamaCppProvider(LLMProvider):
    def __init__(self, config: dict):
        self.base_url = str(config.get("base_url", "http://127.0.0.1:8080")).rstrip("/")
        self.model = str(config.get("model", "auto"))
        self.temperature = float(config.get("temperature", 0.7))
        self.max_tokens = int(config.get("max_tokens", 256))
        self._sampler_provider: SamplerProvider | None = None
        self._resolved_model: str | None = None
        # Persistent client for streaming turns so consecutive turns
        # reuse the TCP connection. Created lazily; closed in unload().
        self._stream_client: object | None = None

    def set_sampler_provider(self, provider: SamplerProvider | None) -> None:
        """Inject a callable that returns the current sampler overrides.

        The framework never imports the harness ``RuntimeSettings``; the
        harness wires this by passing ``RUNTIME_SETTINGS.effective_sampler``
        (a bound method) or an equivalent lambda.
        """
        self._sampler_provider = provider

    @property
    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name="llama.cpp",
            kind="llm",
            ready=False,
            message=(
                f"Configured for OpenAI-compatible llama.cpp server at {self.base_url}."
            ),
            capabilities=ProviderCapabilities(),
            provider_id="llamacpp",
            status_level="configured",
        )

    async def check_status(self) -> ProviderStatus:
        return await self._http_check_status()

    async def probe_status(self) -> ProviderStatus:
        """Cheap probe: check httpx is importable; no HTTP call."""
        try:
            import httpx  # type: ignore[import-not-found]  # noqa: F401
        except Exception as exc:  # pragma: no cover - import path
            return ProviderStatus(
                name="llama.cpp",
                kind="llm",
                ready=False,
                message=(
                    f"llama.cpp provider requires httpx; install with "
                    f"pip install 'converse-framework[llamacpp]'. ({exc})"
                ),
                capabilities=ProviderCapabilities(),
                provider_id="llamacpp",
                status_level="unavailable",
            )
        # httpx is available; return existing cached status.
        return self.status

    async def load_status(self) -> ProviderStatus:
        """Alias for probe_status - HTTP provider has no model loading."""
        return await self.probe_status()

    async def _http_check_status(self) -> ProviderStatus:
        try:
            import httpx  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - import path
            return ProviderStatus(
                name="llama.cpp",
                kind="llm",
                ready=False,
                message=(
                    f"llama.cpp provider requires httpx; install with "
                    f"pip install 'converse-framework[llamacpp]'. ({exc})"
                ),
                capabilities=ProviderCapabilities(),
                provider_id="llamacpp",
            )
        timeout = httpx.Timeout(connect=1.0, read=2.0, write=1.0, pool=1.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                health = await client.get(f"{self.base_url}/health")
                health.raise_for_status()
                health_payload = health.json()
            except Exception as exc:
                return ProviderStatus(
                    name="llama.cpp",
                    kind="llm",
                    ready=False,
                    message=f"Cannot reach llama.cpp at {self.base_url}: {exc}",
                    capabilities=ProviderCapabilities(),
                    provider_id="llamacpp",
                )

            if health_payload.get("status") != "ok":
                message = health_payload.get("error", {}).get(
                    "message", "server did not report ready"
                )
                return ProviderStatus(
                    name="llama.cpp",
                    kind="llm",
                    ready=False,
                    message=f"llama.cpp reachable but not ready: {message}",
                    capabilities=ProviderCapabilities(),
                    provider_id="llamacpp",
                )

            try:
                models = await client.get(f"{self.base_url}/v1/models")
                models.raise_for_status()
                models_payload = models.json()
            except Exception as exc:
                return ProviderStatus(
                    name="llama.cpp",
                    kind="llm",
                    ready=False,
                    message=f"llama.cpp health OK, but /v1/models failed: {exc}",
                    capabilities=ProviderCapabilities(),
                    provider_id="llamacpp",
                )

        model_ids = [
            str(item.get("id", "unknown")) for item in models_payload.get("data", [])
        ]
        if not model_ids:
            return ProviderStatus(
                name="llama.cpp",
                kind="llm",
                ready=False,
                message=(
                    "llama.cpp health OK, but no loaded model was reported by "
                    "/v1/models."
                ),
                capabilities=ProviderCapabilities(),
                provider_id="llamacpp",
            )
        model_list = ", ".join(model_ids[:3])
        selected_model = self.model if self.model != "auto" else model_ids[0]
        if self.model != "auto" and self.model not in model_ids:
            return ProviderStatus(
                name="llama.cpp",
                kind="llm",
                ready=False,
                message=(
                    f"llama.cpp is ready, but configured model '{self.model}' is "
                    f"not in /v1/models. Loaded: {model_list}"
                ),
                capabilities=ProviderCapabilities(),
                provider_id="llamacpp",
            )
        if self._resolved_model is not None and selected_model != self._resolved_model:
            self._resolved_model = None
        active = "auto-selected" if self.model == "auto" else "selected"
        return ProviderStatus(
            name="llama.cpp",
            kind="llm",
            ready=True,
            message=(
                f"Ready at {self.base_url}; {active} model: {selected_model}; "
                f"loaded: {model_list}"
            ),
            capabilities=ProviderCapabilities(),
            provider_id="llamacpp",
        )

    async def stream_response(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        if self._resolved_model is None:
            self._resolved_model = await self._resolve_model()
        model = self._resolved_model
        sampler = self._build_sampler()
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        for key, value in sampler.items():
            payload[key] = value
        url = f"{self.base_url}/v1/chat/completions"
        client = self._ensure_stream_client()
        try:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
        except Exception:
            self._resolved_model = None
            raise

    def _ensure_stream_client(self):
        if self._stream_client is None:
            try:
                import httpx  # type: ignore[import-not-found]
            except Exception as exc:  # pragma: no cover - import path
                raise RuntimeError(
                    "llama.cpp provider requires httpx; install with "
                    "pip install 'converse-framework[llamacpp]'."
                ) from exc
            timeout = httpx.Timeout(connect=3.0, read=60.0, write=10.0, pool=3.0)
            self._stream_client = httpx.AsyncClient(timeout=timeout)
        return self._stream_client

    def _build_sampler(self) -> dict:
        defaults = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self._sampler_provider is not None:
            # Overrides are merged over the constructor defaults so a
            # provider that returns only e.g. {"top_p": 0.9} does not
            # silently drop temperature / max_tokens.
            return {**defaults, **self._sampler_provider()}
        return defaults

    async def _resolve_model(self) -> str:
        if self.model != "auto":
            return self.model
        try:
            import httpx  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - import path
            raise RuntimeError(
                "llama.cpp provider requires httpx; install with "
                "pip install 'converse-framework[llamacpp]'."
            ) from exc
        timeout = httpx.Timeout(connect=1.0, read=2.0, write=1.0, pool=1.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            payload = response.json()
        model_data = payload.get("data", [])
        if not model_data:
            raise RuntimeError(
                "llama.cpp did not report a loaded model from /v1/models"
            )
        return str(model_data[0].get("id", "unknown"))

    async def unload(self) -> ProviderStatus:
        client = self._stream_client
        self._stream_client = None
        if client is not None:
            await client.aclose()  # type: ignore[attr-defined]
        return self.status
