"""Standardised provider lifecycle event helpers.

These helpers produce a consistent event shape for provider loading,
loaded, and error events across VAD, ASR, LLM, and TTS providers.
The pipeline and provider code emit these alongside existing
``asr.progress`` and ``tts.progress`` events for backward compat.

Event types emitted:

* ``provider.loading`` — a provider has begun loading a model.
* ``provider.loaded`` — a provider has finished loading.
* ``provider.error`` — a provider encountered a non-recoverable error.

Payload fields:

* ``kind`` — ``"vad"`` | ``"asr"`` | ``"llm"`` | ``"tts"``
* ``provider`` — provider name (:attr:`ProviderStatus.name`)
* ``provider_id`` — stable identifier (:attr:`ProviderStatus.provider_id`)
* ``stage`` — substage description (``"loading"``, ``"loaded"``, …)
* ``message`` — human-readable detail
* ``error_type`` — exception class name for error events
* ``loaded`` — bool, whether the provider reports loaded after the event
* ``latency_ms`` — elapsed milliseconds when available
* ``turn_id`` and ``mode`` — tied to a turn when emitted from pipeline

Typical usage::

    from converse_framework.provider_events import provider_loading_event

    await sink.emit(**provider_loading_event(
        kind="asr",
        provider="faster-whisper",
        stage="loading",
        message="Loading model...",
    ))

Or with latency and turn context::

    await sink.emit(
        **provider_error_event(
            kind="tts",
            provider="pocket-tts",
            stage="synthesis",
            message=str(exc),
            error_type=type(exc).__name__,
        ),
        turn_id=turn_id,
        mode=turn_mode,
        latency_ms=elapsed_ms(started),
    )
"""

from __future__ import annotations

from typing import Any


def provider_loading_event(
    *,
    kind: str,
    provider: str,
    stage: str = "loading",
    message: str = "",
    **extra: Any,
) -> dict[str, Any]:
    """Build a ``provider.loading`` event payload.

    Args:
        kind: Provider category (``"vad"``, ``"asr"``, ``"llm"``, ``"tts"``).
        provider: Provider name from :attr:`ProviderStatus.name`.
        stage: Sub-stage label (e.g. ``"loading"``, ``"downloading"``).
        message: Human-readable description.
        **extra: Additional fields forwarded verbatim.

    Returns:
        A keyword-expandable dict with ``event_type`` and ``payload``
        suitable for ``await sink.emit(**result)``.
    """
    return {
        "event_type": "provider.loading",
        "kind": kind,
        "provider": provider,
        "stage": stage,
        "message": message,
        "loaded": False,
        **extra,
    }


def provider_loaded_event(
    *,
    kind: str,
    provider: str,
    stage: str = "loaded",
    message: str = "",
    latency_ms: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a ``provider.loaded`` event payload.

    Args:
        kind: Provider category.
        provider: Provider name.
        stage: Sub-stage label (e.g. ``"loaded"``).
        message: Human-readable description.
        latency_ms: Elapsed milliseconds for the load operation.
        **extra: Additional fields forwarded verbatim.

    Returns:
        A keyword-expandable dict for ``await sink.emit(**result)``.
    """
    payload: dict[str, Any] = {
        "event_type": "provider.loaded",
        "kind": kind,
        "provider": provider,
        "stage": stage,
        "message": message,
        "loaded": True,
    }
    if latency_ms is not None:
        payload["latency_ms"] = latency_ms
    payload.update(extra)
    return payload


def provider_error_event(
    *,
    kind: str,
    provider: str,
    stage: str = "",
    message: str = "",
    error_type: str = "Exception",
    loaded: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    """Build a ``provider.error`` event payload.

    Args:
        kind: Provider category.
        provider: Provider name.
        stage: Sub-stage where the error occurred.
        message: Human-readable error description.
        error_type: Exception class name (e.g. ``"RuntimeError"``).
        loaded: Whether the provider was loaded at the time of error.
        **extra: Additional fields forwarded verbatim.

    Returns:
        A keyword-expandable dict for ``await sink.emit(**result)``.
    """
    return {
        "event_type": "provider.error",
        "kind": kind,
        "provider": provider,
        "stage": stage,
        "message": message,
        "error_type": error_type,
        "loaded": loaded,
        **extra,
    }
