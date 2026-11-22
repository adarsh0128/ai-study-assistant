"""Groq API access layer.

This is the ONLY module in the project that imports `groq`. Everything above
it (services, routers) speaks in plain strings and our own error types, so
the provider is swappable by editing this one file.

Responsibilities:
  * build and hold a single Groq client
  * translate Groq's exception classes into our `errors.py` types
  * log token usage (never the API key, never full user text)

Phase 3: implemented.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    Groq,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

from backend.config import Settings, get_settings
from backend.services.errors import (
    LLMConfigurationError,
    LLMConnectionError,
    LLMContextLengthError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
)
from backend.utils.logger import get_logger

logger = get_logger(__name__)

MessageRole = Literal["system", "user", "assistant"]

# Substrings Groq uses when the input is too long. Checked case-insensitively
# against the error body so a context overflow becomes a 400 (client's fault)
# rather than a generic 502 (our fault).
_CONTEXT_MARKERS = (
    "context_length",
    "context length",
    "too many tokens",
    "maximum context",
    "reduce the length",
    "request too large",
)

_client: Groq | None = None


class ChatMessage(TypedDict):
    """One message in a conversation.

    Matches the OpenAI-style schema Groq expects: {"role": ..., "content": ...}
    """

    role: MessageRole
    content: str


class ChatResult(TypedDict):
    """What a successful completion returns.

    Returning the text plus usage (rather than a bare string) lets callers
    log token cost without a second API call.
    """

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    # "stop" (finished naturally), "length" (hit max_tokens), or another
    # provider value. Exposed so callers can tell a truncated answer from a
    # complete one — a truncated JSON response is unrecoverable as-is, but
    # retrying with a bigger budget usually works.
    finish_reason: str


def get_client() -> Groq:
    """Return the shared Groq client, creating it on first use.

    Built lazily so importing this module never requires a valid API key —
    tests can import it freely. A missing key surfaces as a clear
    LLMConfigurationError on first real call instead of an import crash.
    """
    global _client
    if _client is not None:
        return _client

    settings = get_settings()
    if not settings.is_groq_configured:
        # Do not include the key value here — it may be a placeholder, but
        # this log line must be safe even if someone mis-sets it.
        raise LLMConfigurationError(
            "GROQ_API_KEY is missing or still set to a placeholder value. "
            "Add a real key to your .env file (see .env.example)."
        )

    _client = Groq(
        api_key=settings.groq_api_key,
        timeout=settings.groq_timeout_seconds,
        max_retries=settings.groq_max_retries,
    )
    logger.info(
        "Groq client initialised (model=%s, timeout=%ss, key=%s)",
        settings.groq_model,
        settings.groq_timeout_seconds,
        settings.masked_api_key,  # masked: never the raw key
    )
    return _client


def reset_client() -> None:
    """Drop the cached client. Used by tests and after a key change."""
    global _client
    _client = None


def _translate_error(exc: Exception) -> Exception:
    """Map a Groq SDK exception onto our own error hierarchy.

    Order matters: the first matching branch wins, so the most specific
    checks come first. `status_code` is read from the exception where
    available because several failures share a base class but need
    different handling — notably 413 (too large), which the SDK raises as
    APIStatusError rather than BadRequestError.
    """
    # --- Network-level failures (no HTTP response at all) -------------------
    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        return LLMConnectionError(
            "Could not reach Groq (network problem or timeout)."
        )

    status = getattr(exc, "status_code", None)
    message = str(exc)
    lowered = message.lower()

    # --- 401 / 403: credentials ---------------------------------------------
    if isinstance(exc, (AuthenticationError, PermissionDeniedError)) or status in (401, 403):
        return LLMConfigurationError(
            f"Groq rejected the API key or its permissions (HTTP {status}). "
            "Check GROQ_API_KEY in .env."
        )

    # --- 404: model name wrong or decommissioned ----------------------------
    if isinstance(exc, NotFoundError) or status == 404:
        return LLMConfigurationError(
            "Groq does not recognise the requested model (HTTP 404). "
            "The model may be decommissioned — check GROQ_MODEL in .env."
        )

    # --- 429: throttling ----------------------------------------------------
    if isinstance(exc, RateLimitError) or status == 429:
        return LLMRateLimitError(
            "Groq rate limit reached (HTTP 429). Wait a moment and retry."
        )

    # --- 413 / context overflow: input too big ------------------------------
    # Groq returns 413 for requests exceeding the per-minute token budget, and
    # 400 with a length message for exceeding the model's context window.
    # Both mean "your input is too large", so both map to one clear error.
    if status == 413 or any(marker in lowered for marker in _CONTEXT_MARKERS):
        return LLMContextLengthError(
            "The input is too long: it exceeds either the model's context "
            "window or this account's per-minute token limit. Shorten the "
            "notes or reduce the conversation history."
        )

    # --- 400: other bad requests -------------------------------------------
    if isinstance(exc, BadRequestError) or status == 400:
        return LLMProviderError(f"Groq rejected the request (HTTP 400): {message}")

    # --- 5xx ----------------------------------------------------------------
    if isinstance(exc, InternalServerError) or (status and status >= 500):
        return LLMProviderError(
            f"Groq had a server-side error (HTTP {status}). Retry shortly."
        )

    if isinstance(exc, APIStatusError):
        return LLMProviderError(f"Groq returned an unexpected status {status}.")

    return LLMProviderError(f"Unexpected error while calling Groq: {exc}")


def _validate_temperature(temperature: float) -> None:
    """Fail loudly on an out-of-range temperature.

    Groq accepts 0..2. Catching this locally gives a clear error instead of
    a confusing 400 from the provider.
    """
    if not 0.0 <= temperature <= 2.0:
        raise ValueError(
            f"temperature must be between 0.0 and 2.0, got {temperature!r}"
        )


def chat_completion(
    messages: list[ChatMessage],
    *,
    temperature: float = 0.3,
    max_tokens: int | None = None,
    json_mode: bool = False,
) -> ChatResult:
    """Send a chat request to Groq and return the generated text plus usage.

    Args:
        messages: conversation as [{role, content}, ...]. Must include at
            least one message.
        temperature: 0.0 (deterministic) .. 2.0 (wild). Low for factual
            study output, higher for conversational replies.
        max_tokens: cap on the reply length. None lets the provider decide.
        json_mode: when True, tells Groq to emit valid JSON only
            (`response_format={"type": "json_object"}`). Requires the prompt
            to mention JSON as well, otherwise Groq returns a 400.

    Raises:
        LLMConfigurationError: missing/invalid key.
        LLMContextLengthError: input exceeds the model's context window.
        LLMConnectionError: network failure or timeout.
        LLMRateLimitError: provider throttling.
        LLMProviderError: any other provider-side failure.
        LLMResponseError: provider returned unusable content.
        ValueError: invalid arguments detected before the call.
    """
    if not messages:
        raise ValueError("messages must contain at least one message")

    _validate_temperature(temperature)
    settings: Settings = get_settings()
    client = get_client()

    request: dict[str, object] = {
        "model": settings.groq_model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    if json_mode:
        request["response_format"] = {"type": "json_object"}

    logger.debug(
        "Groq request: model=%s messages=%d temperature=%s json_mode=%s",
        settings.groq_model,
        len(messages),
        temperature,
        json_mode,
    )

    try:
        response = client.chat.completions.create(**request)  # type: ignore[arg-type]
    except Exception as exc:  # narrowed to our types immediately below
        translated = _translate_error(exc)
        logger.warning(
            "Groq call failed: %s -> %s",
            type(exc).__name__,
            type(translated).__name__,
        )
        raise translated from exc

    # --- Extract the text ---------------------------------------------------
    if not response.choices:
        raise LLMResponseError("Groq returned no choices.")

    choice = response.choices[0]
    content = choice.message.content

    if content is None or not content.strip():
        # The most common cause is a reasoning model that spent the whole
        # token budget on hidden reasoning before emitting any visible text.
        # Distinguish that case so the error tells the caller what to fix.
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            raise LLMResponseError(
                "Groq returned an empty response because max_tokens was "
                "exhausted before any visible text was produced. This usually "
                "means a reasoning model (e.g. openai/gpt-oss-*) consumed the "
                "budget on internal reasoning. Raise max_tokens or switch to "
                "a non-reasoning model such as qwen/qwen3.8-27b."
            )
        raise LLMResponseError(
            "Groq returned an empty response. Try again, or raise max_tokens "
            "if this persists."
        )

    usage = response.usage
    result: ChatResult = {
        "text": content.strip(),
        "model": response.model,
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        "finish_reason": getattr(choice, "finish_reason", None) or "stop",
    }

    # Token counts are the currency of LLM apps — log them so you can see
    # cost and context-headroom as you develop.
    logger.info(
        "Groq call ok: model=%s prompt_tokens=%d completion_tokens=%d "
        "total=%d finish=%s",
        result["model"],
        result["prompt_tokens"],
        result["completion_tokens"],
        result["total_tokens"],
        result["finish_reason"],
    )
    return result
