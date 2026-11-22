"""Tests for the Groq client layer — no network, no API key required.

IMPORTANT: these tests must never reach the real Groq API. Two safeguards
enforce that:

  1. An autouse fixture that blanks GROQ_API_KEY in the process environment
     and clears the cached Settings/client between tests. This is necessary
     because backend.config calls load_dotenv(), which puts the real key into
     os.environ — so `Settings(_env_file=None)` alone would still pick it up.
  2. The client is always patched via `get_client`, so even a misconfigured
     key cannot result in a live request.

What is covered:
  * error translation: each Groq exception becomes the right LLMError type
  * response extraction: text stripping, empty/null content, missing usage
  * the API key is never exposed in settings output
  * argument validation that must fail before any network call
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from groq import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

from backend import config as config_module
from backend.config import Settings
from backend.services import groq_client
from backend.services.errors import (
    LLMConfigurationError,
    LLMConnectionError,
    LLMContextLengthError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
)


@pytest.fixture(autouse=True)
def _no_network_settings(monkeypatch):
    """Blank the API key and reset caches before and after every test.

    Without the delenv, load_dotenv() leaves a real key in os.environ and a
    test could silently make a billable network call.
    """
    for var in ("GROQ_API_KEY", "GROQ_MODEL", "GROQ_MAX_RETRIES"):
        monkeypatch.delenv(var, raising=False)
    groq_client.reset_client()
    config_module.get_settings.cache_clear()
    yield
    groq_client.reset_client()
    config_module.get_settings.cache_clear()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeStatusError(Exception):
    """Minimal stand-in carrying an HTTP status code."""

    def __init__(self, status_code: int, message: str = "error") -> None:
        super().__init__(message)
        self.status_code = status_code


def _completion(text: str | None, prompt: int = 10, completion: int = 5):
    """Build a fake response shaped like Groq's chat completion object."""
    return SimpleNamespace(
        model="fake-model",
        choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content=text, role="assistant"),
        )],
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        ),
    )


class _FakeClient:
    """Records the request it received and returns a canned response."""

    def __init__(self, response=None, raises: Exception | None = None) -> None:
        self._response = response
        self._raises = raises
        self.last_request: dict | None = None
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.last_request = kwargs
        if self._raises is not None:
            raise self._raises
        return self._response


@pytest.fixture
def fake_settings():
    """Settings with no API key, so the real client is never constructed.

    The autouse fixture has already removed GROQ_API_KEY from the
    environment, so this object reports is_groq_configured == False.
    """
    settings = Settings(_env_file=None, groq_api_key="")  # type: ignore[call-arg]
    with patch.object(groq_client, "get_settings", return_value=settings):
        yield settings


# ---------------------------------------------------------------------------
# Argument validation — must fail before any network call
# ---------------------------------------------------------------------------


def test_empty_messages_rejected() -> None:
    with pytest.raises(ValueError, match="at least one message"):
        groq_client.chat_completion([])


@pytest.mark.parametrize("bad_temp", [-0.1, 2.01, 5.0])
def test_temperature_out_of_range_rejected(bad_temp: float) -> None:
    with pytest.raises(ValueError, match="temperature must be between"):
        groq_client.chat_completion(
            [{"role": "user", "content": "hi"}], temperature=bad_temp
        )


# ---------------------------------------------------------------------------
# Missing / placeholder key
# ---------------------------------------------------------------------------


def test_missing_key_raises_configuration_error(fake_settings) -> None:
    groq_client.reset_client()
    with pytest.raises(LLMConfigurationError):
        groq_client.chat_completion([{"role": "user", "content": "hi"}])


def test_placeholder_key_is_not_treated_as_configured() -> None:
    settings = Settings(_env_file=None, groq_api_key="your_groq_api_key_here")  # type: ignore[call-arg]
    assert settings.is_groq_configured is False


def test_api_key_is_masked() -> None:
    key = "gsk_" + "x" * 48
    settings = Settings(_env_file=None, groq_api_key=key)  # type: ignore[call-arg]
    masked = settings.masked_api_key

    # The full secret must never appear in the masked form.
    assert key not in masked
    # Only a 4-character prefix and suffix are revealed.
    assert masked.startswith("gsk_")
    assert "xxxx" in masked
    assert masked.count("x") == 4


# ---------------------------------------------------------------------------
# Successful responses
# ---------------------------------------------------------------------------


def test_successful_call_returns_text_and_usage(fake_settings) -> None:
    client = _FakeClient(response=_completion("  hello there  ", 12, 4))
    with patch.object(groq_client, "get_client", return_value=client):
        result = groq_client.chat_completion([{"role": "user", "content": "hi"}])
    assert result["text"] == "hello there"  # stripped
    assert result["total_tokens"] == 16


def test_system_and_user_messages_are_forwarded(fake_settings) -> None:
    client = _FakeClient(response=_completion("ok"))
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ]
    with patch.object(groq_client, "get_client", return_value=client):
        groq_client.chat_completion(messages)
    assert client.last_request is not None
    assert client.last_request["messages"] == messages


def test_json_mode_sets_response_format(fake_settings) -> None:
    client = _FakeClient(response=_completion('{"a": 1}'))
    with patch.object(groq_client, "get_client", return_value=client):
        groq_client.chat_completion(
            [{"role": "user", "content": "json please"}], json_mode=True
        )
    assert client.last_request is not None
    assert client.last_request["response_format"] == {"type": "json_object"}


def test_max_tokens_omitted_when_none(fake_settings) -> None:
    client = _FakeClient(response=_completion("ok"))
    with patch.object(groq_client, "get_client", return_value=client):
        groq_client.chat_completion([{"role": "user", "content": "hi"}])
    assert "max_tokens" not in client.last_request


# ---------------------------------------------------------------------------
# Unusable responses
# ---------------------------------------------------------------------------


def test_empty_content_raises_response_error(fake_settings) -> None:
    client = _FakeClient(response=_completion(None))
    with patch.object(groq_client, "get_client", return_value=client):
        with pytest.raises(LLMResponseError):
            groq_client.chat_completion([{"role": "user", "content": "hi"}])


def test_empty_content_with_length_finish_explains_token_exhaustion(
    fake_settings,
) -> None:
    """Regression: a reasoning model can spend the whole budget on thinking.

    The resulting empty response must say *why*, so the caller knows to
    raise max_tokens or switch models, rather than seeing a generic error.
    """
    response = _completion(None)
    response.choices[0].finish_reason = "length"
    client = _FakeClient(response=response)
    with patch.object(groq_client, "get_client", return_value=client):
        with pytest.raises(LLMResponseError, match="max_tokens"):
            groq_client.chat_completion(
                [{"role": "user", "content": "hi"}], max_tokens=10
            )


def test_whitespace_only_content_raises_response_error(fake_settings) -> None:
    client = _FakeClient(response=_completion("   \n  "))
    with patch.object(groq_client, "get_client", return_value=client):
        with pytest.raises(LLMResponseError):
            groq_client.chat_completion([{"role": "user", "content": "hi"}])


def test_no_choices_raises_response_error(fake_settings) -> None:
    response = SimpleNamespace(model="m", choices=[], usage=None)
    client = _FakeClient(response=response)
    with patch.object(groq_client, "get_client", return_value=client):
        with pytest.raises(LLMResponseError):
            groq_client.chat_completion([{"role": "user", "content": "hi"}])


def test_missing_usage_defaults_to_zero(fake_settings) -> None:
    response = _completion("ok")
    response.usage = None
    client = _FakeClient(response=response)
    with patch.object(groq_client, "get_client", return_value=client):
        result = groq_client.chat_completion([{"role": "user", "content": "hi"}])
    assert result["total_tokens"] == 0


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sdk_error", "expected"),
    [
        (AuthenticationError, LLMConfigurationError),
        (PermissionDeniedError, LLMConfigurationError),
        (NotFoundError, LLMConfigurationError),
        (RateLimitError, LLMRateLimitError),
        (APIConnectionError, LLMConnectionError),
        (APITimeoutError, LLMConnectionError),
    ],
)
def test_sdk_errors_translate_to_our_types(
    fake_settings, sdk_error, expected
) -> None:
    """Each SDK exception class maps to the correct LLMError subclass."""
    # Build these via __new__ so no httpx Response object is required.
    raw = sdk_error.__new__(sdk_error)
    Exception.__init__(raw, "simulated")
    client = _FakeClient(raises=raw)
    with patch.object(groq_client, "get_client", return_value=client):
        with pytest.raises(expected):
            groq_client.chat_completion([{"role": "user", "content": "hi"}])


def test_status_413_maps_to_context_length_error() -> None:
    """413 (token budget exceeded) is a too-long-input error, not a generic one."""
    translated = groq_client._translate_error(_FakeStatusError(413, "too large"))
    assert isinstance(translated, LLMContextLengthError)


def test_status_429_maps_to_rate_limit_error() -> None:
    translated = groq_client._translate_error(_FakeStatusError(429, "slow down"))
    assert isinstance(translated, LLMRateLimitError)


def test_status_500_maps_to_provider_error() -> None:
    translated = groq_client._translate_error(_FakeStatusError(500, "boom"))
    assert isinstance(translated, LLMProviderError)


def test_context_marker_in_bad_request_message() -> None:
    """A 400 whose text mentions context length becomes a context error."""
    exc = BadRequestError.__new__(BadRequestError)
    Exception.__init__(exc, "This model's maximum context length is 8192 tokens")
    exc.status_code = 400
    assert isinstance(groq_client._translate_error(exc), LLMContextLengthError)


def test_internal_server_error_maps_to_provider_error() -> None:
    exc = InternalServerError.__new__(InternalServerError)
    Exception.__init__(exc, "server error")
    exc.status_code = 500
    assert isinstance(groq_client._translate_error(exc), LLMProviderError)
