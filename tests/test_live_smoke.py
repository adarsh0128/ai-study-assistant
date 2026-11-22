"""Live smoke tests — these DO call the real Groq API.

They are skipped by default so `pytest` stays free, offline and fast.
Run them deliberately when you want to confirm your key and model work:

    pytest -m live -v

They require a real GROQ_API_KEY in .env. Each call costs a fraction of a
cent; the whole file uses well under 1000 tokens.

Why have both unit tests and live tests?
  * unit tests (test_groq_client.py) prove *our* logic: translation,
    extraction, validation. They must never depend on the network.
  * live tests prove *the integration*: the key is valid, the model name
    exists, the account has quota. Only a real call can tell you that.
"""

import json

import pytest

from backend.config import get_settings
from backend.services.errors import LLMError
from backend.services.groq_client import chat_completion, get_client

pytestmark = [pytest.mark.live, pytest.mark.allow_network]


@pytest.fixture(autouse=True)
def _require_real_key():
    """Skip the whole module when no real key is configured."""
    settings = get_settings()
    if not settings.is_groq_configured:
        pytest.skip("GROQ_API_KEY is not configured — live tests skipped")


def test_configured_model_exists_on_account() -> None:
    """The model in .env must actually be available, or every call 404s."""
    settings = get_settings()
    available = {m.id for m in get_client().models.list().data}
    assert settings.groq_model in available, (
        f"{settings.groq_model!r} is not available on this account. "
        f"Available: {sorted(available)}"
    )


def test_live_completion_returns_text() -> None:
    result = chat_completion(
        [{"role": "user", "content": "Reply with exactly: connection ok"}],
        temperature=0.0,
        max_tokens=50,
    )
    assert "connection ok" in result["text"].lower()
    assert result["total_tokens"] > 0


def test_live_json_mode_returns_parsable_json() -> None:
    """JSON mode underpins quiz generation, so verify it against the real API."""
    result = chat_completion(
        [{"role": "user", "content":
          'Return JSON only, no prose, with a single key "status" set to "ready".'}],
        temperature=0.0,
        max_tokens=200,
        json_mode=True,
    )
    assert json.loads(result["text"])["status"] == "ready"


def test_live_call_with_invalid_model_raises_configuration_error() -> None:
    """A bad model name must surface as a clear config error, not a crash."""
    from backend.config import Settings
    from backend.services import groq_client
    from backend.services.errors import LLMConfigurationError

    settings = get_settings()
    bad = Settings(
        _env_file=None,
        groq_api_key=settings.groq_api_key,
        groq_model="definitely-not-a-real-model-xyz",
        groq_max_retries=0,
    )
    groq_client.reset_client()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(groq_client, "get_settings", lambda: bad)
        with pytest.raises(LLMConfigurationError):
            groq_client.chat_completion(
                [{"role": "user", "content": "hi"}], max_tokens=10
            )
    groq_client.reset_client()


def test_live_oversized_input_raises_context_error() -> None:
    """Exceeding the token budget must map to LLMContextLengthError."""
    from backend.services.errors import LLMContextLengthError

    with pytest.raises(LLMContextLengthError):
        chat_completion(
            [{"role": "user", "content": "biology. " * 20000}],
            max_tokens=10,
        )


def test_live_error_types_are_llm_errors() -> None:
    """Anything raised by a real call must be one of our own error types."""
    try:
        chat_completion(
            [{"role": "user", "content": "cell. " * 40000}], max_tokens=10
        )
    except LLMError:
        pass  # correct: our abstraction held
