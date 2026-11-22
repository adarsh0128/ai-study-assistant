"""Tests for the Streamlit frontend's non-UI logic.

WHAT IS WORTH TESTING HERE — AND WHAT IS NOT
--------------------------------------------
Streamlit pages are mostly declarative UI: given this state, draw these
widgets. Asserting on widget output is brittle (labels and layout change for
cosmetic reasons) and tests very little.

The parts that hold real logic, and therefore real bugs, are:

  * `api_client` — URL construction, payload shape, and the translation of
    every failure mode into a typed `APIError`. A bug here is invisible until
    a user hits an error path, which is exactly when clarity matters most.
  * `ui.validate_notes` — the client-side guard that prevents a doomed request.

These tests therefore drive `api_client` against a fake `requests` module
rather than a live server, so they run offline, in milliseconds, and can
simulate failures (timeouts, 500s, HTML bodies, rate limits) that would be
awkward to reproduce against a real backend.

WHY FRONTEND MODULES ARE IMPORTED BY PATH
-----------------------------------------
`streamlit run` puts `frontend/` on `sys.path`, not the project root, so the
frontend's own imports are flat (`import api_client`). Tests must reproduce
that or they would import a module the app itself cannot import. The
`frontend_path` fixture below does it once.
"""

import sys
from pathlib import Path
from typing import Any

import pytest
import requests

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@pytest.fixture()
def frontend_path(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put `frontend/` on sys.path, mirroring `streamlit run`.

    Also clears any cached `api_client` / `ui` module so each test starts from
    a clean import — important because `api_client` reads `BACKEND_URL` from
    the environment and a stale module would keep the old value.
    """
    monkeypatch.syspath_prepend(str(FRONTEND_DIR))
    for name in ("api_client", "ui"):
        sys.modules.pop(name, None)
    yield FRONTEND_DIR
    for name in ("api_client", "ui"):
        sys.modules.pop(name, None)


@pytest.fixture()
def api_client(frontend_path: Path) -> Any:
    """Import the frontend API client under test."""
    import api_client as module

    return module


class FakeResponse:
    """Minimal stand-in for `requests.Response`."""

    def __init__(self, status_code: int = 200, json_body: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self._json_body = json_body
        self.text = text

    def json(self) -> Any:
        if self._json_body is None:
            raise ValueError("No JSON")
        return self._json_body


class FakeRequests:
    """Records calls and returns/raises whatever the test configures.

    `api_client` catches `requests.Timeout` / `ConnectionError` /
    `RequestException`, so the fake must expose those names too — it stands in
    for the whole `requests` module, not just its `post`/`get` functions.
    """

    #: Re-exported so `except requests.RequestException` still works.
    Timeout = requests.Timeout
    ConnectionError = requests.ConnectionError
    RequestException = requests.RequestException

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response: FakeResponse | None = None
        self.error: Exception | None = None

    def post(self, url: str, *, json: Any = None, timeout: float | None = None) -> FakeResponse:
        self.calls.append({"method": "post", "url": url, "json": json, "timeout": timeout})
        if self.error is not None:
            raise self.error
        assert self.response is not None, "test must set FakeRequests.response"
        return self.response

    def get(self, url: str, *, timeout: float | None = None) -> FakeResponse:
        self.calls.append({"method": "get", "url": url, "timeout": timeout})
        if self.error is not None:
            raise self.error
        assert self.response is not None, "test must set FakeRequests.response"
        return self.response


@pytest.fixture()
def fake_requests(api_client: Any, monkeypatch: pytest.MonkeyPatch) -> FakeRequests:
    """Replace `requests` inside api_client with a controllable fake."""
    fake = FakeRequests()
    monkeypatch.setattr(api_client, "requests", fake)
    return fake


# =============================================================================
# URL construction
# =============================================================================
def test_backend_url_defaults_to_localhost(api_client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BACKEND_URL", raising=False)
    assert api_client.get_backend_url() == "http://127.0.0.1:8000"


def test_backend_url_strips_trailing_slash(api_client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A trailing slash in .env must not produce '//api/v1/quiz'."""
    monkeypatch.setenv("BACKEND_URL", "http://example.test:9000/")
    assert api_client.get_backend_url() == "http://example.test:9000"


def test_endpoint_paths_match_the_backend_contract(api_client: Any, fake_requests: FakeRequests) -> None:
    """Every feature posts to its documented path.

    These paths are duplicated between frontend and backend by necessity (two
    processes, no shared constant). This test is the tripwire: if the backend
    router prefix changes, this fails here rather than as a mystery 404 in the
    browser.
    """
    fake_requests.response = FakeResponse(200, {"data": {}, "model": "m", "usage": {}})

    api_client.summarise("x" * 50)
    api_client.quiz("x" * 50)
    api_client.explain("concept")
    api_client.interview_questions("x" * 50)
    api_client.chat("hello")

    paths = [call["url"].split("8000", 1)[1] for call in fake_requests.calls]
    assert paths == [
        "/api/v1/summarise",
        "/api/v1/quiz",
        "/api/v1/explain",
        "/api/v1/interview-questions",
        "/api/v1/chat",
    ]


# =============================================================================
# Request payloads
# =============================================================================
def test_summarise_sends_options(api_client: Any, fake_requests: FakeRequests) -> None:
    fake_requests.response = FakeResponse(200, {"data": {}, "model": "m", "usage": {}})

    api_client.summarise("my notes", style="paragraph", max_bullets=9)

    assert fake_requests.calls[0]["json"] == {
        "notes": "my notes",
        "style": "paragraph",
        "max_bullets": 9,
    }


def test_quiz_sends_difficulty_and_count(api_client: Any, fake_requests: FakeRequests) -> None:
    fake_requests.response = FakeResponse(200, {"data": {}, "model": "m", "usage": {}})

    api_client.quiz("notes", num_questions=7, difficulty="hard")

    assert fake_requests.calls[0]["json"] == {
        "notes": "notes",
        "num_questions": 7,
        "difficulty": "hard",
    }


def test_explain_omits_notes_when_not_given(api_client: Any, fake_requests: FakeRequests) -> None:
    """The key should be absent, not null — the backend treats '' as omitted."""
    fake_requests.response = FakeResponse(200, {"data": {}, "model": "m", "usage": {}})

    api_client.explain("entropy")

    assert "notes" not in fake_requests.calls[0]["json"]
    assert fake_requests.calls[0]["json"]["concept"] == "entropy"


def test_explain_includes_notes_when_given(api_client: Any, fake_requests: FakeRequests) -> None:
    fake_requests.response = FakeResponse(200, {"data": {}, "model": "m", "usage": {}})

    api_client.explain("entropy", notes="my notes", level="advanced")

    payload = fake_requests.calls[0]["json"]
    assert payload["notes"] == "my notes"
    assert payload["level"] == "advanced"


def test_chat_sends_history(api_client: Any, fake_requests: FakeRequests) -> None:
    fake_requests.response = FakeResponse(200, {"data": {}, "model": "m", "usage": {}})
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]

    api_client.chat("follow-up", history=history)

    assert fake_requests.calls[0]["json"]["history"] == history


def test_chat_defaults_to_empty_history(api_client: Any, fake_requests: FakeRequests) -> None:
    fake_requests.response = FakeResponse(200, {"data": {}, "model": "m", "usage": {}})

    api_client.chat("hello")

    assert fake_requests.calls[0]["json"]["history"] == []


def test_a_timeout_is_configured(api_client: Any, fake_requests: FakeRequests) -> None:
    """A request with no timeout can hang forever on a dead backend."""
    fake_requests.response = FakeResponse(200, {"data": {}, "model": "m", "usage": {}})

    api_client.summarise("notes")

    assert fake_requests.calls[0]["timeout"] is not None


# =============================================================================
# Success path
# =============================================================================
def test_successful_call_returns_the_envelope(api_client: Any, fake_requests: FakeRequests) -> None:
    envelope = {"data": {"summary": "s", "key_points": ["a"]}, "model": "m", "usage": {"total_tokens": 5}}
    fake_requests.response = FakeResponse(200, envelope)

    assert api_client.summarise("notes") == envelope


# =============================================================================
# Error translation
# =============================================================================
def test_backend_error_body_becomes_api_error(api_client: Any, fake_requests: FakeRequests) -> None:
    """The documented ErrorResponse shape maps straight onto APIError."""
    fake_requests.response = FakeResponse(
        429,
        {
            "detail": "Slow down.",
            "error": "rate_limit_exceeded",
            "retry_after_seconds": 30,
        },
    )

    with pytest.raises(api_client.APIError) as caught:
        api_client.quiz("notes")

    error = caught.value
    assert error.detail == "Slow down."
    assert error.code == "rate_limit_exceeded"
    assert error.status_code == 429
    assert error.retry_after_seconds == 30
    assert error.is_retryable is True


def test_fastapi_validation_error_is_flattened(api_client: Any, fake_requests: FakeRequests) -> None:
    """A 422 from FastAPI has `detail` as a list, not a string.

    Showing the raw list would put a Python repr in front of a student; this
    checks it becomes one readable sentence.
    """
    fake_requests.response = FakeResponse(
        422,
        {
            "detail": [
                {"loc": ["body", "notes"], "msg": "String should have at least 40 characters"},
                {"loc": ["body", "num_questions"], "msg": "Input should be less than 20"},
            ]
        },
    )

    with pytest.raises(api_client.APIError) as caught:
        api_client.quiz("notes")

    error = caught.value
    assert error.code == "invalid_request"
    assert "at least 40 characters" in error.detail
    assert "less than 20" in error.detail
    assert "loc" not in error.detail


def test_connection_error_names_the_url(api_client: Any, fake_requests: FakeRequests) -> None:
    """The most common beginner failure must explain itself."""
    fake_requests.error = requests.ConnectionError("refused")

    with pytest.raises(api_client.APIError) as caught:
        api_client.summarise("notes")

    error = caught.value
    assert error.code == "connection_error"
    assert "127.0.0.1:8000" in error.detail
    assert "uvicorn" in error.detail
    assert error.is_retryable is True


def test_timeout_becomes_timeout_code(api_client: Any, fake_requests: FakeRequests) -> None:
    fake_requests.error = requests.Timeout("too slow")

    with pytest.raises(api_client.APIError) as caught:
        api_client.summarise("notes")

    assert caught.value.code == "timeout"
    assert caught.value.is_retryable is True


def test_html_error_body_does_not_crash(api_client: Any, fake_requests: FakeRequests) -> None:
    """A proxy returning an HTML error page must still yield a clean APIError."""
    fake_requests.response = FakeResponse(502, json_body=None, text="<html>Bad Gateway</html>")

    with pytest.raises(api_client.APIError) as caught:
        api_client.summarise("notes")

    assert caught.value.status_code == 502
    assert "502" in caught.value.detail


def test_non_json_success_body_is_rejected(api_client: Any, fake_requests: FakeRequests) -> None:
    """200 with a non-JSON body means misrouted traffic, not success."""
    fake_requests.response = FakeResponse(200, json_body=None, text="<html>")

    with pytest.raises(api_client.APIError) as caught:
        api_client.summarise("notes")

    assert caught.value.code == "bad_response"


def test_json_array_success_body_is_rejected(api_client: Any, fake_requests: FakeRequests) -> None:
    """The envelope must be an object; a bare list would break every page."""
    fake_requests.response = FakeResponse(200, [1, 2, 3])

    with pytest.raises(api_client.APIError) as caught:
        api_client.summarise("notes")

    assert caught.value.code == "bad_response"


def test_configuration_error_is_not_retryable(api_client: Any, fake_requests: FakeRequests) -> None:
    """A 503 means no API key — retrying cannot help, so the UI must not offer it."""
    fake_requests.response = FakeResponse(
        503, {"detail": "No key.", "error": "llm_not_configured"}
    )

    with pytest.raises(api_client.APIError) as caught:
        api_client.summarise("notes")

    assert caught.value.is_retryable is False


# =============================================================================
# Health
# =============================================================================
def test_health_returns_the_body(api_client: Any, fake_requests: FakeRequests) -> None:
    fake_requests.response = FakeResponse(200, {"status": "ok", "version": "0.1.0"})

    assert api_client.health()["status"] == "ok"


def test_health_failure_raises_api_error(api_client: Any, fake_requests: FakeRequests) -> None:
    fake_requests.error = requests.ConnectionError("refused")

    with pytest.raises(api_client.APIError):
        api_client.health()


# =============================================================================
# ui.validate_notes
# =============================================================================
def test_validate_notes_rejects_empty(api_client: Any, frontend_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty notes -> False, without raising."""
    ui = _import_ui_without_streamlit(monkeypatch)
    assert ui.validate_notes("") is False


def test_validate_notes_rejects_short_input(api_client: Any, frontend_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Below the backend's 40-character floor the model would invent content."""
    ui = _import_ui_without_streamlit(monkeypatch)
    assert ui.validate_notes("too short") is False


def test_validate_notes_accepts_long_enough(api_client: Any, frontend_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ui = _import_ui_without_streamlit(monkeypatch)
    assert ui.validate_notes("x" * ui.MIN_NOTES_CHARS) is True


def test_ui_limits_mirror_the_backend(frontend_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The UI's copy of the limits must not drift from the backend's.

    Duplication across the process boundary is unavoidable; this test makes
    the divergence fail loudly at build time instead of as a confusing 422.
    """
    sys.path.insert(0, str(FRONTEND_DIR.parent))
    try:
        from backend.services import prompts
    finally:
        sys.path.pop(0)

    ui = _import_ui_without_streamlit(monkeypatch)
    assert ui.MIN_NOTES_CHARS == prompts.MIN_NOTES_CHARS
    assert ui.MAX_CHAT_MESSAGE_CHARS == prompts.MAX_CHAT_MESSAGE_CHARS
    assert ui.MAX_EXPLAIN_CONCEPT_CHARS == prompts.MAX_EXPLAIN_CONCEPT_CHARS


def _import_ui_without_streamlit(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import `ui` with Streamlit's rendering functions stubbed out.

    `ui` calls `st.*` only inside functions, so importing it is safe. The
    validator functions themselves don't render anything, but Streamlit emits
    'missing ScriptRunContext' noise outside a real run, which is suppressed
    here to keep test output clean.
    """
    import streamlit as st

    for name in ("info", "warning", "error", "caption"):
        monkeypatch.setattr(st, name, lambda *a, **k: None, raising=False)

    import ui as module

    return module
