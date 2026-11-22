"""HTTP access to the FastAPI backend — the only module that uses `requests`.

WHY A SEPARATE MODULE INSTEAD OF CALLING THE API FROM EACH PAGE
---------------------------------------------------------------
If each Streamlit page called `requests.post(...)` directly, every page would
need to know the URL layout, the response envelope, the error shapes, and how
long to wait. Five pages means five places to fix when the backend changes.

Instead the pages call *functions* — `summarise(notes=...)` — and this module
owns everything about the wire. The pages then read like the UI they are,
and this file is the single place that knows the backend exists at all.

This is the frontend mirror of the backend's layering: pages play the role of
routers, this module plays the role of the service client.

HOW A REQUEST FLOWS
-------------------
    page  ->  api_client.summarise()  ->  requests.post  ->  FastAPI
                                          ^                    |
                                          |                    v
    page  <-  plain dict / raises APIError  <--  JSON envelope

ERRORS ARE TYPED, NOT STRINGS
-----------------------------
The backend returns a stable machine-readable `error` code (e.g.
"rate_limit_exceeded") plus a human `detail`. This module turns any non-2xx
response into an `APIError` carrying both, so a page can branch on `code`
without parsing English, and display `detail` verbatim because it was written
to be read by a person.

The alternative — letting `requests.HTTPError` escape — would leave every page
inspecting `response.status_code` and re-deriving what went wrong.
"""

from __future__ import annotations

import os
from typing import Any

import requests

# How long to wait for the backend before giving up.
#
# This must exceed the backend's worst case, or a request that is still
# working looks like a network failure to the user. The worst case is:
#     GROQ_TIMEOUT_SECONDS (30) x (1 + GROQ_MAX_RETRIES) (3)  =  90s per call
#     x 2 calls (the original attempt plus one repair)        = 180s
# So the default sits above 180s with headroom. Override with
# API_CLIENT_TIMEOUT_SECONDS if you change the backend's retry settings.
# If 3 minutes feels too long to wait, lower GROQ_MAX_RETRIES in the backend
# rather than lowering this: a short client timeout cannot make the backend
# finish sooner, it only makes the user see a false failure.
DEFAULT_TIMEOUT_SECONDS = float(os.environ.get("API_CLIENT_TIMEOUT_SECONDS", "200"))

# Where the backend lives. Read from the environment so the same frontend code
# runs locally and against a deployed backend without edits.
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


class APIError(Exception):
    """A call to the backend failed, with enough detail to explain why.

    Attributes:
        detail: human-readable message, safe to show the user.
        code: stable machine-readable code from the backend, or a
            client-side code ("connection_error", "timeout", "bad_response")
            when the request never produced a usable HTTP response.
        status_code: HTTP status, or None if no response was received.
        retry_after_seconds: present for 429, so the UI can offer a countdown.
    """

    def __init__(
        self,
        detail: str,
        *,
        code: str = "unknown_error",
        status_code: int | None = None,
        retry_after_seconds: int | None = None,
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.code = code
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds

    @property
    def is_retryable(self) -> bool:
        """True when waiting and trying again could plausibly succeed.

        Drives whether the UI shows a "Try again" button or a "Fix this" hint.
        """
        return self.code in {
            "rate_limit_exceeded",
            "llm_unreachable",
            "llm_provider_error",
            "connection_error",
            "timeout",
        }


def get_backend_url() -> str:
    """Resolve the backend base URL, without a trailing slash.

    Read at call time rather than import time so tests can set the variable
    after import, and so a Streamlit rerun picks up a changed environment.
    """
    return os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL).strip().rstrip("/")


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST JSON to the backend and return the parsed envelope.

    This is the single place that turns transport failures and error responses
    into `APIError`, so every feature function below behaves identically.

    Args:
        path: endpoint path beginning with "/", e.g. "/api/v1/quiz".
        payload: the request body, already shaped for the backend.

    Returns:
        The decoded JSON body (the `{data, model, usage}` envelope).

    Raises:
        APIError: on any transport failure, non-2xx status, or non-JSON body.
    """
    url = f"{get_backend_url()}{path}"

    try:
        response = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT_SECONDS)
    except requests.Timeout as error:
        # The backend is up but slower than we allowed. Distinct from being
        # down: retrying may well work, and the cause is usually a slow model
        # call rather than a broken server.
        raise APIError(
            f"The backend took longer than {DEFAULT_TIMEOUT_SECONDS:.0f}s to "
            "respond. Try again, or send shorter notes.",
            code="timeout",
        ) from error
    except requests.ConnectionError as error:
        # Nothing listening — almost always "you forgot to start the backend".
        # Saying so explicitly saves a beginner a lot of guessing.
        raise APIError(
            f"Could not reach the backend at {get_backend_url()}. "
            "Is it running?  uvicorn backend.main:app --reload",
            code="connection_error",
        ) from error
    except requests.RequestException as error:
        raise APIError(f"Request to the backend failed: {error}") from error

    if response.status_code >= 400:
        raise _error_from_response(response)

    try:
        body = response.json()
    except ValueError as error:
        # A 2xx that is not JSON means something is proxying or misrouting
        # traffic — never a normal backend response.
        raise APIError(
            "The backend returned a response that was not valid JSON.",
            code="bad_response",
            status_code=response.status_code,
        ) from error

    if not isinstance(body, dict):
        raise APIError(
            "The backend returned an unexpected response shape.",
            code="bad_response",
            status_code=response.status_code,
        )
    return body


def _error_from_response(response: requests.Response) -> APIError:
    """Build an APIError from a non-2xx HTTP response.

    Prefers the backend's documented ErrorResponse body, but falls back
    gracefully: a 422 raised by FastAPI's own validation has a different shape
    (`detail` is a list of field errors), and a proxy might return HTML.
    """
    code = "http_error"
    detail = f"The backend returned HTTP {response.status_code}."
    retry_after: int | None = None

    try:
        body = response.json()
    except ValueError:
        body = None

    if isinstance(body, dict):
        raw_code = body.get("error")
        if isinstance(raw_code, str) and raw_code:
            code = raw_code

        raw_detail = body.get("detail")
        if isinstance(raw_detail, str) and raw_detail:
            detail = raw_detail
        elif isinstance(raw_detail, list):
            # FastAPI validation errors: [{"loc": [...], "msg": "..."}, ...].
            # Flatten to one readable line rather than showing a Python repr.
            messages = [
                str(item.get("msg", "")).strip()
                for item in raw_detail
                if isinstance(item, dict)
            ]
            joined = "; ".join(m for m in messages if m)
            if joined:
                detail = f"The request was rejected: {joined}"
                code = "invalid_request"

        raw_retry = body.get("retry_after_seconds")
        if isinstance(raw_retry, int):
            retry_after = raw_retry

    return APIError(
        detail,
        code=code,
        status_code=response.status_code,
        retry_after_seconds=retry_after,
    )


def _get(path: str) -> dict[str, Any]:
    """GET JSON from the backend. Used by the health indicator only."""
    url = f"{get_backend_url()}{path}"
    try:
        response = requests.get(url, timeout=10)
    except requests.RequestException as error:
        raise APIError(f"Could not reach the backend at {url}.", code="connection_error") from error

    if response.status_code >= 400:
        raise _error_from_response(response)

    try:
        body = response.json()
    except ValueError as error:
        raise APIError("The backend returned invalid JSON.", code="bad_response") from error
    return body if isinstance(body, dict) else {}


# =============================================================================
# The five features — one function per endpoint
# =============================================================================
# Each function's signature mirrors the UI's needs rather than the wire format,
# and returns just the `data` payload. The envelope's `model` and `usage` are
# returned too, because the UI shows them (cost visibility, Phase 6 decision).
def summarise(
    notes: str,
    *,
    style: str = "bullets",
    max_bullets: int = 6,
) -> dict[str, Any]:
    """Condense notes into a summary plus key points."""
    return _post(
        "/api/v1/summarise",
        {"notes": notes, "style": style, "max_bullets": max_bullets},
    )


def quiz(
    notes: str,
    *,
    num_questions: int = 5,
    difficulty: str = "medium",
) -> dict[str, Any]:
    """Generate multiple-choice questions from notes."""
    return _post(
        "/api/v1/quiz",
        {
            "notes": notes,
            "num_questions": num_questions,
            "difficulty": difficulty,
        },
    )


def explain(
    concept: str,
    *,
    notes: str | None = None,
    level: str = "beginner",
) -> dict[str, Any]:
    """Explain one concept, optionally grounded in notes."""
    payload: dict[str, Any] = {"concept": concept, "level": level}
    # Omit `notes` entirely when empty: the backend treats "" as None anyway,
    # but not sending the key keeps the request honest about intent.
    if notes:
        payload["notes"] = notes
    return _post("/api/v1/explain", payload)


def interview_questions(
    notes: str,
    *,
    role: str = "software engineer",
    num_questions: int = 5,
    difficulty: str = "medium",
) -> dict[str, Any]:
    """Turn notes into interview questions with answer guidance."""
    return _post(
        "/api/v1/interview-questions",
        {
            "notes": notes,
            "role": role,
            "num_questions": num_questions,
            "difficulty": difficulty,
        },
    )


def chat(
    message: str,
    *,
    notes: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Ask a question conversationally, optionally from notes."""
    payload: dict[str, Any] = {"message": message, "history": history or []}
    if notes:
        payload["notes"] = notes
    return _post("/api/v1/chat", payload)


def health() -> dict[str, Any]:
    """Check whether the backend is up."""
    return _get("/health")
