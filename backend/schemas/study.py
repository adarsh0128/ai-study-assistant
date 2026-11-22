"""Pydantic request/response models for the study endpoints — the API contract.

WHY A SEPARATE MODULE FROM services/study_service.py
----------------------------------------------------
The service layer already defines models (SummaryOutput, QuizOutput, ...).
Those describe **what the LLM must produce**. The models here describe
**what this HTTP API accepts and returns**. The two are deliberately
separate files because they change for different reasons:

  * a schema in study_service.py changes when we ask the model for something
    different (a new field, a stricter bound);
  * a model here changes when the *wire format* changes (adding `usage` to a
    response, renaming a request field).

Coupling them would mean a prompt tweak silently alters your public API —
and your frontend would break without a single line changing there.

THE SHARED RESPONSE ENVELOPE
----------------------------
Every one of the five endpoints returns the same three keys:

    {
      "data":  { ...feature-specific payload... },
      "model": "qwen/qwen3.8-27b",
      "usage": { "prompt_tokens": 316, "total_tokens": 429, ... }
    }

One envelope means the frontend writes one helper to unwrap responses, and
Phase 7's api_client.py needs no per-endpoint special casing. `usage` is on
the wire, not just in logs, so cost is visible in the UI.

WHY REQUESTS ARE VALIDATED HERE *AND* IN THE PROMPT LAYER
---------------------------------------------------------
This is defence in depth, not duplication for its own sake:

  * Pydantic at the boundary means Swagger documents every limit, and a bad
    request is rejected before any application code runs (FastAPI returns 422
    with a field-level message).
  * The prompt layer still validates, because `study_service` is callable
    without HTTP — from a test, a CLI, or a background job. A boundary check
    protects the boundary; the prompt layer protects the LLM call itself.

The prompt layer remains the authority on what the model can accept. These
models import its constants so the two can never disagree about the number.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.services.prompts import (
    MAX_CHAT_MESSAGE_CHARS,
    MAX_EXPLAIN_CONCEPT_CHARS,
    MAX_INTERVIEW_QUESTIONS,
    MAX_NOTES_CHARS,
    MAX_QUIZ_QUESTIONS,
    MIN_INTERVIEW_QUESTIONS,
    MIN_NOTES_CHARS,
    MIN_QUIZ_QUESTIONS,
)
from backend.services.study_service import (
    ChatOutput,
    ExplainOutput,
    InterviewOutput,
    QuizOutput,
    SummaryOutput,
)

# A non-empty string, used for short required fields. `min_length=1` rejects
# "" while still allowing any normal text.
NonEmptyStr = Annotated[str, Field(min_length=1)]

# Reusable option types. Literal (rather than str) is what makes Swagger render
# a dropdown and makes an invalid value a 422 instead of a 500 from deep inside
# the prompt builder.
SummariseStyle = Literal["bullets", "paragraph"]
Difficulty = Literal["easy", "medium", "hard"]
ExplainLevel = Literal["beginner", "intermediate", "advanced"]

# How many prior messages a chat request may carry. The prompt layer trims to
# the last MAX_HISTORY_TURNS turns anyway; this ceiling exists to stop an
# oversized payload reaching the service at all.
_MAX_HISTORY_MESSAGES = 50


# =============================================================================
# Shared pieces of the response envelope
# =============================================================================
class UsageModel(BaseModel):
    """Token accounting for one request, mirroring the service's Usage dataclass.

    Defined explicitly rather than reusing the dataclass so the *wire format*
    is owned by the API layer. If `Usage` is ever refactored, this contract
    does not move with it.
    """

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(ge=0, description="Tokens in the prompt we sent.")
    completion_tokens: int = Field(ge=0, description="Tokens in the model's reply.")
    total_tokens: int = Field(ge=0, description="Prompt + completion.")
    repair_attempts: int = Field(
        ge=0,
        le=1,
        description=(
            "0 = the model's first response was usable. 1 = it was not, and "
            "one repair retry succeeded. Token counts include both calls."
        ),
    )
    latency_ms: int = Field(ge=0, description="Wall-clock duration in milliseconds.")


class ErrorResponse(BaseModel):
    """Error body returned for every non-2xx response.

    `detail` follows FastAPI's convention (and is what TestClient/httpx users
    expect). `error` is a **stable machine-readable code** so the frontend can
    branch on the failure without string-matching a human sentence — messages
    get reworded, codes do not.

    Documented on every endpoint via `responses=...` so Swagger shows the real
    error shapes instead of only the happy path.
    """

    detail: str = Field(description="Human-readable explanation.")
    error: str = Field(
        description="Stable machine-readable code, e.g. 'rate_limit_exceeded'."
    )
    retry_after_seconds: int | None = Field(
        default=None,
        description="Present for 429 responses: how long to wait before retrying.",
    )


# =============================================================================
# POST /summarise
# =============================================================================
class SummariseRequest(BaseModel):
    """Body for POST /summarise."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "notes": (
                        "Photosynthesis occurs in the chloroplast. The "
                        "light-dependent reactions take place in the thylakoid "
                        "membrane and produce ATP, NADPH and oxygen from water. "
                        "The Calvin cycle takes place in the stroma and fixes "
                        "carbon dioxide into glucose via rubisco."
                    ),
                    "style": "bullets",
                    "max_bullets": 6,
                }
            ]
        },
    )

    notes: str = Field(
        min_length=MIN_NOTES_CHARS,
        max_length=MAX_NOTES_CHARS,
        description=(
            f"The student's notes. {MIN_NOTES_CHARS}-{MAX_NOTES_CHARS} "
            "characters. Very short input is rejected because the model would "
            "have too little to work from and would start inventing."
        ),
    )
    style: SummariseStyle = Field(
        default="bullets", description="'bullets' for key points, 'paragraph' for prose."
    )
    max_bullets: int = Field(
        default=6, ge=1, le=15, description="Upper bound on the number of key points."
    )


class SummariseResponse(BaseModel):
    """Response for POST /summarise."""

    model_config = ConfigDict(extra="forbid")

    data: SummaryOutput = Field(description="Summary plus key points.")
    model: str = Field(description="The model that produced this answer.")
    usage: UsageModel = Field(description="Token accounting for this request.")


# =============================================================================
# POST /quiz
# =============================================================================
class QuizRequest(BaseModel):
    """Body for POST /quiz."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"notes": "Photosynthesis occurs in the chloroplast. ...", "num_questions": 5, "difficulty": "medium"}
            ]
        },
    )

    notes: str = Field(
        min_length=MIN_NOTES_CHARS,
        max_length=MAX_NOTES_CHARS,
        description="Source notes the questions must be answerable from.",
    )
    num_questions: int = Field(
        default=5,
        ge=MIN_QUIZ_QUESTIONS,
        le=MAX_QUIZ_QUESTIONS,
        description=f"How many questions to generate ({MIN_QUIZ_QUESTIONS}-{MAX_QUIZ_QUESTIONS}).",
    )
    difficulty: Difficulty = Field(
        default="medium", description="Controls how much inference each question needs."
    )


class QuizResponse(BaseModel):
    """Response for POST /quiz."""

    model_config = ConfigDict(extra="forbid")

    data: QuizOutput = Field(description="The generated multiple-choice questions.")
    model: str = Field(description="The model that produced this answer.")
    usage: UsageModel = Field(description="Token accounting for this request.")


# =============================================================================
# POST /explain
# =============================================================================
class ExplainRequest(BaseModel):
    """Body for POST /explain.

    `notes` is optional here, unlike the other features. That changes the
    grounding contract: with notes the answer stays inside them; without notes
    the model answers from general knowledge and `verify_note` is populated so
    the student knows to check it.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"concept": "the Calvin cycle", "level": "beginner"}
            ]
        },
    )

    concept: str = Field(
        min_length=1,
        max_length=MAX_EXPLAIN_CONCEPT_CHARS,
        description="The single concept to explain.",
    )
    notes: str | None = Field(
        default=None,
        max_length=MAX_NOTES_CHARS,
        description=(
            "Optional notes to ground the explanation in. Omit to explain "
            "from general knowledge. A blank string is treated as omitted."
        ),
    )
    level: ExplainLevel = Field(
        default="beginner", description="Assumed prior knowledge of the reader."
    )

    @field_validator("notes")
    @classmethod
    def _blank_notes_become_none(cls, value: str | None) -> str | None:
        """Treat "" as "no notes supplied".

        A frontend sends "" when the user leaves the box empty. Rejecting that
        as a validation error would be a poor experience for an input that is
        genuinely optional, so normalise it here rather than making the caller
        special-case an empty box.
        """
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class ExplainResponse(BaseModel):
    """Response for POST /explain."""

    model_config = ConfigDict(extra="forbid")

    data: ExplainOutput = Field(description="Explanation, analogy, key terms, verify note.")
    model: str = Field(description="The model that produced this answer.")
    usage: UsageModel = Field(description="Token accounting for this request.")


# =============================================================================
# POST /interview-questions
# =============================================================================
class InterviewRequest(BaseModel):
    """Body for POST /interview-questions."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"notes": "Photosynthesis occurs in the chloroplast. ...", "role": "biology intern", "num_questions": 5}
            ]
        },
    )

    notes: str = Field(
        min_length=MIN_NOTES_CHARS,
        max_length=MAX_NOTES_CHARS,
        description="Notes to derive interview questions from.",
    )
    role: str = Field(
        default="software engineer",
        min_length=1,
        max_length=120,
        description="The role being interviewed for; shapes the question framing.",
    )
    num_questions: int = Field(
        default=5,
        ge=MIN_INTERVIEW_QUESTIONS,
        le=MAX_INTERVIEW_QUESTIONS,
        description=f"How many questions to generate ({MIN_INTERVIEW_QUESTIONS}-{MAX_INTERVIEW_QUESTIONS}).",
    )
    difficulty: Difficulty = Field(default="medium", description="Question difficulty.")


class InterviewResponse(BaseModel):
    """Response for POST /interview-questions."""

    model_config = ConfigDict(extra="forbid")

    data: InterviewOutput = Field(description="Interview questions with answer guidance.")
    model: str = Field(description="The model that produced this answer.")
    usage: UsageModel = Field(description="Token accounting for this request.")


# =============================================================================
# POST /chat
# =============================================================================
class ChatHistoryMessage(BaseModel):
    """One prior turn in a conversation.

    `role` is restricted to user/assistant — deliberately NOT including
    "system". Allowing a client to submit a system turn would let it inject
    instructions the model weighs as heavily as ours, which is the same attack
    the prompt layer defends against, arriving through the front door.
    """

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"] = Field(description="Who sent this message.")
    content: str = Field(min_length=1, max_length=MAX_NOTES_CHARS)


class ChatRequest(BaseModel):
    """Body for POST /chat."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "message": "What does rubisco do?",
                    "notes": "Photosynthesis occurs in the chloroplast. ...",
                    "history": [],
                }
            ]
        },
    )

    message: str = Field(
        min_length=1,
        max_length=MAX_CHAT_MESSAGE_CHARS,
        description="The student's question.",
    )
    notes: str | None = Field(
        default=None,
        max_length=MAX_NOTES_CHARS,
        description="Optional notes to answer from. A blank string means 'no notes'.",
    )
    history: list[ChatHistoryMessage] = Field(
        default_factory=list,
        max_length=_MAX_HISTORY_MESSAGES,
        description=(
            "Prior turns, oldest first. Only the most recent turns are sent to "
            "the model; older ones are dropped to protect the context window."
        ),
    )

    @field_validator("notes")
    @classmethod
    def _blank_notes_become_none(cls, value: str | None) -> str | None:
        """Treat "" as "no notes supplied" — see ExplainRequest."""
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class ChatResponse(BaseModel):
    """Response for POST /chat."""

    model_config = ConfigDict(extra="forbid")

    data: ChatOutput = Field(description="The assistant's plain-text reply.")
    model: str = Field(description="The model that produced this answer.")
    usage: UsageModel = Field(description="Token accounting for this request.")


# =============================================================================
# Swagger error documentation
# =============================================================================
def error_responses(*codes: int) -> dict[int | str, dict[str, object]]:
    """Build an OpenAPI `responses=` map so every failure shape is documented.

    Without this, Swagger shows only the 200 schema and a frontend developer
    has to guess what a 429 body looks like. One helper keeps the five
    endpoints consistent.

    Usage:
        @router.post("/quiz", responses=error_responses(422, 429, 502))
    """
    return {
        code: {"model": ErrorResponse, "description": _ERROR_DESCRIPTIONS[code]}
        for code in codes
    }


_ERROR_DESCRIPTIONS: dict[int, str] = {
    400: "The input exceeded what the model's context window can hold. Split it up.",
    422: "The request was well-formed but unusable (notes too short, invalid option).",
    429: "The LLM provider is rate limiting us. See the Retry-After header.",
    502: "The provider returned an error, or output we could not use.",
    503: "The server has no usable API key configured.",
    504: "The provider could not be reached, or the request timed out.",
}
