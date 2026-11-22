"""Study endpoints — the HTTP surface over the study service.

    POST /summarise            notes -> summary + key points
    POST /quiz                 notes -> multiple-choice questions
    POST /explain              concept (+ optional notes) -> explanation
    POST /interview-questions  notes -> interview questions + guidance
    POST /chat                 question (+ optional notes/history) -> reply

WHAT THIS LAYER IS ALLOWED TO DO
--------------------------------
Exactly three things:

  1. Describe the endpoint to FastAPI (path, response_model, examples).
  2. Call one service function, passing its arguments through.
  3. Turn an exception into an HTTP status code.

It contains **no prompting, no parsing, no validation logic of its own, and
no knowledge of Groq.** Every rule about what a valid quiz is lives in the
service; this module only speaks HTTP. That separation is what makes
study_service.py testable without an HTTP client, and lets Phase 7's
frontend depend on a contract instead of an implementation.

WHY ONE EXCEPTION HANDLER INSTEAD OF A TRY/EXCEPT PER ENDPOINT
-------------------------------------------------------------
Writing the same try/except into five handlers is five places to forget a
new error type. `_call` centralises it: one mapping table, applied
everywhere, so adding `LLMTimeoutError` later means editing one line and
every endpoint inherits it.

The mapping is a table rather than a chain of `if isinstance(...)` for the
same reason — it is declarative, and a reader can see all six outcomes at
once.
"""

from typing import Any, NoReturn

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from backend.schemas.study import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    ExplainRequest,
    ExplainResponse,
    InterviewRequest,
    InterviewResponse,
    QuizRequest,
    QuizResponse,
    SummariseRequest,
    SummariseResponse,
    UsageModel,
    error_responses,
)
from backend.services import study_service
from backend.services.errors import (
    LLMConfigurationError,
    LLMConnectionError,
    LLMContextLengthError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
)
from backend.services.groq_client import ChatMessage
from backend.services.prompts import PromptValidationError
from backend.services.study_service import StudyResult
from backend.utils.logger import get_logger

router = APIRouter(prefix="/api/v1", tags=["study"])
logger = get_logger(__name__)

# How long we tell a throttled client to wait. Groq's free tier refills on a
# per-minute window, so 30s is a realistic retry point rather than an
# optimistic 1s that would just fail again.
_RETRY_AFTER_SECONDS = 30


# =============================================================================
# A typed HTTP error
# =============================================================================
class StudyAPIError(Exception):
    """An error this API knows how to describe completely.

    WHY NOT plain HTTPException?
    ---------------------------
    FastAPI's built-in handler renders HTTPException as `{"detail": "..."}` —
    one key. That cannot carry the machine-readable `error` code or the
    `retry_after_seconds` value that ErrorResponse promises, so documenting
    ErrorResponse while raising HTTPException would make our own Swagger a
    lie. A dedicated type plus a handler (registered in main.py) lets the
    body match the documented schema exactly.

    Carrying the data as fields rather than as headers means the handler does
    no parsing — it just serialises what it was given.
    """

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        detail: str,
        retry_after_seconds: int | None = None,
    ) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.code = code
        self.detail = detail
        self.retry_after_seconds = retry_after_seconds


async def study_api_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Render a StudyAPIError using the documented ErrorResponse shape.

    Registered in main.py with `add_exception_handler`, because exception
    handlers belong to the application, not to a router.

    Headers are duplicated into HTTP headers as well as the body: `Retry-After`
    is a standard header that HTTP clients and proxies honour automatically,
    while the body field is there for code that only reads JSON.
    """
    assert isinstance(exc, StudyAPIError)  # narrows the type for the checker

    headers = {"X-Error-Code": exc.code}
    if exc.retry_after_seconds is not None:
        headers["Retry-After"] = str(exc.retry_after_seconds)

    body = ErrorResponse(
        detail=exc.detail,
        error=exc.code,
        retry_after_seconds=exc.retry_after_seconds,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=body.model_dump(),
        headers=headers,
    )


# =============================================================================
# Exception -> HTTP status mapping
# =============================================================================
# Each entry: exception type -> (status code, machine-readable code, message
# prefix). Order does not matter because these are distinct types, but the
# lookup walks the MRO so a subclass of a mapped type still resolves.
_ERROR_MAP: dict[type[Exception], tuple[int, str]] = {
    # 503: our fault, not the caller's. The server cannot do its only job.
    LLMConfigurationError: (
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "llm_not_configured",
    ),
    # 400: the caller's input is too large for the model.
    LLMContextLengthError: (
        status.HTTP_400_BAD_REQUEST,
        "input_too_long",
    ),
    # 422: well-formed JSON, unusable content. The constant was renamed in
    # newer Starlette (UNPROCESSABLE_CONTENT); the codepoint and meaning are
    # identical, so prefer whichever this install provides.
    PromptValidationError: (
        getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", None)
        or status.HTTP_422_UNPROCESSABLE_ENTITY,
        "invalid_request",
    ),
    # 429: slow down. Includes Retry-After.
    LLMRateLimitError: (
        status.HTTP_429_TOO_MANY_REQUESTS,
        "rate_limit_exceeded",
    ),
    # 504: upstream unreachable — a gateway timeout, not a client error.
    LLMConnectionError: (
        status.HTTP_504_GATEWAY_TIMEOUT,
        "llm_unreachable",
    ),
    # 502: upstream answered with an error.
    LLMProviderError: (
        status.HTTP_502_BAD_GATEWAY,
        "llm_provider_error",
    ),
    # 502: upstream answered, but with something we cannot use.
    LLMResponseError: (
        status.HTTP_502_BAD_GATEWAY,
        "llm_response_unusable",
    ),
}


def _resolve(error: Exception) -> tuple[int, str]:
    """Find the status/code for an exception by walking its class hierarchy.

    Using the MRO rather than an exact type lookup means a future
    `LLMTimeoutError(LLMConnectionError)` is handled correctly without
    touching this function — it inherits its parent's mapping automatically.
    """
    for cls in type(error).__mro__:
        if cls in _ERROR_MAP:
            return _ERROR_MAP[cls]
    # Any LLMError not in the table is still a gateway problem.
    if isinstance(error, LLMError):
        return status.HTTP_502_BAD_GATEWAY, "llm_error"
    # Should be unreachable: _call only catches mapped types.
    return status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error"


def _fail(error: Exception, operation: str) -> NoReturn:
    """Translate an exception into a StudyAPIError and raise it.

    `NoReturn` is the useful part of the signature: it tells the type checker
    this function never falls through, so callers can treat the try block as
    producing a StudyResult or not returning at all.

    The client sees our own message, never the provider's raw error — those
    can contain request IDs and internal detail. The full exception is logged
    server-side, where you can actually debug it.

    """
    status_code, code = _resolve(error)
    retry_after = (
        _RETRY_AFTER_SECONDS
        if status_code == status.HTTP_429_TOO_MANY_REQUESTS
        else None
    )

    # Log everything the provider said, then send only our own message.
    # `provider_detail` exists precisely for the part that is useful to a
    # developer and unsafe to a client — request IDs, echoed input, org IDs.
    provider_detail = getattr(error, "provider_detail", None)
    logger.warning(
        "%s failed: %s (%s)%s",
        operation,
        error,
        type(error).__name__,
        f" | provider detail: {provider_detail}" if provider_detail else "",
    )

    raise StudyAPIError(
        status_code=status_code,
        code=code,
        detail=str(error),
        retry_after_seconds=retry_after,
    ) from error


def _envelope(result: StudyResult) -> Any:
    """Unwrap a StudyResult into the shared response envelope.

    `.model_dump()` on `data` converts the service's Pydantic model into a
    plain dict, which FastAPI then re-validates against the endpoint's
    `response_model`. That second validation is what guarantees a field added
    to an output model cannot silently leak onto the wire without appearing
    in the documented contract.
    """
    return {
        "data": result.data.model_dump(),
        "model": result.model,
        "usage": UsageModel(**result.usage.as_dict()),
    }


def _call(operation: str, fn: Any, /, **kwargs: Any) -> Any:
    """Run one service call, mapping any failure to the right HTTP status.

    The `/` makes everything before it positional-only, so a service keyword
    argument named `operation` could never collide with this parameter.
    """
    try:
        result: StudyResult = fn(**kwargs)
    except (PromptValidationError, LLMError) as error:
        _fail(error, operation)
    return _envelope(result)


# =============================================================================
# POST /summarise
# =============================================================================
@router.post(
    "/summarise",
    response_model=SummariseResponse,
    summary="Summarise notes",
    description=(
        "Condense the supplied notes into a one-sentence summary plus a list "
        "of key points. Every point is drawn from the notes — the model is "
        "instructed not to add outside facts."
    ),
    responses=error_responses(400, 422, 429, 502, 503, 504),
)
def summarise(body: SummariseRequest) -> Any:
    """Summarise the student's notes."""
    return _call(
        "summarise",
        study_service.summarise_notes,
        notes=body.notes,
        style=body.style,
        max_bullets=body.max_bullets,
    )


# =============================================================================
# POST /quiz
# =============================================================================
@router.post(
    "/quiz",
    response_model=QuizResponse,
    summary="Generate a multiple-choice quiz",
    description=(
        "Generate multiple-choice questions answerable from the notes, each "
        "with four options, one correct answer (`answer_index`) and a short "
        "explanation of why it is correct."
    ),
    responses=error_responses(400, 422, 429, 502, 503, 504),
)
def quiz(body: QuizRequest) -> Any:
    """Generate multiple-choice questions from the notes."""
    return _call(
        "quiz",
        study_service.generate_quiz,
        notes=body.notes,
        num_questions=body.num_questions,
        difficulty=body.difficulty,
    )


# =============================================================================
# POST /explain
# =============================================================================
@router.post(
    "/explain",
    response_model=ExplainResponse,
    summary="Explain a concept",
    description=(
        "Explain one concept at the requested level, with an analogy and a "
        "glossary. Supplying `notes` grounds the answer in them; omitting it "
        "answers from general knowledge, in which case `verify_note` tells the "
        "student to double-check against their course material."
    ),
    responses=error_responses(400, 422, 429, 502, 503, 504),
)
def explain(body: ExplainRequest) -> Any:
    """Explain a single concept, optionally grounded in notes."""
    return _call(
        "explain",
        study_service.explain_concept,
        concept=body.concept,
        notes=body.notes,
        level=body.level,
    )


# =============================================================================
# POST /interview-questions
# =============================================================================
@router.post(
    "/interview-questions",
    response_model=InterviewResponse,
    summary="Generate interview questions",
    description=(
        "Turn the notes into interview-style questions for the given role, "
        "each with the focus of the question and what a strong answer covers."
    ),
    responses=error_responses(400, 422, 429, 502, 503, 504),
)
def interview_questions(body: InterviewRequest) -> Any:
    """Generate interview questions from the notes."""
    return _call(
        "interview",
        study_service.generate_interview_questions,
        notes=body.notes,
        role=body.role,
        num_questions=body.num_questions,
        difficulty=body.difficulty,
    )


# =============================================================================
# POST /chat
# =============================================================================
@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a question",
    description=(
        "Answer a study question conversationally. `notes` and `history` are "
        "optional: notes ground the answer, history gives the model context "
        "from earlier turns. This is the only endpoint returning free text "
        "rather than structured JSON."
    ),
    responses=error_responses(400, 422, 429, 502, 503, 504),
)
def chat(body: ChatRequest) -> Any:
    """Answer a conversational question, optionally from notes."""
    # Convert API models into the TypedDicts the service expects. Done here,
    # at the boundary, so the service keeps depending on its own vocabulary
    # rather than on HTTP schemas.
    history: list[ChatMessage] = [
        {"role": turn.role, "content": turn.content} for turn in body.history
    ]
    return _call(
        "chat",
        study_service.chat,
        message=body.message,
        notes=body.notes,
        history=history,
    )
