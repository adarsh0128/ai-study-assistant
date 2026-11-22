"""Business logic for every study feature — the layer between HTTP and the LLM.

WHAT THIS MODULE DOES
---------------------
A router's only job is HTTP: read the request, call a function here, return
the result. This module owns everything else:

    input validation -> prompt construction -> LLM call -> response parsing
                     -> schema validation -> result assembly + logging

Keeping that sequence here (not in the router) means the same logic can be
called from a test, a CLI, or a background job without an HTTP client.

--------------------------------------------------------------------------------
THE HARD PART: TRUSTING LLM OUTPUT
--------------------------------------------------------------------------------
`chat_completion` returns a string. For `summarise`, `quiz`, `explain` and
`interview_questions` that string is supposed to be JSON matching a shape we
described in the prompt. But a prompt is a request, not a guarantee. What can
actually come back:

    1. Perfect JSON, correct shape                  -> happy path
    2. JSON wrapped in ```json fences or prose      -> strip the wrapper
    3. Valid JSON, wrong shape (list not dict,
       options=3 instead of 4, answer_index=7)      -> coerce or reject
    4. Truncated JSON, cut off mid-string           -> retry with more budget
    5. Not JSON at all                              -> repair retry, then 502

We handle these in that order. Each step narrows what we accept, so a bad
response fails here — in one place, with a clear error — instead of producing
a KeyError deep inside the frontend later.

WHY PYDDANTIC MODELS FOR OUTPUT
    JSON mode guarantees *syntactically* valid JSON. It says nothing about
    our schema: the model can return `"options": "A, B, C"` and be perfectly
    valid JSON. Pydantic validates types and bounds, so
    `answer_index: 7` or `options` with three entries is caught immediately.

WHY EXACTLY ONE REPAIR RETRY
    Retrying on malformed output does help — usually the same prompt with the
    parse error appended works. But retrying forever multiplies cost and
    latency. One repair attempt covers the common transient case; after that
    the error is real and should surface.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, Field, ValidationError, field_validator

from backend.services import prompts
from backend.services.errors import LLMResponseError
from backend.services.groq_client import ChatMessage, chat_completion
from backend.services.prompts import PromptSpec, PromptValidationError
from backend.utils.logger import get_logger

logger = get_logger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)


# =============================================================================
# Output schemas — the contract we hold the model to
# =============================================================================
class KeyTerm(BaseModel):
    """One glossary entry inside an explanation."""

    term: str = Field(min_length=1)
    definition: str = Field(min_length=1)


class SummaryOutput(BaseModel):
    """Shape of a summarise response."""

    summary: str = Field(min_length=1)
    key_points: list[str] = Field(min_length=1)


class QuizQuestion(BaseModel):
    """One multiple-choice question.

    The validator is the important part: a model that returns three options,
    or an `answer_index` of 7, has produced something the UI cannot render
    correctly. Better to reject it here than to let a student see a broken
    question — or, worse, have a correct answer graded wrong.
    """

    question: str = Field(min_length=1)
    # The prompt asks for exactly 4. Validation accepts 2-6 rather than
    # exactly 4 on purpose: an otherwise-correct question with 3 options is
    # still usable, and rejecting it would turn a cosmetic deviation into a
    # 502. The upper bound stops a model that dumps a whole list as options.
    options: list[str] = Field(min_length=2, max_length=6)
    answer_index: int = Field(ge=0)
    explanation: str = Field(default="")

    @field_validator("options")
    @classmethod
    def _options_not_blank(cls, value: list[str]) -> list[str]:
        cleaned = [o.strip() for o in value]
        if any(not o for o in cleaned):
            raise ValueError("options must not contain blank entries")
        return cleaned

    def model_post_init(self, __context: Any) -> None:
        """Cross-field check: the answer must point at a real option."""
        if self.answer_index >= len(self.options):
            raise ValueError(
                f"answer_index {self.answer_index} is out of range for "
                f"{len(self.options)} options"
            )


class QuizOutput(BaseModel):
    """Shape of a quiz response."""

    questions: list[QuizQuestion] = Field(min_length=1)


class InterviewQuestion(BaseModel):
    """One interview question plus what a strong answer covers."""

    question: str = Field(min_length=1)
    focus: str = Field(default="")
    difficulty: str = Field(default="")
    what_a_strong_answer_covers: list[str] = Field(default_factory=list)


class InterviewOutput(BaseModel):
    """Shape of an interview-questions response."""

    questions: list[InterviewQuestion] = Field(min_length=1)


class ExplainOutput(BaseModel):
    """Shape of an explain response."""

    explanation: str = Field(min_length=1)
    analogy: str = Field(default="")
    key_terms: list[KeyTerm] = Field(default_factory=list)
    verify_note: str = Field(default="")


class ChatOutput(BaseModel):
    """Plain-text chat reply, wrapped so chat matches the other features.

    Every feature returns a `StudyResult` whose `data` is a Pydantic model.
    Chat has no structure to validate, but wrapping the string keeps one
    response envelope across the API instead of special-casing one endpoint.
    """

    reply: str = Field(min_length=1)


# =============================================================================
# Results — what a feature returns to the router
# =============================================================================
@dataclass(frozen=True)
class Usage:
    """Token accounting for one completed feature call.

    Surfaced to the caller rather than only logged: cost visibility belongs
    next to the response, and tests can assert the numbers are real.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    repair_attempts: int = 0
    latency_ms: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "repair_attempts": self.repair_attempts,
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True)
class StudyResult:
    """A parsed, validated result plus provenance.

    Generic over the payload so each feature keeps its own typed model while
    sharing the travel arrangements.
    """

    data: BaseModel
    model: str
    usage: Usage


# =============================================================================
# JSON extraction — steps 2 and 4 of the trust problem
# =============================================================================
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull a JSON object out of whatever the model actually returned.

    Order matters:
      1. Try the whole string. Cheap and correct when the model behaves.
      2. Try the contents of a ```json fence. Models add fences constantly
         despite being told not to.
      3. Fall back to the outermost {...} span, in case prose surrounds it.

    Raises:
        LLMResponseError: no JSON object could be recovered.
    """
    if not text or not text.strip():
        raise LLMResponseError("The model returned an empty response.")

    candidates = [text.strip()]

    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1).strip())

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            # Some models answer with a bare array. Wrap it so downstream
            # validation can decide whether that is acceptable.
            return {"questions": parsed}

    # Nothing parseable. Truncation is the likely cause, so say so.
    preview = text.strip()[:120].replace("\n", " ")
    raise LLMResponseError(
        "The model's response was not valid JSON. "
        f"Response started with: {preview!r}"
    )


# =============================================================================
# The core pipeline — used by every JSON feature
# =============================================================================
def _call_and_parse(
    spec: PromptSpec,
    schema: type[TModel],
    *,
    repair_retry: bool = True,
) -> tuple[TModel, str, Usage]:
    """Run a prompt, parse the JSON, validate against a schema.

    This single function is why every feature behaves consistently: same
    retry policy, same validation, same logging, same error types.

    Returns:
        (validated model, model name, usage)

    Raises:
        LLMResponseError: the response could not be parsed or validated,
            even after a repair attempt.
        (plus every error from `chat_completion`, passed through untouched)
    """
    started = time.perf_counter()
    attempts = 0

    result = chat_completion(
        spec.messages,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        json_mode=spec.json_mode,
    )
    prompt_tokens = result["prompt_tokens"]
    completion_tokens = result["completion_tokens"]
    total_tokens = result["total_tokens"]

    try:
        parsed = extract_json_object(result["text"])
        validated = schema.model_validate(parsed)
        latency = int((time.perf_counter() - started) * 1000)
        return (
            validated,
            result["model"],
            Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                repair_attempts=attempts,
                latency_ms=latency,
            ),
        )
    except (LLMResponseError, ValidationError) as first_error:
        if not repair_retry:
            raise _as_response_error(first_error, spec.name) from first_error

        # --- One repair attempt -------------------------------------------
        # Tell the model exactly what was wrong. Vague "try again" rarely
        # helps; the specific error usually does.
        logger.warning(
            "%s: response failed validation, attempting one repair (%s)",
            spec.name,
            type(first_error).__name__,
        )
        attempts = 1

        repair_messages: list[ChatMessage] = list(spec.messages)
        # Echo the bad output back so the model sees what it produced. This
        # costs tokens but is far more effective than a blind retry.
        repair_messages.append(
            {"role": "assistant", "content": result["text"]}
        )
        repair_messages.append(
            {
                "role": "user",
                "content": (
                    "That response could not be used. Problem: "
                    f"{_describe_error(first_error)}. "
                    "Return ONLY corrected JSON that matches the required "
                    "schema exactly. No prose, no markdown fences."
                ),
            }
        )

        # A bigger budget helps when the first attempt was truncated.
        repair_max_tokens = min(
            (spec.max_tokens or 1024) * 2, prompts.MAX_COMPLETION_TOKENS
        )

        repair_result = chat_completion(
            repair_messages,
            temperature=spec.temperature,
            max_tokens=repair_max_tokens,
            json_mode=spec.json_mode,
        )
        # Repair costs tokens too, so the totals include both calls. Hiding
        # the cost of a retry would make your usage numbers lie.
        prompt_tokens += repair_result["prompt_tokens"]
        completion_tokens += repair_result["completion_tokens"]
        total_tokens += repair_result["total_tokens"]

        try:
            parsed = extract_json_object(repair_result["text"])
            validated = schema.model_validate(parsed)
        except (LLMResponseError, ValidationError) as second_error:
            logger.error(
                "%s: repair attempt failed too (%s)",
                spec.name,
                type(second_error).__name__,
            )
            raise _as_response_error(
                second_error, spec.name, repaired=True
            ) from second_error

        latency = int((time.perf_counter() - started) * 1000)
        logger.info("%s: repair attempt succeeded", spec.name)
        return (
            validated,
            repair_result["model"],
            Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                repair_attempts=attempts,
                latency_ms=latency,
            ),
        )


def _describe_error(error: Exception) -> str:
    """Turn a validation/parse error into a short message for the repair turn.

    Pydantic's full message is long and stack-shaped. The model needs the
    gist — which field, what was wrong — not a traceback.
    """
    if isinstance(error, ValidationError):
        parts: list[str] = []
        for err in error.errors()[:4]:
            location = ".".join(str(p) for p in err["loc"]) or "response"
            parts.append(f"{location}: {err['msg']}")
        return "; ".join(parts)
    return str(error)


def _as_response_error(
    error: Exception, feature: str, *, repaired: bool = False
) -> LLMResponseError:
    """Wrap a parse/validation failure as a single, user-facing error type."""
    suffix = " (also failed after one repair attempt)" if repaired else ""
    return LLMResponseError(
        f"The model returned a response we could not use for "
        f"'{feature}'{suffix}: {_describe_error(error)}"
    )


def _log_success(spec: PromptSpec, usage: Usage, model: str) -> None:
    """One structured log line per completed call."""
    logger.info(
        "feature=%s model=%s prompt_tokens=%d completion_tokens=%d "
        "total_tokens=%d repairs=%d latency_ms=%d",
        spec.name,
        model,
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.total_tokens,
        usage.repair_attempts,
        usage.latency_ms,
    )


# =============================================================================
# Feature 1 — Summarise
# =============================================================================
def summarise_notes(
    notes: str,
    *,
    style: str = "bullets",
    max_bullets: int = 6,
) -> StudyResult:
    """Condense notes into a summary plus key points.

    Args:
        notes: the student's text. Untrusted; wrapped by the prompt layer.
        style: "bullets" or "paragraph".
        max_bullets: upper bound on key points.

    Raises:
        PromptValidationError: notes or options are unusable (-> 422).
        LLMError subclasses: provider problems.
        LLMResponseError: output unusable even after repair (-> 502).
    """
    spec = prompts.build_summarise_prompt(
        notes, style=style, max_bullets=max_bullets
    )
    data, model, usage = _call_and_parse(spec, SummaryOutput)
    _log_success(spec, usage, model)
    logger.debug("summarise produced %d key points", len(data.key_points))
    return StudyResult(data=data, model=model, usage=usage)


# =============================================================================
# Feature 2 — Quiz
# =============================================================================
def generate_quiz(
    notes: str,
    *,
    num_questions: int = 5,
    difficulty: str = "medium",
) -> StudyResult:
    """Generate multiple-choice questions from notes."""
    spec = prompts.build_quiz_prompt(
        notes, num_questions=num_questions, difficulty=difficulty
    )
    data, model, usage = _call_and_parse(spec, QuizOutput)
    _log_success(spec, usage, model)

    # The model may return a different count than requested. That is not an
    # error — it is information the caller should see, so it gets logged
    # rather than silently accepted.
    if len(data.questions) != num_questions:
        logger.info(
            "quiz requested %d questions, model returned %d",
            num_questions,
            len(data.questions),
        )
    return StudyResult(data=data, model=model, usage=usage)


# =============================================================================
# Feature 3 — Explain
# =============================================================================
def explain_concept(
    concept: str,
    *,
    notes: str | None = None,
    level: str = "beginner",
) -> StudyResult:
    """Explain a single concept, optionally grounded in notes."""
    spec = prompts.build_explain_prompt(concept, notes=notes, level=level)
    data, model, usage = _call_and_parse(spec, ExplainOutput)
    _log_success(spec, usage, model)
    return StudyResult(data=data, model=model, usage=usage)


# =============================================================================
# Feature 4 — Interview questions
# =============================================================================
def generate_interview_questions(
    notes: str,
    *,
    role: str = "software engineer",
    num_questions: int = 5,
    difficulty: str = "medium",
) -> StudyResult:
    """Turn notes into interview-style questions with answer guidance."""
    spec = prompts.build_interview_prompt(
        notes,
        role=role,
        num_questions=num_questions,
        difficulty=difficulty,
    )
    data, model, usage = _call_and_parse(spec, InterviewOutput)
    _log_success(spec, usage, model)
    return StudyResult(data=data, model=model, usage=usage)


# =============================================================================
# Feature 5 — Chat
# =============================================================================
def chat(
    message: str,
    *,
    notes: str | None = None,
    history: list[ChatMessage] | None = None,
) -> StudyResult:
    """Answer a question conversationally (plain text, the one non-JSON feature).

    There is no schema to validate here, so no repair retry: an empty reply is
    already caught by `chat_completion`, and there is no structure to repair.
    A second call would just be a guess.
    """
    spec = prompts.build_chat_prompt(message, notes=notes, history=history)
    started = time.perf_counter()

    result = chat_completion(
        spec.messages,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        json_mode=spec.json_mode,
    )

    text = result["text"].strip()
    if not text:
        raise LLMResponseError("The model returned an empty chat response.")

    usage = Usage(
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        total_tokens=result["total_tokens"],
        repair_attempts=0,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
    _log_success(spec, usage, result["model"])

    # Chat returns a plain-text model so the router can wrap it consistently
    # with the other features.
    return StudyResult(
        data=ChatOutput(reply=text),
        model=result["model"],
        usage=usage,
    )


# =============================================================================
# Health/preflight helper
# =============================================================================
def prompt_budget_report() -> dict[str, int | str]:
    """Static facts about the prompt budget, for the health endpoint.

    Exposing this makes the context-window limits visible instead of buried
    in a constant, which is genuinely useful when a user reports that their
    notes will not fit.
    """
    return {
        "context_window_tokens": prompts.CONTEXT_WINDOW_TOKENS,
        "max_completion_tokens": prompts.MAX_COMPLETION_TOKENS,
        "max_notes_chars": prompts.MAX_NOTES_CHARS,
        "min_notes_chars": prompts.MIN_NOTES_CHARS,
        "features": len(prompts.PROMPT_BUILDERS),
    }
