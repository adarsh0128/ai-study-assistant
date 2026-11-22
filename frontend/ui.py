"""Shared Streamlit UI pieces used by every page.

WHY THIS FILE EXISTS
--------------------
The approved architecture lists no shared frontend helper, but three pages need
the same four things: a labelled notes box, a way to render a backend error, a
token-usage caption, and the envelope unpacking. Copy-pasting those into each
page would mean fixing every bug in five places — the same reason the backend
has `study_service` instead of logic in each router.

This module holds presentation and nothing else. It never calls the API
directly; pages call `api_client` and hand the result here to be drawn.

THE STREAMLIT EXECUTION MODEL (important, and surprising at first)
------------------------------------------------------------------
Streamlit re-runs the *entire script* top to bottom on every interaction —
every button click, every text edit. Nothing persists between runs except
`st.session_state`.

That is why:
  * widget values are read into local variables each run, not "kept",
  * results that must survive a rerun live in `st.session_state`,
  * a click is detected as a boolean (`if st.button(...)`) that is True only
    on the run triggered by that click.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

# NOTE ON THIS IMPORT — it looks wrong but is correct.
#
# `streamlit run frontend/app.py` puts the *main script's directory*
# (`frontend/`) on `sys.path` — not the project root. So inside the frontend,
# `from frontend import api_client` raises ModuleNotFoundError, while
# `import api_client` works. Verified against Streamlit's own bootstrap code.
# Keep every frontend import flat for this reason.
import api_client

# Mirrors of the backend's limits (backend/services/prompts.py). The backend
# remains the authority — it validates every request regardless of what the UI
# allows. These exist so the UI can prevent a doomed request instead of
# letting it round-trip to a 422, and so the character counter is meaningful.
MIN_NOTES_CHARS = 40
MAX_CHAT_MESSAGE_CHARS = 4000
MAX_EXPLAIN_CONCEPT_CHARS = 300


def page_setup(title: str, icon: str = "📚") -> None:
    """Apply the shared page config. Must be the first Streamlit call on a page.

    `set_page_config` can only be called once per script run and only before
    any other Streamlit output, which is why each page calls this first.
    """
    st.set_page_config(page_title=f"{title} · AI Study Assistant", page_icon=icon, layout="wide")
    render_backend_status()


def render_backend_status() -> None:
    """Show a one-line banner if the backend is unreachable.

    Called on every page load because the most common beginner failure is
    starting the frontend and forgetting the backend. Detecting it here turns
    a confusing "connection_error" on submit into an immediate explanation.
    """
    try:
        info = api_client.health()
    except api_client.APIError:
        st.error(
            "**Backend not reachable.** Start it in another terminal:\n\n"
            "```\nuvicorn backend.main:app --reload\n```"
        )
        return

    version = info.get("version", "?")
    st.caption(f"Backend connected · v{version} · {api_client.get_backend_url()}")


def notes_input(
    *,
    label: str = "Your notes",
    key: str,
    height: int = 260,
    help_text: str | None = None,
) -> str:
    """Render the notes text area and return its current contents.

    Args:
        label: field label shown above the box.
        key: unique `session_state` key. Streamlit requires keys to be unique
            across widgets in a run, so each page passes its own.
        height: pixel height, tall enough that a real paragraph is readable.
        help_text: optional tooltip explaining what good notes look like.

    Returns:
        The text, stripped. Blank means "nothing entered yet".
    """
    return st.text_area(
        label,
        key=key,
        height=height,
        help=help_text
        or (
            "Paste the notes you want to study from. The more specific they are, "
            "the more grounded the answer — the model is told to stay inside "
            "what you give it."
        ),
        placeholder=(
            "e.g. Photosynthesis occurs in the chloroplast. Chlorophyll absorbs "
            "light energy, which drives the light-dependent reactions..."
        ),
    ).strip()


def validate_notes(notes: str) -> bool:
    """Check notes are long enough, showing guidance when they are not.

    A very short input is refused by the backend (min 40 characters) because
    there is too little to work from and the model starts inventing — the
    hallmark of hallucination. Catching it here explains *why* rather than
    showing a bare 422.
    """
    if not notes:
        st.info("Paste some notes above to get started.")
        return False
    if len(notes) < MIN_NOTES_CHARS:
        remaining = MIN_NOTES_CHARS - len(notes)
        st.warning(
            f"Notes are too short ({len(notes)} characters). Add at least "
            f"**{remaining} more** — with very little text the model has "
            "nothing to ground its answer in and will start making things up."
        )
        return False
    return True


def show_error(error: api_client.APIError) -> None:
    """Render a failed API call in terms the user can act on.

    Every backend failure already carries a human-readable `detail` written to
    be shown, so we display it as-is and add only what the *code* tells us
    about whether retrying is worth it.
    """
    st.error(error.detail)

    if error.code == "rate_limit_exceeded" and error.retry_after_seconds:
        st.info(f"The free tier is rate limited. Try again in about {error.retry_after_seconds}s.")
    elif error.is_retryable:
        st.info("This looks temporary — try again in a moment.")
    elif error.status_code == 422:
        st.info("Check the input above: something in it was outside the allowed range.")
    elif error.status_code == 503:
        st.info("The server has no usable API key. Add `GROQ_API_KEY` to `.env` and restart it.")


def show_usage(envelope: dict[str, Any]) -> None:
    """Caption the model name and token usage for one response.

    Shown on every result on purpose: token counts are the unit you are billed
    in, and seeing them next to each answer is the fastest way to build
    intuition for how prompt size drives cost.
    """
    usage = envelope.get("usage") or {}
    model = envelope.get("model", "unknown")

    parts = [f"`{model}`"]
    if usage.get("total_tokens") is not None:
        parts.append(
            f"{usage.get('prompt_tokens', 0)} in + "
            f"{usage.get('completion_tokens', 0)} out = "
            f"**{usage['total_tokens']} tokens**"
        )
    if usage.get("latency_ms") is not None:
        parts.append(f"{usage['latency_ms'] / 1000:.2f}s")
    # Only mentioned when it happened, since a repair means the model's first
    # answer was unusable — worth knowing, but not worth alarming about.
    if usage.get("repair_attempts"):
        parts.append("repaired once")

    st.caption(" · ".join(parts))


def download_button(data: str, *, filename: str, label: str = "Download as text") -> None:
    """Offer a small text download of a result."""
    st.download_button(label, data=data, file_name=filename, mime="text/plain")
