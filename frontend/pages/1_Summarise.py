"""Page 1 — Summarise notes.

Wrapped in a function via `main()` purely for readability; Streamlit runs the
whole file top to bottom when the page is opened. There is no `if __name__`
guard needed because nothing imports this file.
"""

import api_client
import streamlit as st
import ui

ui.page_setup("Summarise", "📝")

st.title("📝 Summarise your notes")
st.caption("Condense long notes into a short summary plus the key points worth remembering.")

# --- Controls -------------------------------------------------------------
# These run before the button so the widget value is always read on every
# rerun, whether or not the button was clicked.
style_label = st.radio(
    "Summary style",
    options=["Bullet points", "Paragraph"],
    horizontal=True,
    help=(
        "**Bullet points** returns a one-line summary plus a list of key points. "
        "**Paragraph** returns flowing prose, better for reading in one pass."
    ),
)

left, right = st.columns([3, 1])
with left:
    max_bullets = st.slider(
        "Maximum key points",
        min_value=3,
        max_value=12,
        value=6,
        help="Only used for the bullet-points style. More points means more tokens and a slightly slower response.",
    )
with right:
    st.write("")  # small vertical alignment nudge for the slider

# The UI label is friendly; the API expects the enum value.
style = "bullets" if style_label == "Bullet points" else "paragraph"

notes = ui.notes_input(key="summarise_notes")
if notes:
    st.caption(f"{len(notes):,} characters in your notes")

# --- Action --------------------------------------------------------------
# st.button returns True only on the run triggered by the click, so this block
# executes exactly once per click — not on every subsequent rerun.
if st.button("Summarise", type="primary", width="stretch"):
    if ui.validate_notes(notes):
        # A spinner is shown while the request is in flight. An LLM call takes
        # seconds, and a frozen page looks broken.
        with st.spinner("Summarising…"):
            try:
                result = api_client.summarise(notes, style=style, max_bullets=max_bullets)
            except api_client.APIError as error:
                ui.show_error(error)
            else:
                # `result` is the full envelope: {"data": ..., "model": ..., "usage": ...}.
                # The actual summary lives under "data".
                st.session_state["summarise_result"] = result

# --- Render --------------------------------------------------------------
# Reading from session_state (rather than only from the block above) is what
# makes the result SURVIVE a rerun. Change the style radio after summarising
# and the page reruns — the result stays on screen instead of vanishing.
result = st.session_state.get("summarise_result")
if result:
    data = result["data"]

    st.divider()
    st.subheader("Summary")
    st.write(data["summary"])

    key_points = data.get("key_points") or []
    if key_points:
        st.subheader(f"Key points ({len(key_points)})")
        for point in key_points:
            st.markdown(f"- {point}")

    ui.show_usage(result)

    # Plain-text export built from the same fields that were rendered above.
    export = [data["summary"], "", "Key points:"]
    export += [f"- {p}" for p in key_points]
    ui.download_button("\n".join(export), filename="summary.txt")
