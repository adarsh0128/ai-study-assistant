"""AI Study Assistant — Streamlit home page.

Run with:
    streamlit run frontend/app.py

HOW STREAMLIT TURNS FILES INTO A SITE
-------------------------------------
Streamlit watches the script you point it at and its `pages/` folder:

    frontend/app.py            -> the home page (this file)
    frontend/pages/1_Summarise.py  -> sidebar entry "Summarise"
    frontend/pages/2_Quiz.py       -> sidebar entry "Quiz"

The number prefix sets the order; the rest of the filename becomes the label
(underscores turn into spaces). No routing code, no HTML — add a file, get a
page. Files in `pages/` are NOT imported by anything; Streamlit runs each one
as its own script when you click it.

Note that `api_client.py` and `ui.py` are deliberately NOT in `pages/` — any
`.py` file in that folder becomes a clickable page, so shared modules live one
level up.
"""

import streamlit as st

import ui
from ui import page_setup

# First Streamlit call on the page: sets the browser tab title, the favicon,
# and the layout width. Must come before any other st.* output.
page_setup("Home", "📚")

st.title("📚 AI Study Assistant")
st.caption("An LLM-powered study companion — summaries, quizzes, explanations, and interview prep from your own notes.")

st.markdown(
    """
### What it does

Every feature takes **your notes** and sends them to a large language model as
context. The model is instructed to stay grounded in what you gave it, which is
what stops it from wandering off and inventing facts.
"""
)

# A three-column row of cards describing the features. Streamlit's `columns`
# lays them out side by side; on a narrow screen they stack automatically.
left, middle, right = st.columns(3)

with left:
    st.subheader("📝 Summarise")
    st.write(
        "Condense long notes into a short summary and a list of key points. "
        "Useful before an exam when you need the shape of a topic fast."
    )
    st.page_link("pages/1_Summarise.py", label="Open Summarise", icon="📝")

with middle:
    st.subheader("❓ Quiz")
    st.write(
        "Generate multiple-choice questions with answers and explanations, "
        "built only from what your notes actually say."
    )
    st.page_link("pages/2_Quiz.py", label="Open Quiz", icon="❓")

with right:
    st.subheader("Coming next")
    st.write(
        "Explain a concept, generate interview questions, and chat with your "
        "notes. These pages are added in later phases once their endpoints exist."
    )

st.divider()

st.markdown(
    """
### How the pieces fit together

```
 Streamlit page  ──HTTP/JSON──▶  FastAPI  ──HTTPS──▶  Groq  ──▶  Llama model
 (your browser)                  (backend)           (inference API)
```

- **Streamlit** draws the interface and collects your notes.
- **FastAPI** validates the request, builds the prompt, and calls the model.
- **Groq** runs the Llama model and returns text; FastAPI parses it into the
  structured shape the UI expects.

The API key that authorises those LLM calls lives only on the backend, in
`.env`. The frontend never sees it — which is why the browser cannot leak it.
"""
)

st.info(
    "**Before using the app:** the backend must be running in another terminal — "
    "`uvicorn backend.main:app --reload`. The status line at the top of each page tells you whether it is."
)
