"""Page 2 — Generate and take a multiple-choice quiz.

THE HARD PART OF THIS PAGE: STATEFUL WIDGETS
--------------------------------------------
Streamlit reruns the whole script on every interaction. Interactive forms like
this quiz have two consequences worth understanding, because both cause
confusing bugs if ignored:

1. **Widget values survive reruns by key.** Each `st.radio` here has a unique
   `key`. When you select an option, Streamlit stores it under that key and
   replays it on the next run. So we never assign the radio's value ourselves —
   we just read it.

2. **Keys must be fresh for a new quiz.** If the user generates a second quiz,
   the old answers would still be sitting under "quiz_q0", "quiz_q1", … and
   would appear pre-selected against the new questions. So each generated quiz
   gets an incrementing id, and that id is part of every widget key. New quiz →
   new keys → clean slate. This is the standard Streamlit pattern for
   "resetting" widgets without touching session_state directly.

Correctness is deliberately checked with a button rather than live, so you can
answer all questions before finding out how you did.
"""

import api_client
import streamlit as st
import ui

ui.page_setup("Quiz", "❓")

st.title("❓ Quiz yourself")
st.caption("Generate multiple-choice questions from your notes, answer them, then check your score.")

# --- Controls -------------------------------------------------------------
left, right = st.columns(2)

with left:
    difficulty = st.select_slider(
        "Difficulty",
        options=["easy", "medium", "hard"],
        value="medium",
        help=(
            "**Easy** tests recall of facts stated directly in your notes. "
            "**Hard** requires applying or connecting ideas. Difficulty changes "
            "the instruction in the prompt, not the model."
        ),
    )

with right:
    num_questions = st.slider(
        "Number of questions",
        min_value=1,
        max_value=20,
        value=5,
        help="Each question adds to the response length and the number of tokens billed.",
    )

notes = ui.notes_input(key="quiz_notes", height=220)
if notes:
    st.caption(f"{len(notes):,} characters in your notes")

# --- Generate -------------------------------------------------------------
if st.button("Generate quiz", type="primary", width="stretch"):
    if ui.validate_notes(notes):
        with st.spinner(f"Writing {num_questions} questions…"):
            try:
                result = api_client.quiz(notes, num_questions=num_questions, difficulty=difficulty)
            except api_client.APIError as error:
                ui.show_error(error)
            else:
                st.session_state["quiz_result"] = result
                # Bump the id so every radio below gets brand-new keys and
                # therefore starts unanswered. Without this, answers from the
                # previous quiz would carry over onto the new questions.
                st.session_state["quiz_id"] = st.session_state.get("quiz_id", 0) + 1
                # Hide the score from any previous quiz.
                st.session_state["quiz_checked"] = False

# --- Render --------------------------------------------------------------
result = st.session_state.get("quiz_result")
if not result:
    st.info("Paste your notes above and press **Generate quiz** to begin.")
    st.stop()

data = result["data"]
questions = data["questions"]
quiz_id = st.session_state.get("quiz_id", 0)
checked = st.session_state.get("quiz_checked", False)

st.divider()
st.subheader(f"{len(questions)} questions · {difficulty}")

# `st.form` batches all the radios and submits them together. Without a form,
# every radio click would rerun the script and call the model again — slow and
# costly. `clear_on_submit=False` keeps selections after submitting so the
# user can review their answers next to the results.
with st.form(key=f"quiz_form_{quiz_id}"):
    selections: list[int | None] = []

    for index, question in enumerate(questions):
        st.markdown(f"**{index + 1}. {question['question']}**")
        options = question["options"]

        choice = st.radio(
            "Choose one",
            options=list(range(len(options))),
            # `format_func` keeps the widget's value as a clean index while
            # showing the user "A) …", "B) …". Never build logic on display text.
            format_func=lambda i, opts=options: f"{'ABCDEFGH'[i]}) {opts[i]}",
            key=f"quiz_q{index}_{quiz_id}",
            index=None,  # nothing pre-selected
            label_visibility="collapsed",
        )
        selections.append(choice)
        st.write("")

    submitted = st.form_submit_button("Check answers", type="primary", width="stretch")

if submitted:
    st.session_state["quiz_checked"] = True
    checked = True

if checked:
    # Score and per-question feedback.
    answered = [(i, s) for i, s in enumerate(selections) if s is not None]
    correct = sum(1 for i, s in answered if s == questions[i]["answer_index"])

    if not answered:
        st.warning("No answers selected yet — pick an option for at least one question.")
    else:
        percent = round(100 * correct / len(questions))
        if percent >= 80:
            st.success(f"**{correct} / {len(questions)}** correct ({percent}%) — solid grasp.")
        elif percent >= 50:
            st.warning(f"**{correct} / {len(questions)}** correct ({percent}%) — review the misses below.")
        else:
            st.error(f"**{correct} / {len(questions)}** correct ({percent}%) — worth re-reading your notes.")

    st.divider()

    for index, question in enumerate(questions):
        options = question["options"]
        answer_index = question["answer_index"]
        chosen = selections[index] if index < len(selections) else None

        if chosen is None:
            icon, verdict = "⬜", "Not answered"
        elif chosen == answer_index:
            icon, verdict = "✅", "Correct"
        else:
            icon, verdict = "❌", "Incorrect"

        with st.expander(f"{icon} Q{index + 1} — {verdict}", expanded=chosen != answer_index):
            st.markdown(f"**{question['question']}**")
            for option_index, option in enumerate(options):
                letter = "ABCDEFGH"[option_index]
                if option_index == answer_index:
                    st.markdown(f"- **{letter}) {option}** ✅ *correct answer*")
                elif option_index == chosen:
                    st.markdown(f"- {letter}) {option} ❌ *your answer*")
                else:
                    st.markdown(f"- {letter}) {option}")

            st.info(f"**Why:** {question['explanation']}")

ui.show_usage(result)

# Text export of the whole quiz with answers marked, so it can be printed or
# reused away from the app.
lines = []
for index, question in enumerate(questions):
    lines.append(f"{index + 1}. {question['question']}")
    for option_index, option in enumerate(question["options"]):
        marker = " *" if option_index == question["answer_index"] else ""
        lines.append(f"   {'ABCDEFGH'[option_index]}) {option}{marker}")
    lines.append(f"   Answer: {'ABCDEFGH'[question['answer_index']]}")
    lines.append(f"   Why: {question['explanation']}")
    lines.append("")
lines.append("(* marks the correct answer)")
ui.download_button("\n".join(lines), filename="quiz.txt")
