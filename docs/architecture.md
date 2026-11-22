# AI Study Assistant — Architecture

How a click in the browser becomes an LLM answer, and why the code is divided the way
it is.

```
┌─────────────────┐   HTTP/JSON    ┌──────────────────────────────────────┐   HTTPS    ┌───────────┐
│   Streamlit     │ ─────────────▶ │            FastAPI backend           │ ─────────▶ │   Groq    │
│   (port 8501)   │ ◀───────────── │            (port 8000)               │ ◀───────── │   API     │
└─────────────────┘                └──────────────────────────────────────┘            └───────────┘
   frontend/                          backend/                                          external
```

The frontend never sees the API key. The backend never renders UI. Groq never knows
that Streamlit exists.

---

## The request lifecycle

Following one `POST /api/v1/summarise` from click to screen:

```
1.  Streamlit page          pages/1_Summarise.py
    user clicks "Summarise"
        │
        │  validation happens here too, so obvious mistakes cost no tokens
        ▼
2.  api_client.summarise()  frontend/api_client.py
    builds the JSON body, POSTs to BACKEND_URL
        │
        ▼  HTTP over localhost
3.  Uvicorn                 (ASGI server)
    parses the HTTP request into a Python object
        │
        ▼
4.  Router                  routers/study.py
    matches the path, calls study_service.summarise_notes(...)
        │
        ▼
5.  Pydantic schema         schemas/study.py
    validates the request body BEFORE the function body runs.
    Invalid input stops here with 422 — no tokens spent.
        │
        ▼
6.  Service                 services/study_service.py
    picks the prompt, decides the options, calls the provider,
    validates what comes back, retries once if it is malformed
        │
        ├──▶ 7. Prompts        services/prompts.py
        │      builds system + user messages
        │
        └──▶ 8. Provider       services/groq_client.py
               the only file that imports `groq`; sends the request,
               normalises vendor errors into our own exception types
                    │
                    ▼  HTTPS
               ┌─────────┐
               │  Groq   │  runs the model, returns JSON text + token counts
               └─────────┘
                    │
        ◀───────────┘
9.  Service maps the raw text into a Pydantic response model
    (retry once with the validation error if it does not parse)
        │
        ▼
10. Router wraps it in the response envelope: { data, model, usage }
        │
        ▼
11. api_client               frontend/api_client.py
    checks the status code, turns errors into APIError
        │
        ▼
12. Page renders it         pages/1_Summarise.py
    summary, key points, token usage, download button
```

**Two validation points, deliberately.** Step 5 rejects a malformed *request* before any
money is spent. Step 9 checks the model's *response* before it reaches a user. The LLM is
an untrusted external service; its output gets the same scrutiny as user input.

---

## Layers and their responsibilities

| Layer | Files | May import | Must never |
|---|---|---|---|
| **UI** | `frontend/app.py`, `pages/`, `ui.py` | `api_client`, `ui` | call Groq, read the API key, build URLs |
| **HTTP client** | `frontend/api_client.py` | `requests` | render anything, know about Streamlit |
| **Routes** | `backend/routers/` | schemas, services, errors | contain business logic or prompt text |
| **Contract** | `backend/schemas/` | Pydantic | import services or routers |
| **Business logic** | `backend/services/study_service.py` | prompts, groq_client, errors | import FastAPI or `groq` |
| **Prompts** | `backend/services/prompts.py` | (stdlib only) | call the network |
| **Provider** | `backend/services/groq_client.py` | `groq`, config, errors | know what a "summarise" is |
| **Config** | `backend/config.py` | `pydantic-settings` | be imported by prompts |

The import rules are the architecture. A layer may only import from layers below it, and
these four prohibitions are what keep the design honest:

1. **`groq` appears in exactly one file.** Swapping to another provider — or to a local
   model — means rewriting `groq_client.py` behind the same function signatures. Nothing
   above it changes.

2. **Prompt text appears in exactly one file.** Prompt engineering is iterative: you
   reword, re-test, reword again. Keeping every prompt in `prompts.py` means tuning is a
   single-file diff, and prompt tests can import them directly.

3. **`study_service.py` imports neither FastAPI nor `groq`.** It takes plain arguments
   and returns plain dataclasses. That is why 69 of the tests can exercise real business
   logic with no server and no API key.

4. **`api_client.py` is the only file that knows the backend's URL.** Pages call
   `api_client.summarise(notes)`. If the API moves to `/v2` or a different host, one file
   changes.

---

## Why there is a backend at all

For a personal script, Streamlit could call Groq directly and you would delete half this
project. The backend exists for reasons that matter as soon as more than one person uses
it:

- **The key stays server-side.** A Streamlit app runs partly in the user's browser; a key
  used there is a key you have published. Here the browser only ever talks to FastAPI.
- **One place to enforce rules.** Length caps, prompt choice, retry policy and error
  mapping live in the backend. Every client inherits them.
- **A decoupled UI.** The same API serves Streamlit today, and could serve a mobile app
  or a CLI tomorrow without touching business logic.
- **Testability.** A router callable with `TestClient` and a service callable with plain
  arguments are both easy to test. A Streamlit script is not.

---

## Data contracts

### The response envelope

Every successful call returns the same three keys:

```json
{ "data": {...}, "model": "qwen/qwen3.8-27b",
  "usage": { "prompt_tokens": 295, "completion_tokens": 94, "total_tokens": 389,
             "repair_attempts": 0, "latency_ms": 1220 } }
```

- `data` — endpoint-specific payload, itself validated by a Pydantic model.
- `model` — which model actually answered. Useful the moment you change `GROQ_MODEL`
  and forget which run produced which output.
- `usage` — `repair_attempts` is our own counter, not Groq's: it records how many times
  the model returned malformed JSON and had to be re-asked. Watching it climb is how you
  notice a prompt drifting.

A single envelope means one response handler in the frontend and one place to add fields
later.

### Typed errors

Exceptions are defined in `services/errors.py`, mapped to HTTP status codes in
`routers/study.py`, and rendered by one handler in `main.py`:

| Exception | Status | Code |
|---|---|---|
| `LLMConfigurationError` | 503 | `llm_not_configured` |
| `LLMContextLengthError` | 400 | `input_too_long` |
| `PromptValidationError` | 422 | `invalid_request` |
| `LLMRateLimitError` | 429 | `rate_limit_exceeded` |
| `LLMConnectionError` | 504 | `llm_unreachable` |
| `LLMProviderError` | 502 | `llm_provider_error` |
| `LLMResponseError` | 502 | `llm_response_unusable` |

The lookup walks the exception's MRO, so a future `LLMTimeoutError(LLMConnectionError)`
inherits the right mapping without edits.

**The split between `str(error)` and `provider_detail` is a security boundary.**
`str(error)` is written for the user and may be returned. `provider_detail` holds Groq's
raw text — request IDs, organisation IDs, sometimes the caller's own input echoed back —
and is logged server-side only. Tests assert it never reaches the wire.

---

## How the LLM is constrained

An LLM returns free text. Four layers turn that into something a program can trust:

1. **Structured prompting.** The prompt specifies the exact JSON shape, with a worked
   example. Fewer surprises than describing it in prose.
2. **Schema validation.** The parsed JSON must satisfy a Pydantic model. Wrong types,
   missing fields and out-of-range values are rejected.
3. **One repair attempt.** On failure, the model is re-asked with the validation error
   included, so it can see what was wrong. This fixes most near-misses cheaply.
4. **Honest failure.** If the second attempt also fails, the request returns
   `502 llm_response_unusable`. A clear error beats plausible-looking garbage.

**Grounding.** Prompts for summarise, quiz and interview-questions state that the notes
are the only source and instruct the model not to add outside facts. The notes are
delimited in the user message as data, and the system prompt says the text inside is
content to work on, never instructions to follow. `explain` is the exception: without
notes it answers from general knowledge, and returns a `verify_note` telling the student
to check against course material — because that is exactly where hallucination is likely.

**Temperature** is set low for structured tasks (deterministic JSON) and higher where
prose variety helps. It is a per-task decision in `study_service.py`, not one global
setting.

---

## Testing strategy

| File | What it proves | Needs a key? |
|---|---|---|
| `test_health.py` | health endpoint and its shape | no |
| `test_study.py` | every endpoint: happy path, validation, error mapping, envelope | no |
| `test_study_service.py` | business logic, retry/repair, grounding rules | no |
| `test_prompts.py` | prompt structure and anti-injection framing | no |
| `test_groq_client.py` | vendor error → our exception mapping | no |
| `test_frontend.py` | request building, error translation, no live server | no |
| `test_live_*.py` | the real API still behaves | **yes** |

`pytest` runs everything except `live`, so the default path is offline, free and fast.
`pytest -m live` opts in. The safe option is the automatic one.

Tests import the app directly via `TestClient`, and mock at the service boundary — the
same seam the router uses — so a passing test means the actual route, schema and error
handler work together, not just that a function returns a value.

---

## Security posture

| Concern | Handling |
|---|---|
| Key leakage in code | read from `.env` only; `.env` git-ignored; `.env.example` holds placeholders |
| Key leakage in responses | canary key planted in tests, asserted absent from every response body |
| Key leakage to the browser | browser talks only to FastAPI; the key never leaves the server |
| Provider detail leakage | raw error text logged, not returned |
| Prompt injection via notes | notes wrapped and labelled as data; system prompt forbids following instructions inside them |
| Cost blowout | request length caps plus `max_tokens` per task |
| Runaway retries | bounded SDK retries and one repair attempt |

**Not included, by design:** authentication, per-user rate limiting, persistent storage.
This is a local learning project. Do not expose it to the internet as-is.

---

## Deliberately out of scope for v1

- **RAG and vector databases.** Notes are pasted per request. Retrieval would matter for
  a large corpus, and would add an embedding model, a vector store and a chunking
  strategy — three new failure modes before the basics are solid.
- **LangChain / LangGraph.** The orchestration here is a function call and a retry. A
  framework would hide exactly the mechanics worth learning.
- **Conversation memory beyond the request.** `chat` accepts `history` from the client
  rather than storing it server-side. Statelessness keeps the backend simple to reason
  about and to scale.

Each is a natural next step once the current layer is understood.
