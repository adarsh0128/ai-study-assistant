"""Business logic and external integrations.

groq_client.py  — the only module that imports `groq`
prompts.py      — the only module holding prompt text
study_service.py — orchestrates the two

Phase 1: structure only.
Phase 3: groq_client.py implemented.
Phase 4: prompts.py implemented — every prompt lives there, with per-feature
         temperature, token budgets, an explicit JSON schema, and prompt
         injection defences around untrusted notes.
Phase 5: study_service.py implemented — one function per feature, with JSON
         extraction, Pydantic output validation, a bounded repair retry,
         token accounting, and provider errors passed through untouched so
         the routers can map them to HTTP status codes.
Phase 6: routers/schemas expose these five functions over HTTP.
"""
