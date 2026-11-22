# AI Study Assistant — Architecture

Status: **Phase 1 complete** (structure only). Content filled in during Phase 9.

## Layers

```
Streamlit (8501)  ->  FastAPI (8000)  ->  Groq API
  frontend/            backend/            external
```

| Layer | Files | Responsibility |
|---|---|---|
| HTTP routes | `backend/routers/` | parse request, call service, map errors to status codes |
| Contract | `backend/schemas/` | Pydantic models; validation + Swagger docs |
| Business logic | `backend/services/study_service.py` | orchestration |
| Provider | `backend/services/groq_client.py` | the only `groq` import |
| Prompts | `backend/services/prompts.py` | the only prompt text |
| Config | `backend/config.py` | `.env` -> typed settings |
| UI | `frontend/` | Streamlit pages + `api_client.py` |
