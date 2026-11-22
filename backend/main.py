"""Application entry point.

Run in development (auto-reload on file save):

    uvicorn backend.main:app --reload

`backend.main` is the module path; `app` is the FastAPI object below.
Uvicorn imports this module and serves the object it finds.

Phase 2: creates the app, registers /health, exposes Swagger UI at /docs.
Phase 6: registers the study router.
Phase 7: enables CORS so the Streamlit frontend can call this API.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend import __version__
from backend.config import get_settings  # importing also loads .env
from backend.routers import health, study
from backend.routers.study import StudyAPIError, study_api_error_handler
from backend.utils.logger import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Run startup/shutdown logic around the server's lifetime.

    Everything before the `yield` is startup; everything after is shutdown.
    """
    settings = get_settings()

    # Report configuration state without ever printing the key itself.
    logger.info(
        "AI Study Assistant API v%s starting up (env=%s, model=%s, key=%s)",
        __version__,
        settings.app_env,
        settings.groq_model,
        settings.masked_api_key,
    )

    if not settings.is_groq_configured:
        # A missing key is not fatal in development: /health still works,
        # so you can confirm the server is fine before fixing the config.
        # In production it must be fatal — a deployment that cannot serve
        # its only purpose should not accept traffic.
        if settings.is_production:
            settings.require_groq_configured()
        logger.warning(
            "GROQ_API_KEY is not configured. LLM endpoints will return 503 "
            "until you add a real key to .env"
        )

    yield
    logger.info("AI Study Assistant API shutting down")


DESCRIPTION = """
Backend for the **AI Study Assistant**.

Turns pasted study notes into summaries, explanations, quizzes and
interview questions using an LLM (Llama on Groq).

Endpoints are added phase by phase; see `docs/architecture.md`.
"""

app = FastAPI(
    title="AI Study Assistant API",
    description=DESCRIPTION,
    version=__version__,
    docs_url="/docs",      # interactive Swagger UI
    redoc_url="/redoc",    # alternative, read-only docs view
    lifespan=lifespan,
)

# Routers own their own paths; registering them here is the only place that
# knows the complete list of endpoints. As each phase lands, one line changes.
app.include_router(health.router)
app.include_router(study.router)

# --- CORS ------------------------------------------------------------------
# Streamlit (port 8501) and this API (port 8000) are different origins, so a
# browser refuses to let the frontend read our responses until the server says
# the origin is allowed. Without this, every frontend request fails with an
# opaque "CORS policy" TypeError in the browser console while the backend logs
# show nothing at all — a confusing failure worth understanding early.
#
# `allow_origins` is an explicit list from settings (see config.py for why not
# "*"). Credentials stay off: this API is unauthenticated and the browser
# should not attach cookies to it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Exception handlers belong to the app rather than to a router: FastAPI only
# consults handlers registered on the application (or via an APIRouter's own
# handler list). This one renders our error body in the documented
# ErrorResponse shape instead of FastAPI's bare {"detail": ...}.
app.add_exception_handler(StudyAPIError, study_api_error_handler)
