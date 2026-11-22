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

from backend import __version__
from backend.routers import health
from backend.utils.logger import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Run startup/shutdown logic around the server's lifetime.

    Everything before the `yield` is startup; everything after is shutdown.
    Phase 3 will validate the GROQ_API_KEY here so a misconfigured server
    fails immediately at boot instead of on the first user request.
    """
    logger.info("AI Study Assistant API v%s starting up", __version__)
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
