"""GET /health — liveness probe.

Deliberately does NOT call Groq. This endpoint answers one question only:
"is the web process up and routing requests?" Keeping it dependency-free
means it stays fast and free, and when it fails you know the problem is
your app rather than the LLM provider.
"""

from fastapi import APIRouter

from backend import __version__
from backend.schemas.health import HealthResponse
from backend.utils.logger import get_logger

router = APIRouter(tags=["health"])
logger = get_logger(__name__)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Returns 200 with a fixed payload when the server is running.",
)
def health_check() -> HealthResponse:
    """Return a static health payload."""
    logger.debug("Health check requested")
    return HealthResponse(
        status="ok",
        service="ai-study-assistant",
        version=__version__,
    )
