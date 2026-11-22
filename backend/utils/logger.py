"""Shared logging configuration.

Every module in the backend does:

    from backend.utils.logger import get_logger
    logger = get_logger(__name__)

so log lines carry the module that produced them.

Phase 2: reads LOG_LEVEL from the process environment, defaulting to INFO.
Phase 3: forces UTF-8 on stdout so LLM output (which contains characters like
         em dashes and arrows) cannot crash the logger on Windows cp1252
         consoles. config.py is imported before this runs, which puts .env
         values into the environment so LOG_LEVEL from .env is honoured.
"""

import logging
import os
import sys

# Example: 2026-01-15 10:23:41 | INFO     | backend.routers.health | message
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Guards against adding a second handler when configure_logging() is called
# more than once (uvicorn --reload re-imports modules; tests import repeatedly).
_configured = False


def configure_logging(level: str | None = None) -> None:
    """Configure the root logger exactly once.

    Args:
        level: explicit level name (e.g. "DEBUG"). Falls back to the
            LOG_LEVEL environment variable, then to INFO.
    """
    global _configured
    if _configured:
        return

    # Windows consoles default to cp1252, which cannot encode characters LLMs
    # routinely emit (—, ≈, →, ✓). Without this, logging one of those raises
    # UnicodeEncodeError and can take down a request. errors="replace" is the
    # last line of defence so a stray glyph degrades to "?" instead of a crash.
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # Stream already closed or not reconfigurable — logging still works.
            pass

    resolved = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    # getattr() keeps an unknown value like "VERBOSE" from crashing startup —
    # an invalid log level should never take the service down.
    log_level = getattr(logging, resolved, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(log_level)
    root.addHandler(handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name."""
    return logging.getLogger(name)
