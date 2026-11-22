"""Configuration loading and validation.

Reads settings from `.env` (and from real environment variables, which take
precedence) into a typed, validated object.

Key rules enforced here:
  * The API key is read at runtime from the environment. It is NEVER written
    into source code.
  * A missing or placeholder key does not crash an import — it is detected
    and reported clearly. See `require_groq_configured()`.

Phase 3: implemented.
"""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/config.py -> backend/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# Load .env into os.environ as a side effect of importing this module.
#
# Why, when Settings can read the file itself? Because pydantic-settings
# parses .env into the Settings object WITHOUT putting the values into the
# environment — so backend/utils/logger.py, which reads os.getenv("LOG_LEVEL"),
# would never see it. load_dotenv() bridges that gap.
#
# override=False means a real environment variable (e.g. set by Docker or CI)
# wins over .env, which is the correct precedence for deployments.
load_dotenv(dotenv_path=ENV_FILE, override=False)

# Values that mean "not actually configured". Copied straight from
# .env.example, so a fresh clone is recognised as unconfigured rather than
# being sent to Groq to fail with a confusing 401.
_PLACEHOLDER_KEYS = {
    "",
    "your_groq_api_key_here",
    "changeme",
    "replace_me",
}


class Settings(BaseSettings):
    """Typed application settings, loaded from .env / environment."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # unrelated OS env vars must not break startup
    )

    # --- Groq ---------------------------------------------------------------
    groq_api_key: str = Field(
        default="",
        description="Secret key for the Groq API. Never logged, never committed.",
    )
    # Falls back to this only when GROQ_MODEL is absent from .env. Kept in
    # sync with .env.example so a missing variable does not silently select a
    # decommissioned model. Deliberately NOT an openai/gpt-oss-* model: those
    # are reasoning models that can consume a small max_tokens budget on
    # hidden reasoning and return empty content.
    groq_model: str = Field(default="qwen/qwen3.8-27b")
    groq_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    groq_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="SDK-level retries for transient failures (timeouts, 5xx).",
    )

    # --- Application --------------------------------------------------------
    backend_url: str = Field(default="http://127.0.0.1:8000")
    log_level: str = Field(default="INFO")
    app_env: str = Field(default="development")

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.strip().upper()
        if upper not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(allowed)}, got {value!r}")
        return upper

    @field_validator("app_env")
    @classmethod
    def _normalise_app_env(cls, value: str) -> str:
        return value.strip().lower()

    # --- Derived helpers ----------------------------------------------------
    @property
    def is_groq_configured(self) -> bool:
        """True when a real-looking API key is present."""
        return self.groq_api_key.strip().lower() not in _PLACEHOLDER_KEYS

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def masked_api_key(self) -> str:
        """A safe representation of the key, for logs.

        Shows only enough to confirm *which* key is loaded, never the key.
        """
        key = self.groq_api_key.strip()
        if not key:
            return "<not set>"
        if len(key) <= 8:
            return "<too short to mask safely>"
        return f"{key[:4]}...{key[-4:]} ({len(key)} chars)"

    def require_groq_configured(self) -> None:
        """Raise RuntimeError if no usable API key is configured.

        Called at startup. Raised (rather than merely logged) only in
        production, so a misconfigured deployment dies immediately instead
        of serving errors to users.
        """
        if not self.is_groq_configured:
            raise RuntimeError(
                "GROQ_API_KEY is missing or still a placeholder. "
                f"Add a real key to {ENV_FILE} (see .env.example). "
                "Get one at https://console.groq.com"
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings instance.

    Cached so the .env file is parsed once per process. Tests that need
    different settings call `get_settings.cache_clear()` then re-fetch.
    """
    return Settings()
