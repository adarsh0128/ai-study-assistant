"""Provider-agnostic error types for LLM calls.

Why not let Groq's own exceptions bubble up to the routers?

Because then every router would import the Groq SDK just to catch its
errors, and swapping providers would mean editing every layer. These types
are the *contract*: `groq_client.py` translates provider errors into them,
and nothing above that layer knows Groq exists.

Each type maps to one HTTP status code in the routers (Phase 6):

    LLMConfigurationError  -> 503  (server is misconfigured, not your fault)
    LLMContextLengthError  -> 400  (input too big)
    LLMConnectionError     -> 504  (upstream unreachable / timed out)
    LLMRateLimitError      -> 429  (slow down)
    LLMProviderError       -> 502  (upstream returned an error)
    LLMResponseError       -> 502  (upstream returned something unusable)
"""


class LLMError(Exception):
    """Base class for every LLM-related failure."""


class LLMConfigurationError(LLMError):
    """The API key is missing, wrong, or lacks permission."""


class LLMContextLengthError(LLMError):
    """The prompt plus history exceeds the model's context window."""


class LLMConnectionError(LLMError):
    """Could not reach the provider, or the request timed out."""


class LLMRateLimitError(LLMError):
    """The provider is throttling us."""


class LLMProviderError(LLMError):
    """The provider returned an error we cannot classify more precisely.

    `str()` of this exception is sent to the HTTP client, so subclasses carry
    only a message written for a human outside this process. When the provider
    said something useful but unsuitable for the wire, it is attached as
    `provider_detail` — logged server-side, never serialised into a response.
    """

    def __init__(self, message: str, *, provider_detail: str | None = None) -> None:
        super().__init__(message)
        self.provider_detail = provider_detail


class LLMResponseError(LLMError):
    """The provider answered, but with content we cannot use."""
