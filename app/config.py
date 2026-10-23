import os
from dataclasses import dataclass
from math import isfinite

DEFAULT_MODEL_TIMEOUT_SECONDS = 30.0
DEFAULT_OPENAI_MODEL = "gpt-5-nano"
DEFAULT_OPENAI_MAX_OUTPUT_TOKENS = 512
DEFAULT_RAG_MAX_CONTEXT_TOKENS = 6_000
DEFAULT_RAG_MAX_GENERATION_INPUT_TOKENS = 8_000
DEFAULT_OPENAI_REASONING_EFFORT = "minimal"
DEFAULT_OPENAI_MAX_CONCURRENT_REQUESTS = 2


class ConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    model_timeout_seconds: float
    database_url: str


@dataclass(frozen=True)
class RagGenerationSettings:
    """Hold trusted OpenAI and context limits for Phase 9 generation."""

    openai_api_key: str
    openai_model: str
    timeout_seconds: float
    max_output_tokens: int
    max_context_tokens: int
    max_generation_input_tokens: int
    reasoning_effort: str
    max_concurrent_requests: int
    externally_allowed_visibilities: tuple[str, ...]


def _positive_int_environment(name: str, default: int, *, minimum: int = 1) -> int:
    """Load one bounded-work setting while rejecting booleans and invalid text."""
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be an integer.") from error
    if value < minimum:
        raise ConfigurationError(f"{name} must be at least {minimum}.")
    return value


def load_rag_generation_settings() -> RagGenerationSettings:
    """Load Phase 9 settings without requiring unrelated database configuration."""
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_api_key:
        raise ConfigurationError("OPENAI_API_KEY is required.")
    openai_model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip()
    if not openai_model:
        raise ConfigurationError("OPENAI_MODEL must not be empty.")

    raw_timeout = os.getenv(
        "OPENAI_TIMEOUT_SECONDS",
        os.getenv("MODEL_TIMEOUT_SECONDS", str(DEFAULT_MODEL_TIMEOUT_SECONDS)),
    )
    try:
        timeout_seconds = float(raw_timeout)
    except ValueError as error:
        raise ConfigurationError("OPENAI_TIMEOUT_SECONDS must be positive.") from error
    if not isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ConfigurationError("OPENAI_TIMEOUT_SECONDS must be positive.")

    max_output_tokens = _positive_int_environment(
        "OPENAI_MAX_OUTPUT_TOKENS", DEFAULT_OPENAI_MAX_OUTPUT_TOKENS, minimum=16
    )
    max_context_tokens = _positive_int_environment(
        "RAG_MAX_CONTEXT_TOKENS", DEFAULT_RAG_MAX_CONTEXT_TOKENS
    )
    max_generation_input_tokens = _positive_int_environment(
        "RAG_MAX_GENERATION_INPUT_TOKENS",
        DEFAULT_RAG_MAX_GENERATION_INPUT_TOKENS,
    )
    if max_context_tokens >= max_generation_input_tokens:
        raise ConfigurationError(
            "RAG_MAX_CONTEXT_TOKENS must be smaller than "
            "RAG_MAX_GENERATION_INPUT_TOKENS."
        )

    reasoning_effort = os.getenv(
        "OPENAI_REASONING_EFFORT", DEFAULT_OPENAI_REASONING_EFFORT
    ).strip()
    if reasoning_effort not in {"minimal", "low", "medium", "high"}:
        raise ConfigurationError("OPENAI_REASONING_EFFORT is unsupported.")
    max_concurrent_requests = _positive_int_environment(
        "OPENAI_MAX_CONCURRENT_REQUESTS",
        DEFAULT_OPENAI_MAX_CONCURRENT_REQUESTS,
    )

    raw_visibilities = os.getenv("RAG_EXTERNAL_ALLOWED_VISIBILITIES", "public")
    visibilities = tuple(
        value.strip() for value in raw_visibilities.split(",") if value.strip()
    )
    unsupported = set(visibilities) - {"public", "internal"}
    if not visibilities or unsupported or len(visibilities) != len(set(visibilities)):
        raise ConfigurationError(
            "RAG_EXTERNAL_ALLOWED_VISIBILITIES must contain unique public/internal values."
        )
    return RagGenerationSettings(
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        timeout_seconds=timeout_seconds,
        max_output_tokens=max_output_tokens,
        max_context_tokens=max_context_tokens,
        max_generation_input_tokens=max_generation_input_tokens,
        reasoning_effort=reasoning_effort,
        max_concurrent_requests=max_concurrent_requests,
        externally_allowed_visibilities=visibilities,
    )


def load_settings() -> Settings:
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_api_key:
        raise ConfigurationError("OPENAI_API_KEY is required.")

    raw_timeout = os.getenv(
        "MODEL_TIMEOUT_SECONDS", str(DEFAULT_MODEL_TIMEOUT_SECONDS)
    )
    try:
        model_timeout_seconds = float(raw_timeout)
    except ValueError as error:
        raise ConfigurationError(
            "MODEL_TIMEOUT_SECONDS must be a positive number."
        ) from error

    if not isfinite(model_timeout_seconds) or model_timeout_seconds <= 0:
        raise ConfigurationError(
            "MODEL_TIMEOUT_SECONDS must be a positive number."
        )

    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise ConfigurationError("DATABASE_URL is required.")
    if not database_url.startswith("postgresql+asyncpg://"):
        raise ConfigurationError(
            "DATABASE_URL must use the postgresql+asyncpg:// scheme."
        )

    return Settings(
        openai_api_key=openai_api_key,
        model_timeout_seconds=model_timeout_seconds,
        database_url=database_url,
    )
