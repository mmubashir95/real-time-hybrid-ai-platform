import os
from dataclasses import dataclass
from math import isfinite

DEFAULT_MODEL_TIMEOUT_SECONDS = 30.0


class ConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    model_timeout_seconds: float


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

    return Settings(
        openai_api_key=openai_api_key,
        model_timeout_seconds=model_timeout_seconds,
    )
