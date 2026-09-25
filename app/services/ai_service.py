import json
import logging
from functools import lru_cache

import openai
from openai import OpenAI
from pydantic import ValidationError

from app.config import ConfigurationError, Settings, load_settings
from app.schemas import AnalysisResult

logger = logging.getLogger(__name__)

MODEL = "gpt-4o-mini"
ANALYSIS_INSTRUCTIONS = """Analyze the customer support message.

Return the category, urgency, summary, entities, and sentiment.

Urgency must be low, medium, or high. Use high only when the customer was
charged incorrectly, is locked out of their account, or cannot use the service
at all.

Entities are specific values stated in the message, such as order IDs, email
addresses, product names, monetary amounts, and dates. Use short snake_case
entity types such as order_id, email, product, amount, and date. Return an
empty list when the message contains no such values.
"""
ANALYSIS_SCHEMA = AnalysisResult.model_json_schema()
ANALYSIS_SCHEMA["additionalProperties"] = False
# OpenAI Structured Outputs requires every object to declare its keys, so a
# free-form dict cannot be requested directly. The model returns entities as
# type/value pairs, which analyze_message converts to the public dict contract.
ANALYSIS_SCHEMA["properties"]["entities"] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "type": {"type": "string"},
            "value": {"type": "string"},
        },
        "required": ["type", "value"],
        "additionalProperties": False,
    },
}


class AIServiceError(RuntimeError):
    pass


@lru_cache
def _get_client() -> tuple[OpenAI, Settings]:
    settings = load_settings()
    client = OpenAI(
        api_key=settings.openai_api_key,
        timeout=settings.model_timeout_seconds,
        # Disable SDK retries so a failed request returns within one timeout
        # instead of up to three timeouts plus backoff.
        max_retries=0,
    )
    return client, settings


def analyze_message(message: str) -> AnalysisResult:
    try:
        client, settings = _get_client()
    except ConfigurationError as error:
        logger.error("AI service configuration is invalid: %s", error)
        raise AIServiceError("AI service is not configured.") from error

    try:
        response = client.responses.create(
            model=MODEL,
            instructions=ANALYSIS_INSTRUCTIONS,
            input=message,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "analysis_result",
                    "strict": True,
                    "schema": ANALYSIS_SCHEMA,
                }
            },
        )
    except openai.APITimeoutError as error:
        logger.warning(
            "AI provider request timed out provider=openai timeout_seconds=%s",
            settings.model_timeout_seconds,
        )
        raise AIServiceError("AI provider request timed out.") from error
    except openai.APIError as error:
        logger.error(
            "AI provider request failed provider=openai error_type=%s",
            type(error).__name__,
        )
        raise AIServiceError("AI provider request failed.") from error

    try:
        analysis = json.loads(response.output_text)
        analysis["entities"] = {
            entity["type"]: entity["value"] for entity in analysis["entities"]
        }
        return AnalysisResult.model_validate(analysis)
    except (
        AttributeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValidationError,
    ) as error:
        logger.error(
            "AI provider returned an invalid response provider=openai error_type=%s",
            type(error).__name__,
        )
        raise AIServiceError("AI provider returned an invalid response.") from error
