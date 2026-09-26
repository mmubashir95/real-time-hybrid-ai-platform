import asyncio
import json
import logging
from functools import lru_cache
from time import perf_counter

import openai
from openai import AsyncOpenAI
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
def _get_client() -> tuple[AsyncOpenAI, Settings]:
    settings = load_settings()
    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        timeout=settings.model_timeout_seconds,
        # Disable SDK retries so a failed request returns within one timeout
        # instead of up to three timeouts plus backoff.
        max_retries=0,
    )
    return client, settings


async def analyze_message(
    message: str,
    *,
    request_id: str = "-",
) -> AnalysisResult:
    try:
        client, settings = _get_client()
    except ConfigurationError as error:
        logger.error("AI service configuration is invalid: %s", error)
        raise AIServiceError("AI service is not configured.") from error

    model_started_at = perf_counter()
    model_status = "error"
    try:
        # Awaiting the async client releases the event loop while the HTTP
        # request is in flight, so other requests can progress meanwhile.
        response = await client.responses.create(
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
        model_status = "success"
    except asyncio.CancelledError:
        model_status = "cancelled"
        raise
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
    finally:
        logger.info(
            "performance request_id=%s operation=model_call model=%s "
            "model_latency_ms=%.3f status=%s",
            request_id,
            MODEL,
            (perf_counter() - model_started_at) * 1000,
            model_status,
        )

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
