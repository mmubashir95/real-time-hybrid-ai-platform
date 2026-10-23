"""Call the OpenAI Responses API behind a small Phase 9 client boundary."""

from __future__ import annotations

import asyncio
import logging
from time import perf_counter
from typing import Protocol, TypeVar

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from app.config import RagGenerationSettings
from app.rag.generation.models import (
    CostEstimate,
    GenerationErrorCode,
    GenerationPipelineError,
    GPT5_NANO_CACHED_INPUT_USD_PER_MILLION,
    GPT5_NANO_INPUT_USD_PER_MILLION,
    GPT5_NANO_OUTPUT_USD_PER_MILLION,
    LlmGenerationResult,
    LlmStructuredResult,
    TokenUsage,
)


logger = logging.getLogger(__name__)


class ResponsesClient(Protocol):
    """Describe the SDK surface needed by the generation client."""

    responses: object


PRICED_MODEL_PREFIX = "gpt-5-nano"
StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)


def estimate_cost(usage: TokenUsage | None, model: str) -> CostEstimate | None:
    """Calculate GPT-5 Nano text cost using separately priced cached input.

    Returns None for any other model because only GPT-5 Nano rates are recorded;
    applying them to another model would report a wrong estimate as valid.
    """
    if usage is None or not model.startswith(PRICED_MODEL_PREFIX):
        return None
    uncached_input = usage.input_tokens - usage.cached_input_tokens
    if uncached_input < 0:
        raise GenerationPipelineError(
            GenerationErrorCode.GENERATION_ERROR,
            "Provider usage reported more cached tokens than input tokens.",
        )
    cost = (
        uncached_input * GPT5_NANO_INPUT_USD_PER_MILLION
        + usage.cached_input_tokens * GPT5_NANO_CACHED_INPUT_USD_PER_MILLION
        + usage.output_tokens * GPT5_NANO_OUTPUT_USD_PER_MILLION
    ) / 1_000_000
    return CostEstimate(estimated_cost_usd=round(cost, 10))


def _usage_from_response(provider_usage: object | None) -> TokenUsage | None:
    """Validate the optional SDK usage object without inventing missing values."""
    if provider_usage is None:
        return None
    try:
        input_tokens = provider_usage.input_tokens
        output_tokens = provider_usage.output_tokens
        total_tokens = provider_usage.total_tokens
        cached_tokens = provider_usage.input_tokens_details.cached_tokens
        reasoning_tokens = provider_usage.output_tokens_details.reasoning_tokens
        values = (
            input_tokens,
            output_tokens,
            total_tokens,
            cached_tokens,
            reasoning_tokens,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("usage values must be integers")
        usage = TokenUsage(
            input_tokens=input_tokens,
            cached_input_tokens=cached_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise GenerationPipelineError(
            GenerationErrorCode.GENERATION_ERROR,
            "OpenAI returned invalid token usage.",
        ) from error
    if usage.total_tokens != usage.input_tokens + usage.output_tokens:
        raise GenerationPipelineError(
            GenerationErrorCode.GENERATION_ERROR,
            "OpenAI returned inconsistent token usage.",
        )
    return usage


class OpenAIResponsesClient:
    """Reuse one asynchronous OpenAI client for stateless answer generation."""

    def __init__(
        self,
        settings: RagGenerationSettings,
        client: ResponsesClient | None = None,
    ) -> None:
        """Bind validated settings and disable duplicated retry layers."""
        self.settings = settings
        self._request_slots = asyncio.Semaphore(settings.max_concurrent_requests)
        self._client = client or AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.timeout_seconds,
            # One SDK retry covers transient failures; this class adds no retry.
            max_retries=1,
        )

    async def generate(
        self,
        system_instruction: str,
        user_message: str,
        *,
        request_id: str = "-",
    ) -> LlmGenerationResult:
        """Generate one complete answer and validate its status, text, and usage."""
        if not system_instruction.strip() or not user_message.strip():
            raise ValueError("Generation instructions and input must not be empty")
        started = perf_counter()
        status = "error"
        try:
            # Bound simultaneous paid requests per reusable client instance.
            async with self._request_slots:
                response = await self._client.responses.create(
                    model=self.settings.openai_model,
                    instructions=system_instruction,
                    input=user_message,
                    max_output_tokens=self.settings.max_output_tokens,
                    reasoning={"effort": self.settings.reasoning_effort},
                    store=False,
                )
            if response.status != "completed":
                reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
                logger.warning(
                    "openai_generation_incomplete request_id=%s reason=%s",
                    request_id,
                    reason or "unknown",
                )
                raise GenerationPipelineError(
                    GenerationErrorCode.GENERATION_ERROR,
                    "OpenAI generation did not complete.",
                )
            answer = response.output_text.strip()
            if not answer:
                raise GenerationPipelineError(
                    GenerationErrorCode.GENERATION_ERROR,
                    "OpenAI returned an empty answer.",
                )
            usage = _usage_from_response(response.usage)
            status = "success"
            latency_ms = (perf_counter() - started) * 1000
            return LlmGenerationResult(
                answer=answer,
                model=response.model,
                provider_response_id=getattr(response, "id", None),
                usage=usage,
                cost=estimate_cost(usage, response.model),
                latency_ms=latency_ms,
            )
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except GenerationPipelineError:
            raise
        except openai.APITimeoutError as error:
            raise GenerationPipelineError(
                GenerationErrorCode.LLM_PROVIDER_ERROR,
                "OpenAI generation timed out.",
            ) from error
        except (openai.AuthenticationError, openai.PermissionDeniedError) as error:
            raise GenerationPipelineError(
                GenerationErrorCode.LLM_CONFIGURATION_ERROR,
                "OpenAI credentials or model access are invalid.",
            ) from error
        except (openai.RateLimitError, openai.APIConnectionError) as error:
            raise GenerationPipelineError(
                GenerationErrorCode.LLM_PROVIDER_ERROR,
                "OpenAI generation is temporarily unavailable.",
            ) from error
        except openai.BadRequestError as error:
            raise GenerationPipelineError(
                GenerationErrorCode.LLM_CONFIGURATION_ERROR,
                "OpenAI rejected the generation configuration.",
            ) from error
        except openai.APIError as error:
            raise GenerationPipelineError(
                GenerationErrorCode.LLM_PROVIDER_ERROR,
                "OpenAI generation failed.",
            ) from error
        finally:
            logger.info(
                "performance request_id=%s operation=rag_generation model=%s "
                "latency_ms=%.3f status=%s",
                request_id,
                self.settings.openai_model,
                (perf_counter() - started) * 1000,
                status,
            )

    async def generate_structured(
        self,
        system_instruction: str,
        user_message: str,
        output_model: type[StructuredOutput],
        *,
        request_id: str = "-",
        max_output_tokens: int | None = None,
    ) -> LlmStructuredResult[StructuredOutput]:
        """Generate and validate provider-native structured output.

        This deliberately shares Phase 9's client, limits, retries, model,
        usage validation, and cost accounting instead of adding a second client.
        """
        if not system_instruction.strip() or not user_message.strip():
            raise ValueError("Generation instructions and input must not be empty")
        started = perf_counter()
        status = "error"
        try:
            async with self._request_slots:
                response = await self._client.responses.parse(
                    model=self.settings.openai_model,
                    instructions=system_instruction,
                    input=user_message,
                    text_format=output_model,
                    max_output_tokens=max_output_tokens or self.settings.max_output_tokens,
                    reasoning={"effort": self.settings.reasoning_effort},
                    store=False,
                )
            if response.status != "completed":
                raise GenerationPipelineError(
                    GenerationErrorCode.GENERATION_ERROR,
                    "OpenAI structured generation did not complete.",
                )
            parsed = getattr(response, "output_parsed", None)
            if not isinstance(parsed, output_model):
                raise GenerationPipelineError(
                    GenerationErrorCode.GENERATION_ERROR,
                    "OpenAI returned invalid structured output.",
                )
            usage = _usage_from_response(response.usage)
            status = "success"
            return LlmStructuredResult(
                output=parsed,
                model=response.model,
                provider_response_id=getattr(response, "id", None),
                usage=usage,
                cost=estimate_cost(usage, response.model),
                latency_ms=(perf_counter() - started) * 1000,
            )
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except GenerationPipelineError:
            raise
        except ValidationError as error:
            # The SDK validates JSON inside parse(), before status is visible, so
            # truncated or locally invalid output surfaces here, not as APIError.
            raise GenerationPipelineError(
                GenerationErrorCode.GENERATION_ERROR,
                "OpenAI returned invalid or incomplete structured output.",
            ) from error
        except openai.APITimeoutError as error:
            raise GenerationPipelineError(
                GenerationErrorCode.LLM_PROVIDER_ERROR,
                "OpenAI structured generation timed out.",
            ) from error
        except (openai.AuthenticationError, openai.PermissionDeniedError) as error:
            raise GenerationPipelineError(
                GenerationErrorCode.LLM_CONFIGURATION_ERROR,
                "OpenAI credentials or model access are invalid.",
            ) from error
        except (openai.RateLimitError, openai.APIConnectionError) as error:
            raise GenerationPipelineError(
                GenerationErrorCode.LLM_PROVIDER_ERROR,
                "OpenAI structured generation is temporarily unavailable.",
            ) from error
        except openai.BadRequestError as error:
            raise GenerationPipelineError(
                GenerationErrorCode.LLM_CONFIGURATION_ERROR,
                "OpenAI rejected the structured generation configuration.",
            ) from error
        except openai.APIError as error:
            raise GenerationPipelineError(
                GenerationErrorCode.LLM_PROVIDER_ERROR,
                "OpenAI structured generation failed.",
            ) from error
        finally:
            logger.info(
                "performance request_id=%s operation=rag_structured_generation "
                "model=%s latency_ms=%.3f status=%s",
                request_id,
                self.settings.openai_model,
                (perf_counter() - started) * 1000,
                status,
            )
