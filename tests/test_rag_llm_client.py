"""Verify OpenAI Responses API requests, parsing, errors, usage, and cost."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import openai
from pydantic import ValidationError

from app.config import RagGenerationSettings
from app.rag.generation.models import GenerationErrorCode, GenerationPipelineError, TokenUsage
from app.rag.grounding.models import GroundedDraft, GroundingStatus
from app.rag.generation.openai_client import OpenAIResponsesClient, estimate_cost


def settings() -> RagGenerationSettings:
    """Build a complete low-cost generation configuration fixture."""
    return RagGenerationSettings(
        openai_api_key="test-key",
        openai_model="gpt-5-nano",
        timeout_seconds=20.0,
        max_output_tokens=256,
        max_context_tokens=1_000,
        max_generation_input_tokens=2_000,
        reasoning_effort="minimal",
        max_concurrent_requests=2,
        externally_allowed_visibilities=("public",),
    )


def usage(**overrides):
    """Build an SDK-shaped usage fixture."""
    values = {
        "input_tokens": 1_000,
        "output_tokens": 100,
        "total_tokens": 1_100,
        "cached_tokens": 200,
        "reasoning_tokens": 20,
        **overrides,
    }
    return SimpleNamespace(
        input_tokens=values["input_tokens"],
        output_tokens=values["output_tokens"],
        total_tokens=values["total_tokens"],
        input_tokens_details=SimpleNamespace(cached_tokens=values["cached_tokens"]),
        output_tokens_details=SimpleNamespace(reasoning_tokens=values["reasoning_tokens"]),
    )


class OpenAIResponsesClientTests(unittest.IsolatedAsyncioTestCase):
    """Exercise the provider boundary without live network requests."""

    def setUp(self) -> None:
        """Create one mock SDK client for each test."""
        self.create = AsyncMock()
        self.sdk = SimpleNamespace(responses=SimpleNamespace(create=self.create))
        self.client = OpenAIResponsesClient(settings(), self.sdk)

    async def test_complete_response_uses_stateless_bounded_request(self) -> None:
        """Send supported parameters and return actual identifiers and usage."""
        self.create.return_value = SimpleNamespace(
            status="completed",
            output_text=" Verified duplicate charges may qualify. ",
            usage=usage(),
            model="gpt-5-nano-2025-08-07",
            id="resp_123",
        )
        with patch(
            "app.rag.generation.openai_client.perf_counter", side_effect=(10.0, 10.2, 10.2)
        ):
            result = await self.client.generate("system", "user", request_id="req-1")

        self.create.assert_awaited_once_with(
            model="gpt-5-nano",
            instructions="system",
            input="user",
            max_output_tokens=256,
            reasoning={"effort": "minimal"},
            store=False,
        )
        self.assertEqual(result.answer, "Verified duplicate charges may qualify.")
        self.assertEqual(result.model, "gpt-5-nano-2025-08-07")
        self.assertEqual(result.provider_response_id, "resp_123")
        self.assertEqual(result.usage.cached_input_tokens, 200)
        self.assertAlmostEqual(result.latency_ms, 200.0)
        self.assertAlmostEqual(result.cost.estimated_cost_usd, 0.000081)

    def test_sdk_client_is_reused_with_one_bounded_retry(self) -> None:
        """Configure timeout and SDK retry once rather than per request."""
        with patch("app.rag.generation.openai_client.AsyncOpenAI") as constructor:
            client = OpenAIResponsesClient(settings())
        constructor.assert_called_once_with(
            api_key="test-key", timeout=20.0, max_retries=1
        )
        self.assertIs(client._client, constructor.return_value)

    async def test_missing_usage_is_explicitly_unavailable(self) -> None:
        """Return no cost instead of treating absent usage as zero tokens."""
        self.create.return_value = SimpleNamespace(
            status="completed", output_text="Answer", usage=None,
            model="gpt-5-nano", id="resp_1",
        )
        result = await self.client.generate("system", "user")
        self.assertIsNone(result.usage)
        self.assertIsNone(result.cost)

    async def test_structured_response_uses_provider_native_parse(self) -> None:
        """Reuse the provider client while requiring a validated Pydantic output."""
        parsed = GroundedDraft(status=GroundingStatus.INSUFFICIENT_EVIDENCE)
        self.sdk.responses.parse = AsyncMock(return_value=SimpleNamespace(
            status="completed", output_parsed=parsed, usage=None,
            model="gpt-5-nano-test", id="resp_structured",
        ))
        result = await self.client.generate_structured(
            "system", "user", GroundedDraft, max_output_tokens=128
        )
        self.assertIs(result.output, parsed)
        self.sdk.responses.parse.assert_awaited_once_with(
            model="gpt-5-nano", instructions="system", input="user",
            text_format=GroundedDraft, max_output_tokens=128,
            reasoning={"effort": "minimal"}, store=False,
        )

    async def test_invalid_or_truncated_structured_output_is_controlled(self) -> None:
        """Map SDK-side Pydantic validation failures to the safe error contract."""
        try:
            GroundedDraft.model_validate_json('{"status":"grounded","claims":[]}')
        except ValidationError as error:
            validation_error = error
        self.sdk.responses.parse = AsyncMock(side_effect=validation_error)
        with self.assertRaises(GenerationPipelineError) as caught:
            await self.client.generate_structured("system", "user", GroundedDraft)
        self.assertEqual(caught.exception.code, GenerationErrorCode.GENERATION_ERROR)

    async def test_incomplete_empty_and_malformed_usage_are_rejected(self) -> None:
        """Never report incomplete or malformed provider output as generated."""
        responses = (
            SimpleNamespace(
                status="incomplete", incomplete_details=SimpleNamespace(reason="max_output_tokens")
            ),
            SimpleNamespace(status="completed", output_text=" ", usage=None),
            SimpleNamespace(status="completed", output_text="answer", usage=usage(total_tokens=9)),
        )
        for response in responses:
            with self.subTest(response=response):
                self.create.return_value = response
                with self.assertRaises(GenerationPipelineError) as caught:
                    await self.client.generate("system", "user")
                self.assertEqual(caught.exception.code, GenerationErrorCode.GENERATION_ERROR)

    async def test_timeout_rate_limit_and_authentication_have_safe_categories(self) -> None:
        """Differentiate permanent configuration from transient provider failures."""
        failures = (
            (openai.APITimeoutError(request=Mock()), GenerationErrorCode.LLM_PROVIDER_ERROR),
            (
                openai.RateLimitError("rate", response=Mock(status_code=429), body=None),
                GenerationErrorCode.LLM_PROVIDER_ERROR,
            ),
            (
                openai.AuthenticationError("auth", response=Mock(status_code=401), body=None),
                GenerationErrorCode.LLM_CONFIGURATION_ERROR,
            ),
        )
        for failure, expected_code in failures:
            with self.subTest(failure=type(failure).__name__):
                self.create.side_effect = failure
                with self.assertRaises(GenerationPipelineError) as caught:
                    await self.client.generate("system", "user")
                self.assertEqual(caught.exception.code, expected_code)
                self.create.side_effect = None

    def test_cost_rejects_impossible_cached_usage(self) -> None:
        """Avoid negative uncached-token cost from malformed provider data."""
        invalid = TokenUsage(
            input_tokens=2,
            cached_input_tokens=3,
            output_tokens=1,
            reasoning_tokens=0,
            total_tokens=3,
        )
        with self.assertRaises(GenerationPipelineError):
            estimate_cost(invalid, "gpt-5-nano")

    def test_cost_is_only_estimated_for_priced_model(self) -> None:
        """Never apply GPT-5 Nano rates to a differently configured model."""
        valid = TokenUsage(
            input_tokens=2_000,
            cached_input_tokens=0,
            output_tokens=200,
            reasoning_tokens=0,
            total_tokens=2_200,
        )
        self.assertAlmostEqual(
            estimate_cost(valid, "gpt-5-nano-2025-08-07").estimated_cost_usd, 0.00018
        )
        self.assertIsNone(estimate_cost(valid, "gpt-5-2025-08-07"))


if __name__ == "__main__":
    unittest.main()
