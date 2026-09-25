import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import openai

from app.config import ConfigurationError, Settings
from app.schemas import AnalysisResult
from app.services.ai_service import AIServiceError, _get_client, analyze_message


class AIServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = Mock()
        self.settings = Settings(
            openai_api_key="test-key",
            model_timeout_seconds=30.0,
        )
        self.client_patch = patch(
            "app.services.ai_service._get_client",
            return_value=(self.client, self.settings),
        )
        self.client_patch.start()
        self.addCleanup(self.client_patch.stop)

    def test_client_uses_configured_timeout_without_retries(self) -> None:
        _get_client.cache_clear()
        self.addCleanup(_get_client.cache_clear)

        with (
            patch(
                "app.services.ai_service.load_settings",
                return_value=self.settings,
            ),
            patch("app.services.ai_service.OpenAI") as openai_client,
        ):
            client, settings = _get_client()

        openai_client.assert_called_once_with(
            api_key="test-key",
            timeout=30.0,
            max_retries=0,
        )
        self.assertIs(client, openai_client.return_value)
        self.assertEqual(settings, self.settings)

    def test_valid_provider_response_returns_analysis_result(self) -> None:
        self.client.responses.create.return_value = SimpleNamespace(
            output_text=json.dumps(
                {
                    "category": "billing",
                    "urgency": "high",
                    "summary": "Customer reports duplicate billing.",
                    "entities": [{"type": "amount", "value": "$10"}],
                    "sentiment": "negative",
                }
            )
        )

        result = analyze_message("I was charged $10 twice.")

        self.assertIsInstance(result, AnalysisResult)
        self.assertEqual(result.entities, {"amount": "$10"})

    def test_missing_configuration_is_controlled(self) -> None:
        with patch(
            "app.services.ai_service._get_client",
            side_effect=ConfigurationError("OPENAI_API_KEY is required."),
        ):
            with self.assertRaisesRegex(AIServiceError, "not configured"):
                analyze_message("Hello")

    def test_timeout_is_controlled(self) -> None:
        self.client.responses.create.side_effect = openai.APITimeoutError(
            request=Mock()
        )

        with self.assertRaisesRegex(AIServiceError, "timed out"):
            analyze_message("Hello")

    def test_provider_failure_is_controlled(self) -> None:
        self.client.responses.create.side_effect = openai.APIError(
            "Provider failure", Mock(), body=None
        )

        with self.assertRaisesRegex(AIServiceError, "request failed"):
            analyze_message("Hello")

    def test_malformed_response_is_controlled(self) -> None:
        self.client.responses.create.return_value = SimpleNamespace(
            output_text='{"category": "billing", "entities": []}'
        )

        with self.assertRaisesRegex(AIServiceError, "invalid response"):
            analyze_message("Hello")


if __name__ == "__main__":
    unittest.main()
