import os
import unittest
from unittest.mock import patch

from app.config import (
    ConfigurationError,
    RagGenerationSettings,
    Settings,
    load_rag_generation_settings,
    load_settings,
)


class SettingsTests(unittest.TestCase):
    def test_missing_api_key_is_rejected(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ConfigurationError, "OPENAI_API_KEY"):
                load_settings()

    def test_timeout_must_be_positive_number(self) -> None:
        for timeout in ("invalid", "0", "-1", "nan", "inf"):
            with self.subTest(timeout=timeout):
                with patch.dict(
                    os.environ,
                    {
                        "OPENAI_API_KEY": "test-key",
                        "MODEL_TIMEOUT_SECONDS": timeout,
                        "DATABASE_URL": (
                            "postgresql+asyncpg://user:password@localhost/test"
                        ),
                    },
                    clear=True,
                ):
                    with self.assertRaisesRegex(
                        ConfigurationError, "MODEL_TIMEOUT_SECONDS"
                    ):
                        load_settings()

    def test_missing_database_url_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-key"},
            clear=True,
        ):
            with self.assertRaisesRegex(ConfigurationError, "DATABASE_URL"):
                load_settings()

    def test_database_url_requires_asyncpg_scheme(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                "DATABASE_URL": "postgresql://user:password@localhost/test",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ConfigurationError, r"postgresql\+asyncpg"):
                load_settings()

    def test_valid_settings_are_loaded(self) -> None:
        environment = {
            "OPENAI_API_KEY": "test-key",
            "MODEL_TIMEOUT_SECONDS": "12",
            "DATABASE_URL": "postgresql+asyncpg://user:password@localhost/test",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = load_settings()

        self.assertEqual(
            settings,
            Settings(
                openai_api_key="test-key",
                model_timeout_seconds=12.0,
                database_url=environment["DATABASE_URL"],
            ),
        )


class RagGenerationSettingsTests(unittest.TestCase):
    """Validate Phase 9 cost and token-control environment settings."""

    def test_defaults_are_low_cost_and_public_only(self) -> None:
        """Load the selected model and conservative external-data policy."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            settings = load_rag_generation_settings()
        self.assertEqual(settings.openai_model, "gpt-5-nano")
        self.assertEqual(settings.reasoning_effort, "minimal")
        self.assertEqual(settings.externally_allowed_visibilities, ("public",))

    def test_complete_generation_settings_are_loaded(self) -> None:
        """Accept explicit token, timeout, model, and visibility configuration."""
        environment = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_MODEL": "gpt-5-nano-2025-08-07",
            "OPENAI_TIMEOUT_SECONDS": "12.5",
            "OPENAI_MAX_OUTPUT_TOKENS": "300",
            "RAG_MAX_CONTEXT_TOKENS": "1000",
            "RAG_MAX_GENERATION_INPUT_TOKENS": "2000",
            "OPENAI_REASONING_EFFORT": "low",
            "OPENAI_MAX_CONCURRENT_REQUESTS": "3",
            "RAG_EXTERNAL_ALLOWED_VISIBILITIES": "public,internal",
        }
        with patch.dict(os.environ, environment, clear=True):
            loaded = load_rag_generation_settings()
        self.assertEqual(
            loaded,
            RagGenerationSettings(
                "test-key", "gpt-5-nano-2025-08-07", 12.5, 300,
                1000, 2000, "low", 3, ("public", "internal"),
            ),
        )

    def test_invalid_generation_limits_and_visibility_are_rejected(self) -> None:
        """Reject unsafe or contradictory environment configuration."""
        invalid = (
            {"OPENAI_MAX_OUTPUT_TOKENS": "15"},
            {"RAG_MAX_CONTEXT_TOKENS": "2000", "RAG_MAX_GENERATION_INPUT_TOKENS": "2000"},
            {"OPENAI_REASONING_EFFORT": "maximum"},
            {"OPENAI_MAX_CONCURRENT_REQUESTS": "0"},
            {"RAG_EXTERNAL_ALLOWED_VISIBILITIES": "public,secret"},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                with patch.dict(
                    os.environ,
                    {"OPENAI_API_KEY": "test-key", **overrides},
                    clear=True,
                ):
                    with self.assertRaises(ConfigurationError):
                        load_rag_generation_settings()


if __name__ == "__main__":
    unittest.main()
