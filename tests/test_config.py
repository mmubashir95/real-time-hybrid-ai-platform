import os
import unittest
from unittest.mock import patch

from app.config import ConfigurationError, Settings, load_settings


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


if __name__ == "__main__":
    unittest.main()
