import os
import unittest
from unittest.mock import patch

from app.config import ConfigurationError, load_settings


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
                    },
                    clear=True,
                ):
                    with self.assertRaisesRegex(
                        ConfigurationError, "MODEL_TIMEOUT_SECONDS"
                    ):
                        load_settings()


if __name__ == "__main__":
    unittest.main()
