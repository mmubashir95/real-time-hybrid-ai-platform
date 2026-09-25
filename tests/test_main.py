import unittest
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

from app.main import analyze
from app.schemas import AnalysisRequest
from app.services.ai_service import AIServiceError


class AnalyzeRouteTests(unittest.TestCase):
    def test_invalid_requests_are_rejected_before_service_call(self) -> None:
        invalid_requests = ({}, {"message": None}, {"message": ["hello"]})

        with patch("app.main.analyze_message") as analyze_message:
            for request_data in invalid_requests:
                with self.subTest(request_data=request_data):
                    with self.assertRaises(ValidationError):
                        AnalysisRequest.model_validate(request_data)

        analyze_message.assert_not_called()

    def test_service_failure_returns_safe_503(self) -> None:
        with patch(
            "app.main.analyze_message",
            side_effect=AIServiceError("sensitive provider details"),
        ):
            with self.assertRaises(HTTPException) as raised:
                analyze(AnalysisRequest(message="Hello"))

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(
            raised.exception.detail,
            "AI analysis is temporarily unavailable.",
        )
        self.assertNotIn("sensitive", raised.exception.detail)


if __name__ == "__main__":
    unittest.main()
