import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.ai_service import AIServiceError


class AnalyzeRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_invalid_requests_are_rejected_before_service_call(self) -> None:
        invalid_requests = ({}, {"message": None}, {"message": ["hello"]})

        with patch("app.main.analyze_message") as analyze_message:
            for request_data in invalid_requests:
                with self.subTest(request_data=request_data):
                    response = self.client.post("/analyze", json=request_data)
                    self.assertEqual(response.status_code, 422)

        analyze_message.assert_not_called()

    def test_service_failure_returns_safe_503(self) -> None:
        with patch(
            "app.main.analyze_message",
            side_effect=AIServiceError("sensitive provider details"),
        ):
            response = self.client.post("/analyze", json={"message": "Hello"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(),
            {"detail": "AI analysis is temporarily unavailable."},
        )
        self.assertNotIn("sensitive", response.text)


if __name__ == "__main__":
    unittest.main()
