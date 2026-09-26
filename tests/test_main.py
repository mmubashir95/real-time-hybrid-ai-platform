import ast
import asyncio
import inspect
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient

import app.main as main_module
import app.services.ai_service as ai_service_module
from app.main import app, fake_async_model_call
from app.services.ai_service import AIServiceError


class AnalyzeRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_invalid_requests_are_rejected_before_service_call(self) -> None:
        invalid_requests = ({}, {"message": None}, {"message": ["hello"]})

        with patch(
            "app.main.analyze_message", new_callable=AsyncMock
        ) as analyze_message:
            for request_data in invalid_requests:
                with self.subTest(request_data=request_data):
                    response = self.client.post("/analyze", json=request_data)
                    self.assertEqual(response.status_code, 422)

        analyze_message.assert_not_called()

    def test_service_failure_returns_safe_503(self) -> None:
        with patch(
            "app.main.analyze_message",
            new_callable=AsyncMock,
            side_effect=AIServiceError("sensitive provider details"),
        ):
            response = self.client.post("/analyze", json={"message": "Hello"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(),
            {"detail": "AI analysis is temporarily unavailable."},
        )
        self.assertNotIn("sensitive", response.text)

    def test_analyze_awaits_service_and_preserves_response(self) -> None:
        expected = {
            "category": "billing",
            "urgency": "high",
            "summary": "Duplicate charge.",
            "entities": {"amount": "$10"},
            "sentiment": "negative",
        }
        with patch(
            "app.main.analyze_message",
            new_callable=AsyncMock,
            return_value=expected,
        ) as analyze_message:
            response = self.client.post(
                "/analyze", json={"message": "I was charged twice."}
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected)
        analyze_message.assert_awaited_once_with("I was charged twice.")


class AsyncDemoTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        transport = httpx.ASGITransport(app=app)
        self.client = httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def test_async_endpoint_returns_expected_response(self) -> None:
        with patch(
            "app.main.fake_async_model_call",
            new_callable=AsyncMock,
            return_value="async result",
        ) as model_call:
            response = await self.client.get("/async-demo")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"result": "async result"})
        model_call.assert_awaited_once_with()

    async def test_two_async_requests_overlap_while_waiting(self) -> None:
        both_requests_started = asyncio.Event()
        release_requests = asyncio.Event()
        started_count = 0

        async def controlled_model_call() -> str:
            nonlocal started_count
            started_count += 1
            if started_count == 2:
                both_requests_started.set()
            await release_requests.wait()
            return "async result"

        with patch("app.main.fake_async_model_call", side_effect=controlled_model_call):
            requests = [
                asyncio.create_task(self.client.get("/async-demo"))
                for _ in range(2)
            ]
            await asyncio.wait_for(both_requests_started.wait(), timeout=1.0)
            release_requests.set()
            responses = await asyncio.gather(*requests)

        self.assertEqual(started_count, 2)
        self.assertTrue(all(response.status_code == 200 for response in responses))

    async def test_demo_operation_uses_an_awaited_async_sleep(self) -> None:
        with patch("app.main.asyncio.sleep", new_callable=AsyncMock) as sleep:
            result = await fake_async_model_call()

        self.assertEqual(result, "async result")
        sleep.assert_awaited_once()

    async def test_demo_operation_can_be_cancelled_at_await(self) -> None:
        operation = asyncio.create_task(fake_async_model_call())
        await asyncio.sleep(0)
        operation.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await operation

    def test_real_async_paths_do_not_call_time_sleep(self) -> None:
        """Guard against the `time.sleep(...)` spelling only.

        This does not prove the paths are non-blocking; the overlap test and
        the AsyncMock-based service tests carry that.
        """
        source_files = (
            Path(inspect.getsourcefile(main_module)),
            Path(inspect.getsourcefile(ai_service_module)),
        )
        for source_file in source_files:
            tree = ast.parse(source_file.read_text())
            blocking_calls = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "time"
                and node.func.attr == "sleep"
            ]
            self.assertEqual(blocking_calls, [], source_file)


if __name__ == "__main__":
    unittest.main()
