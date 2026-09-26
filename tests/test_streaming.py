import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import DatabaseError, get_session
from app.main import app
from app.schemas import AnalysisResult
from app.services.ai_service import AIServiceError
from app.services.streaming_service import stream_analysis


def parse_sse(body: str) -> list[tuple[str, dict[str, object]]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = block.splitlines()
        event = lines[0].removeprefix("event: ")
        payload = json.loads(lines[1].removeprefix("data: "))
        events.append((event, payload))
    return events


class StreamingRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = Mock(spec=AsyncSession)

        async def override_session():
            yield self.session

        app.dependency_overrides[get_session] = override_session
        self.addCleanup(app.dependency_overrides.clear)
        self.client = TestClient(app)
        self.analysis = AnalysisResult(
            category="billing",
            urgency="high",
            summary="Duplicate charge.",
            entities={"amount": "$10"},
            sentiment="negative",
        )

    def test_successful_stream_has_ordered_json_events(self) -> None:
        with patch(
            "app.services.streaming_service.analyze_message",
            new_callable=AsyncMock,
            return_value=self.analysis,
        ) as analyze_message, patch(
            "app.services.streaming_service.save_analysis",
            new_callable=AsyncMock,
            return_value=SimpleNamespace(id=42),
        ) as save_analysis:
            response = self.client.post(
                "/analyze/stream",
                json={"message": "I was charged twice."},
                headers={"Accept": "text/event-stream"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            response.headers["content-type"].startswith("text/event-stream")
        )
        events = parse_sse(response.text)
        self.assertEqual(
            [event for event, _payload in events],
            [
                "processing_started",
                "analysis_complete",
                "saved",
                "completed",
            ],
        )
        self.assertEqual(events[1][1], self.analysis.model_dump())
        self.assertEqual(events[2][1], {"analysis_id": 42})
        self.assertEqual(
            events[3][1], {"status": "completed", "analysis_id": 42}
        )
        analyze_message.assert_awaited_once()
        self.assertEqual(
            analyze_message.await_args.args, ("I was charged twice.",)
        )
        request_id = analyze_message.await_args.kwargs["request_id"]
        self.assertTrue(request_id)
        save_analysis.assert_awaited_once()
        self.assertIs(save_analysis.await_args.args[0], self.session)
        self.assertEqual(save_analysis.await_args.args[1], "I was charged twice.")
        self.assertEqual(save_analysis.await_args.args[2], self.analysis)
        self.assertGreaterEqual(save_analysis.await_args.args[3], 0)
        self.assertEqual(save_analysis.await_args.kwargs["request_id"], request_id)

    def test_invalid_request_uses_normal_validation(self) -> None:
        with patch(
            "app.services.streaming_service.analyze_message",
            new_callable=AsyncMock,
        ) as analyze_message:
            response = self.client.post("/analyze/stream", json={})

        self.assertEqual(response.status_code, 422)
        analyze_message.assert_not_awaited()

    def test_ai_failures_emit_one_terminal_safe_error(self) -> None:
        failures = (
            "AI provider request timed out.",
            "AI provider request failed.",
            "AI provider returned an invalid response.",
        )
        for failure in failures:
            with self.subTest(failure=failure), patch(
                "app.services.streaming_service.analyze_message",
                new_callable=AsyncMock,
                side_effect=AIServiceError(failure),
            ), patch(
                "app.services.streaming_service.save_analysis",
                new_callable=AsyncMock,
            ) as save_analysis:
                response = self.client.post(
                    "/analyze/stream", json={"message": "Hello"}
                )

            self.assertEqual(response.status_code, 200)
            events = parse_sse(response.text)
            self.assertEqual(
                [event for event, _payload in events],
                ["processing_started", "error"],
            )
            self.assertEqual(events[-1][1]["code"], "AI_UNAVAILABLE")
            self.assertNotIn(failure, response.text)
            save_analysis.assert_not_awaited()

    def test_database_failure_stops_before_saved_and_completed(self) -> None:
        with patch(
            "app.services.streaming_service.analyze_message",
            new_callable=AsyncMock,
            return_value=self.analysis,
        ), patch(
            "app.services.streaming_service.save_analysis",
            new_callable=AsyncMock,
            side_effect=DatabaseError("postgresql://secret@host/database"),
        ):
            response = self.client.post(
                "/analyze/stream", json={"message": "Hello"}
            )

        events = parse_sse(response.text)
        self.assertEqual(
            [event for event, _payload in events],
            ["processing_started", "analysis_complete", "error"],
        )
        self.assertEqual(events[-1][1]["code"], "STORAGE_UNAVAILABLE")
        self.assertNotIn("secret", response.text)

    def test_unexpected_failure_emits_terminal_internal_error(self) -> None:
        with patch(
            "app.services.streaming_service.analyze_message",
            new_callable=AsyncMock,
            return_value=self.analysis,
        ), patch(
            "app.services.streaming_service.save_analysis",
            new_callable=AsyncMock,
            side_effect=RuntimeError("secret internal detail"),
        ):
            response = self.client.post(
                "/analyze/stream", json={"message": "Hello"}
            )

        events = parse_sse(response.text)
        self.assertEqual(
            [event for event, _payload in events],
            ["processing_started", "analysis_complete", "error"],
        )
        self.assertEqual(events[-1][1]["code"], "INTERNAL_ERROR")
        self.assertNotIn("secret", response.text)

    def test_session_stays_open_until_stream_finishes(self) -> None:
        # The stream saves after the endpoint function has returned, so the
        # session dependency must only be cleaned up once the response finishes.
        session_closed = False

        async def tracked_session():
            nonlocal session_closed
            try:
                yield self.session
            finally:
                session_closed = True

        app.dependency_overrides[get_session] = tracked_session
        closed_when_saving = []

        async def save(*_args: object, **_kwargs: object) -> SimpleNamespace:
            closed_when_saving.append(session_closed)
            return SimpleNamespace(id=7)

        with patch(
            "app.services.streaming_service.analyze_message",
            new_callable=AsyncMock,
            return_value=self.analysis,
        ), patch(
            "app.services.streaming_service.save_analysis",
            side_effect=save,
        ):
            response = self.client.post(
                "/analyze/stream", json={"message": "Hello"}
            )

        self.assertEqual(parse_sse(response.text)[-1][0], "completed")
        self.assertEqual(closed_when_saving, [False])
        self.assertTrue(session_closed)


class StreamingCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_disconnect_after_model_call_skips_persistence(self) -> None:
        request = Mock()
        request.is_disconnected = AsyncMock(return_value=True)
        session = Mock(spec=AsyncSession)
        analysis = AnalysisResult(
            category="general",
            urgency="low",
            summary="Greeting.",
            entities={},
            sentiment="neutral",
        )

        with patch(
            "app.services.streaming_service.analyze_message",
            new_callable=AsyncMock,
            return_value=analysis,
        ), patch(
            "app.services.streaming_service.save_analysis",
            new_callable=AsyncMock,
        ) as save_analysis:
            events = [event async for event in stream_analysis(request, session, "Hi")]

        self.assertEqual(
            [event for event, _payload in parse_sse("".join(events))],
            ["processing_started", "analysis_complete"],
        )
        save_analysis.assert_not_awaited()

    async def test_cancellation_propagates_during_model_call(self) -> None:
        request = Mock()
        request.is_disconnected = AsyncMock(return_value=False)
        session = Mock(spec=AsyncSession)
        provider_started = asyncio.Event()

        async def waiting_model_call(
            _message: str, **_kwargs: object
        ) -> AnalysisResult:
            provider_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        with patch(
            "app.services.streaming_service.analyze_message",
            side_effect=waiting_model_call,
        ):
            events = stream_analysis(request, session, "Hello")
            first_event = await anext(events)
            next_event = asyncio.create_task(anext(events))
            await asyncio.wait_for(provider_started.wait(), timeout=1.0)
            next_event.cancel()

            with self.assertRaises(asyncio.CancelledError):
                await next_event

        self.assertEqual(parse_sse(first_event)[0][0], "processing_started")


class StreamingConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.session = Mock(spec=AsyncSession)

        async def override_session():
            yield self.session

        app.dependency_overrides[get_session] = override_session
        self.addCleanup(app.dependency_overrides.clear)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def test_concurrent_streams_do_not_mix_request_state(self) -> None:
        both_models_started = asyncio.Event()
        release_models = asyncio.Event()
        started = 0

        async def analyze(
            message: str, **_kwargs: object
        ) -> AnalysisResult:
            nonlocal started
            started += 1
            if started == 2:
                both_models_started.set()
            await release_models.wait()
            return AnalysisResult(
                category="test",
                urgency="low",
                summary=message,
                entities={},
                sentiment="neutral",
            )

        async def save(
            _session: AsyncSession,
            message: str,
            _analysis: AnalysisResult,
            _latency_ms: int,
            **_kwargs: object,
        ) -> SimpleNamespace:
            return SimpleNamespace(id={"message-a": 1, "message-b": 2}[message])

        with patch(
            "app.services.streaming_service.analyze_message",
            side_effect=analyze,
        ), patch(
            "app.services.streaming_service.save_analysis",
            side_effect=save,
        ):
            requests = [
                asyncio.create_task(
                    self.client.post("/analyze/stream", json={"message": message})
                )
                for message in ("message-a", "message-b")
            ]
            await asyncio.wait_for(both_models_started.wait(), timeout=1.0)
            release_models.set()
            responses = await asyncio.gather(*requests)

        first_events, second_events = (
            parse_sse(response.text) for response in responses
        )
        self.assertEqual(first_events[1][1]["summary"], "message-a")
        self.assertEqual(first_events[2][1]["analysis_id"], 1)
        self.assertEqual(second_events[1][1]["summary"], "message-b")
        self.assertEqual(second_events[2][1]["analysis_id"], 2)
