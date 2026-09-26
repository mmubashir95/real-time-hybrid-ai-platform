import asyncio
import unittest

import httpx

from scripts.benchmark import (
    RequestMeasurement,
    calculate_error_rate,
    calculate_rps,
    percentile,
    run_benchmark,
    summarize,
)


class BenchmarkMathTests(unittest.TestCase):
    def test_percentiles_use_linear_interpolation(self) -> None:
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0]

        self.assertEqual(percentile(latencies, 50), 30.0)
        self.assertAlmostEqual(percentile(latencies, 95), 48.0)
        self.assertAlmostEqual(percentile(latencies, 99), 49.6)

    def test_percentile_edge_cases(self) -> None:
        self.assertEqual(percentile([100.0, 200.0, 300.0], 50), 200.0)
        self.assertEqual(percentile([500.0, 100.0, 300.0, 200.0, 400.0], 50), 300.0)
        self.assertEqual(percentile([100.0], 99), 100.0)
        with self.assertRaises(ValueError):
            percentile([], 50)

    def test_error_rate_for_zero_some_and_all_failures(self) -> None:
        self.assertEqual(calculate_error_rate(10, 0), 0.0)
        self.assertEqual(calculate_error_rate(10, 2), 20.0)
        self.assertEqual(calculate_error_rate(10, 10), 100.0)

    def test_requests_per_second_uses_completed_requests(self) -> None:
        self.assertEqual(calculate_rps(20, 4.0), 5.0)

    def test_summary_uses_successful_requests_for_latency_percentiles(self) -> None:
        summary = summarize(
            [
                RequestMeasurement(latency_ms=100.0, success=True),
                RequestMeasurement(latency_ms=200.0, success=True),
                RequestMeasurement(latency_ms=5.0, success=False),
            ],
            duration_seconds=1.5,
        )

        self.assertEqual(summary.total_requests, 3)
        self.assertEqual(summary.successful_requests, 2)
        self.assertEqual(summary.failed_requests, 1)
        self.assertEqual(summary.p50_ms, 150.0)
        self.assertEqual(summary.requests_per_second, 2.0)
        self.assertAlmostEqual(summary.successful_requests_per_second, 4 / 3)
        self.assertAlmostEqual(summary.error_rate_percent, 100 / 3)

    def test_all_failures_have_no_success_latency_percentiles(self) -> None:
        summary = summarize(
            [RequestMeasurement(latency_ms=10.0, success=False)],
            duration_seconds=1.0,
        )

        self.assertIsNone(summary.p50_ms)
        self.assertIsNone(summary.p95_ms)
        self.assertIsNone(summary.p99_ms)


class BenchmarkRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_non_2xx_and_transport_errors_count_as_failures(self) -> None:
        outcomes = iter((200, 422, 503, "connect-error"))

        def handler(request: httpx.Request) -> httpx.Response:
            outcome = next(outcomes)
            if outcome == "connect-error":
                raise httpx.ConnectError("refused", request=request)
            return httpx.Response(outcome, json={})

        summary = await run_benchmark(
            "http://test",
            request_count=4,
            concurrency=1,
            message="Hello",
            timeout_seconds=1.0,
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(summary.total_requests, 4)
        self.assertEqual(summary.successful_requests, 1)
        self.assertEqual(summary.failed_requests, 3)
        self.assertEqual(summary.error_rate_percent, 75.0)

    async def test_concurrency_limit_is_respected(self) -> None:
        in_flight = 0
        max_in_flight = 0

        async def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            # Yield so other requests can start if the limit were not enforced.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            in_flight -= 1
            return httpx.Response(200, json={})

        summary = await run_benchmark(
            "http://test",
            request_count=20,
            concurrency=3,
            message="Hello",
            timeout_seconds=1.0,
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(summary.successful_requests, 20)
        self.assertEqual(max_in_flight, 3)

    async def test_warmup_requests_are_sent_but_excluded(self) -> None:
        received = 0

        def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal received
            received += 1
            # Fail every warm-up request so any leak into the results shows up.
            return httpx.Response(503 if received <= 2 else 200, json={})

        summary = await run_benchmark(
            "http://test",
            request_count=5,
            concurrency=1,
            message="Hello",
            timeout_seconds=1.0,
            warmup_requests=2,
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(received, 7)
        self.assertEqual(summary.total_requests, 5)
        self.assertEqual(summary.failed_requests, 0)


if __name__ == "__main__":
    unittest.main()
