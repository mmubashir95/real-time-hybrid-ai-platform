import unittest

from scripts.benchmark import (
    RequestMeasurement,
    calculate_error_rate,
    calculate_rps,
    percentile,
    summarize,
)


class BenchmarkMathTests(unittest.TestCase):
    def test_percentiles_use_linear_interpolation(self) -> None:
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0]

        self.assertEqual(percentile(latencies, 50), 30.0)
        self.assertAlmostEqual(percentile(latencies, 95), 48.0)
        self.assertAlmostEqual(percentile(latencies, 99), 49.6)

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
        self.assertEqual(summary.completed_requests, 3)
        self.assertEqual(summary.successful_requests, 2)
        self.assertEqual(summary.failed_requests, 1)
        self.assertEqual(summary.p50_ms, 150.0)
        self.assertEqual(summary.requests_per_second, 2.0)
        self.assertAlmostEqual(summary.error_rate_percent, 100 / 3)

    def test_all_failures_have_no_success_latency_percentiles(self) -> None:
        summary = summarize(
            [RequestMeasurement(latency_ms=10.0, success=False)],
            duration_seconds=1.0,
        )

        self.assertIsNone(summary.p50_ms)
        self.assertIsNone(summary.p95_ms)
        self.assertIsNone(summary.p99_ms)
