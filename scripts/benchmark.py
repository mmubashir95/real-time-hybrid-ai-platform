"""Run a small concurrent benchmark against POST /analyze."""

import argparse
import asyncio
import math
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter

import httpx

from app.services.ai_service import MODEL

DEFAULT_MESSAGE = "I was charged twice for my subscription."


@dataclass(frozen=True)
class RequestMeasurement:
    latency_ms: float
    success: bool


@dataclass(frozen=True)
class BenchmarkSummary:
    total_requests: int
    completed_requests: int
    successful_requests: int
    failed_requests: int
    duration_seconds: float
    requests_per_second: float
    error_rate_percent: float
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None


def percentile(values: list[float], percentile_value: float) -> float:
    """Calculate a percentile using linear interpolation between closest ranks."""
    if not values:
        raise ValueError("At least one value is required.")
    if not 0 <= percentile_value <= 100:
        raise ValueError("Percentile must be between 0 and 100.")

    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value / 100
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]

    weight = position - lower_index
    return ordered[lower_index] + (
        ordered[upper_index] - ordered[lower_index]
    ) * weight


def calculate_rps(completed_requests: int, duration_seconds: float) -> float:
    if completed_requests < 0:
        raise ValueError("Completed requests cannot be negative.")
    if duration_seconds <= 0:
        raise ValueError("Duration must be positive.")
    return completed_requests / duration_seconds


def calculate_error_rate(total_requests: int, failed_requests: int) -> float:
    if total_requests <= 0:
        raise ValueError("Total requests must be positive.")
    if not 0 <= failed_requests <= total_requests:
        raise ValueError("Failed requests must be between zero and total requests.")
    return failed_requests / total_requests * 100


def summarize(
    measurements: list[RequestMeasurement],
    duration_seconds: float,
) -> BenchmarkSummary:
    successful_latencies = [
        measurement.latency_ms
        for measurement in measurements
        if measurement.success
    ]
    failed_requests = sum(not measurement.success for measurement in measurements)
    if successful_latencies:
        p50_ms = percentile(successful_latencies, 50)
        p95_ms = percentile(successful_latencies, 95)
        p99_ms = percentile(successful_latencies, 99)
    else:
        p50_ms = p95_ms = p99_ms = None

    return BenchmarkSummary(
        total_requests=len(measurements),
        completed_requests=len(measurements),
        successful_requests=len(successful_latencies),
        failed_requests=failed_requests,
        duration_seconds=duration_seconds,
        requests_per_second=calculate_rps(len(measurements), duration_seconds),
        error_rate_percent=calculate_error_rate(
            len(measurements), failed_requests
        ),
        p50_ms=p50_ms,
        p95_ms=p95_ms,
        p99_ms=p99_ms,
    )


async def measure_request(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    endpoint: str,
    message: str,
) -> RequestMeasurement:
    async with semaphore:
        started_at = perf_counter()
        try:
            response = await client.post(endpoint, json={"message": message})
            success = 200 <= response.status_code < 300
        except httpx.HTTPError:
            success = False
        latency_ms = (perf_counter() - started_at) * 1000
    return RequestMeasurement(latency_ms=max(0.0, latency_ms), success=success)


async def run_benchmark(
    base_url: str,
    request_count: int,
    concurrency: int,
    message: str,
    timeout_seconds: float,
) -> BenchmarkSummary:
    if request_count <= 0:
        raise ValueError("Request count must be positive.")
    if concurrency <= 0:
        raise ValueError("Concurrency must be positive.")
    if timeout_seconds <= 0:
        raise ValueError("Timeout must be positive.")

    endpoint = f"{base_url.rstrip('/')}/analyze"
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        started_at = perf_counter()
        measurements = await asyncio.gather(
            *(
                measure_request(client, semaphore, endpoint, message)
                for _ in range(request_count)
            )
        )
        duration_seconds = perf_counter() - started_at
    return summarize(measurements, duration_seconds)


def _format_latency(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f} ms"


def print_report(
    summary: BenchmarkSummary,
    base_url: str,
    concurrency: int,
) -> None:
    print("Performance Benchmark")
    print("Environment:")
    print(f"  Run time (UTC): {datetime.now(UTC).isoformat()}")
    print(f"  Machine: {platform.platform()}")
    print(f"  Endpoint: {base_url.rstrip('/')}/analyze")
    print(f"  Model: {MODEL}")
    print("  Database: PostgreSQL")
    print(f"  Requests: {summary.total_requests}")
    print(f"  Concurrency: {concurrency}")
    print("End-to-end client-observed latency (successful requests):")
    print(f"  p50: {_format_latency(summary.p50_ms)}")
    print(f"  p95: {_format_latency(summary.p95_ms)}")
    print(f"  p99: {_format_latency(summary.p99_ms)}")
    print("Throughput:")
    print(f"  Completed: {summary.completed_requests}")
    print(f"  Successful: {summary.successful_requests}")
    print(f"  Failed: {summary.failed_requests}")
    print(f"  Duration: {summary.duration_seconds:.2f} s")
    print(f"  Requests/sec: {summary.requests_per_second:.2f}")
    print("Errors:")
    print(
        f"  {summary.failed_requests} / {summary.total_requests} "
        f"({summary.error_rate_percent:.2f}%)"
    )
    print(
        "Server logs contain correlated model_latency_ms, database_latency_ms, "
        "and end_to_end_latency_ms values."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--message", default=DEFAULT_MESSAGE)
    parser.add_argument("--timeout", type=float, default=60.0)
    arguments = parser.parse_args()
    if arguments.requests <= 0:
        parser.error("--requests must be positive")
    if arguments.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if arguments.timeout <= 0:
        parser.error("--timeout must be positive")
    return arguments


def main() -> None:
    arguments = parse_args()
    summary = asyncio.run(
        run_benchmark(
            arguments.url,
            arguments.requests,
            arguments.concurrency,
            arguments.message,
            arguments.timeout,
        )
    )
    print_report(summary, arguments.url, arguments.concurrency)


if __name__ == "__main__":
    main()
