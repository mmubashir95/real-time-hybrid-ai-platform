"""Aggregate Phase 11 latency, usage, cost, and reliability observations."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from app.rag.evaluation.metrics import latency_percentiles, rate


class RequestOutcome(str, Enum):
    """Classify one request without conflating abstention and failure."""

    SUCCESS = "success"
    CORRECT_ABSTENTION = "correct_abstention"
    RETRIEVAL_FAILURE = "retrieval_failure"
    RERANKER_FAILURE = "reranker_failure"
    GENERATION_FAILURE = "generation_failure"
    VERIFICATION_FAILURE = "verification_failure"
    INVALID_RESPONSE = "invalid_response"
    TIMEOUT = "timeout"
    UNHANDLED_ERROR = "unhandled_error"


class OperationalSample(BaseModel):
    """Record actual stage timing, provider usage, cost, and request outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question_id: str
    retrieval_ms: float | None = Field(default=None, ge=0)
    reranking_ms: float | None = Field(default=None, ge=0)
    context_building_ms: float | None = Field(default=None, ge=0)
    generation_ms: float | None = Field(default=None, ge=0)
    verification_ms: float | None = Field(default=None, ge=0)
    final_rendering_ms: float | None = Field(default=None, ge=0)
    end_to_end_ms: float | None = Field(default=None, ge=0)
    generation_input_tokens: int | None = Field(default=None, ge=0)
    generation_output_tokens: int | None = Field(default=None, ge=0)
    verification_input_tokens: int | None = Field(default=None, ge=0)
    verification_output_tokens: int | None = Field(default=None, ge=0)
    generation_estimated_cost_usd: float | None = Field(default=None, ge=0)
    verification_estimated_cost_usd: float | None = Field(default=None, ge=0)
    outcome: RequestOutcome


class OperationalSummary(BaseModel):
    """Expose stage distributions, totals, and failure denominators."""

    model_config = ConfigDict(frozen=True)

    request_count: int
    successful_requests: int
    correct_abstentions: int
    failure_counts: dict[str, int]
    success_rate: float | None
    failure_rate: float | None
    latency_ms: dict[str, dict[str, float | int] | None]
    generation_input_tokens: int | None
    generation_output_tokens: int | None
    verification_input_tokens: int | None
    verification_output_tokens: int | None
    total_tokens: int | None
    generation_estimated_cost_usd: float | None
    verification_estimated_cost_usd: float | None
    total_estimated_cost_usd: float | None


def _sum_if_complete(samples: tuple[OperationalSample, ...], field: str) -> int | float | None:
    """Sum a measurement only when every request reported it."""
    values = [getattr(sample, field) for sample in samples]
    return sum(values) if values and all(value is not None for value in values) else None


def summarize_operations(samples: tuple[OperationalSample, ...]) -> OperationalSummary:
    """Aggregate measured operations without replacing unavailable values with zero."""
    latency_fields = (
        "retrieval_ms", "reranking_ms", "context_building_ms", "generation_ms",
        "verification_ms", "final_rendering_ms", "end_to_end_ms",
    )
    latency: dict[str, dict[str, float | int] | None] = {}
    for field in latency_fields:
        values = tuple(
            value for sample in samples
            if (value := getattr(sample, field)) is not None
        )
        percentiles = latency_percentiles(values)
        latency[field] = (
            {"sample_count": len(values), "p50": percentiles[0], "p95": percentiles[1]}
            if percentiles else None
        )

    successful = sum(
        sample.outcome in {RequestOutcome.SUCCESS, RequestOutcome.CORRECT_ABSTENTION}
        for sample in samples
    )
    correct_abstentions = sum(
        sample.outcome is RequestOutcome.CORRECT_ABSTENTION for sample in samples
    )
    failure_counts = {
        outcome.value: sum(sample.outcome is outcome for sample in samples)
        for outcome in RequestOutcome
        if outcome not in {RequestOutcome.SUCCESS, RequestOutcome.CORRECT_ABSTENTION}
    }
    generation_input = _sum_if_complete(samples, "generation_input_tokens")
    generation_output = _sum_if_complete(samples, "generation_output_tokens")
    verification_input = _sum_if_complete(samples, "verification_input_tokens")
    verification_output = _sum_if_complete(samples, "verification_output_tokens")
    token_parts = (generation_input, generation_output, verification_input, verification_output)
    total_tokens = sum(token_parts) if all(value is not None for value in token_parts) else None
    generation_cost = _sum_if_complete(samples, "generation_estimated_cost_usd")
    verification_cost = _sum_if_complete(samples, "verification_estimated_cost_usd")
    total_cost = (
        generation_cost + verification_cost
        if generation_cost is not None and verification_cost is not None
        else None
    )
    return OperationalSummary(
        request_count=len(samples),
        successful_requests=successful,
        correct_abstentions=correct_abstentions,
        failure_counts=failure_counts,
        success_rate=rate(successful, len(samples)),
        failure_rate=rate(len(samples) - successful, len(samples)),
        latency_ms=latency,
        generation_input_tokens=generation_input,
        generation_output_tokens=generation_output,
        verification_input_tokens=verification_input,
        verification_output_tokens=verification_output,
        total_tokens=total_tokens,
        generation_estimated_cost_usd=generation_cost,
        verification_estimated_cost_usd=verification_cost,
        total_estimated_cost_usd=total_cost,
    )
