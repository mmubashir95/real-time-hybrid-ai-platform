"""Calculate inspectable Phase 11 quality and operational metrics."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from statistics import median


EvidenceKey = tuple[str, str, str, str]


@dataclass(frozen=True)
class RankedEvidence:
    """Associate one retrieved source anchor with its optional relevance grade."""

    key: EvidenceKey
    relevance_grade: int = 0

    def __post_init__(self) -> None:
        """Reject negative grades before they distort ranking calculations."""
        if isinstance(self.relevance_grade, bool) or not isinstance(self.relevance_grade, int):
            raise TypeError("relevance_grade must be an integer")
        if self.relevance_grade < 0:
            raise ValueError("relevance_grade must be nonnegative")


@dataclass(frozen=True)
class RetrievalMetrics:
    """Carry per-question metrics with undefined no-answer values as None."""

    precision_at_k: float
    recall_at_k: float | None
    complete_evidence_coverage_at_k: bool | None
    reciprocal_rank_at_k: float | None
    ndcg_at_k: float | None
    unique_returned_at_k: int


def evaluate_ranking(
    ranked: tuple[RankedEvidence, ...],
    gold_relevance: dict[EvidenceKey, int],
    k: int,
) -> RetrievalMetrics:
    """Score one ordered result list against stable source-level relevance.

    Precision uses K as its denominator even when fewer than K unique results
    are returned. Recall, coverage, MRR, and NDCG are undefined for no-answer
    questions and therefore return None instead of an artificial perfect score.
    Duplicate source anchors receive credit only on their first occurrence.
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    if k <= 0:
        raise ValueError("k must be positive")
    if any(grade <= 0 for grade in gold_relevance.values()):
        raise ValueError("Gold relevance grades must be positive")

    unique_ranked: list[RankedEvidence] = []
    seen: set[EvidenceKey] = set()
    for item in ranked:
        if item.key not in seen:
            seen.add(item.key)
            unique_ranked.append(item)
        if len(unique_ranked) == k:
            break

    relevant = [item for item in unique_ranked if item.key in gold_relevance]
    precision = len(relevant) / k
    if not gold_relevance:
        return RetrievalMetrics(precision, None, None, None, None, len(unique_ranked))

    found = {item.key for item in relevant}
    recall = len(found) / len(gold_relevance)
    first_rank = next(
        (rank for rank, item in enumerate(unique_ranked, start=1) if item.key in gold_relevance),
        None,
    )
    reciprocal_rank = 1.0 / first_rank if first_rank else 0.0
    dcg = sum(
        (2 ** gold_relevance.get(item.key, 0) - 1) / log2(rank + 1)
        for rank, item in enumerate(unique_ranked, start=1)
        if item.key in gold_relevance
    )
    ideal_grades = sorted(gold_relevance.values(), reverse=True)[:k]
    ideal_dcg = sum(
        (2 ** grade - 1) / log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, start=1)
    )
    return RetrievalMetrics(
        precision_at_k=precision,
        recall_at_k=recall,
        complete_evidence_coverage_at_k=found == set(gold_relevance),
        reciprocal_rank_at_k=reciprocal_rank,
        ndcg_at_k=dcg / ideal_dcg,
        unique_returned_at_k=len(unique_ranked),
    )


def mean_defined(values: tuple[float | None, ...]) -> float | None:
    """Average only mathematically defined values and preserve empty denominators."""
    defined = [value for value in values if value is not None]
    return sum(defined) / len(defined) if defined else None


def rate(numerator: int, denominator: int) -> float | None:
    """Return a transparent rate or None when no reviewed observations exist."""
    if numerator < 0 or denominator < 0 or numerator > denominator:
        raise ValueError("Rate counts must satisfy 0 <= numerator <= denominator")
    return numerator / denominator if denominator else None


def latency_percentiles(samples: tuple[float, ...]) -> tuple[float, float] | None:
    """Return median and nearest-rank p95, or None when no samples were measured."""
    if any(value < 0 for value in samples):
        raise ValueError("Latency samples must be nonnegative")
    if not samples:
        return None
    ordered = sorted(samples)
    p95_index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return median(ordered), ordered[p95_index]
