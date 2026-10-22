"""Calculate reproducible retrieval-quality metrics from source-based labels."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from statistics import median


@dataclass(frozen=True)
class EvaluationExample:
    """Describe one ranked source result against one or more gold sources."""

    ranked_sources: tuple[tuple[str, ...], ...]
    relevant_sources: frozenset[tuple[str, ...]]


def recall_at_k(examples: tuple[EvaluationExample, ...], k: int) -> float:
    """Return mean relevant-evidence coverage among answerable examples."""
    if k <= 0:
        raise ValueError("k must be positive")
    answerable = [example for example in examples if example.relevant_sources]
    if not answerable:
        raise ValueError("Recall requires at least one labeled answerable example")
    return sum(
        len(set(example.ranked_sources[:k]) & example.relevant_sources)
        / len(example.relevant_sources)
        for example in answerable
    ) / len(answerable)


def mrr_at_k(examples: tuple[EvaluationExample, ...], k: int) -> float:
    """Return mean reciprocal rank of the first relevant source."""
    if k <= 0:
        raise ValueError("k must be positive")
    answerable = [example for example in examples if example.relevant_sources]
    if not answerable:
        raise ValueError("MRR requires at least one labeled answerable example")
    reciprocal_ranks: list[float] = []
    for example in answerable:
        first_rank = next(
            (rank for rank, source in enumerate(example.ranked_sources[:k], start=1)
             if source in example.relevant_sources),
            None,
        )
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def ndcg_at_k(examples: tuple[EvaluationExample, ...], k: int) -> float:
    """Return binary-relevance NDCG for labels without graded judgments."""
    if k <= 0:
        raise ValueError("k must be positive")
    answerable = [example for example in examples if example.relevant_sources]
    if not answerable:
        raise ValueError("NDCG requires at least one labeled answerable example")
    values: list[float] = []
    for example in answerable:
        dcg = sum(
            1.0 / log2(rank + 1)
            for rank, source in enumerate(example.ranked_sources[:k], start=1)
            if source in example.relevant_sources
        )
        ideal_count = min(k, len(example.relevant_sources))
        ideal = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_count + 1))
        values.append(dcg / ideal)
    return sum(values) / len(values)


def latency_percentiles(samples: tuple[float, ...]) -> tuple[float, float]:
    """Return median and nearest-rank p95 latency for measured samples."""
    if not samples:
        raise ValueError("Latency summary requires at least one sample")
    if any(value < 0 for value in samples):
        raise ValueError("Latency samples must be nonnegative")
    ordered = sorted(samples)
    p95_index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return median(ordered), ordered[p95_index]
