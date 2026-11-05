"""Build reproducible Phase 11 per-question and aggregate evaluation records."""

from __future__ import annotations

from enum import Enum
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field

from app.rag.evaluation.dataset import GoldEvidence
from app.rag.evaluation.metrics import (
    EvidenceKey,
    RankedEvidence,
    evaluate_ranking,
    latency_percentiles,
    mean_defined,
)


class EvaluationConfiguration(str, Enum):
    """Name the five Phase 11 experiment configurations."""

    DENSE = "A"
    KEYWORD = "B"
    HYBRID = "C"
    HYBRID_RERANKED = "D"
    GROUNDED = "E"


class CandidateRecord(BaseModel):
    """Preserve one ordered candidate and its stable source provenance."""

    model_config = ConfigDict(frozen=True)

    rank: int = Field(ge=1)
    chunk_id: str
    tenant_id: str
    document_id: str
    document_version: str
    section_key: str
    relevance_grade: int = Field(ge=0)
    score: float | None = None
    # Canonical gold section credited for this passage when it matched through
    # an equivalent section; None keeps the passage's own section identity.
    credited_section_key: str | None = None
    # Retrieval-stage provenance, persisted so ranking diagnostics can be read
    # from the artifact instead of reconstructed. None means the stage did not
    # run for this configuration (for example, no BM25 rank in dense-only A).
    retrieval_rank: int | None = Field(default=None, ge=1)
    bm25_rank: int | None = Field(default=None, ge=1)
    bm25_score: float | None = None
    semantic_rank: int | None = Field(default=None, ge=1)
    semantic_score: float | None = None
    fusion_score: float | None = None
    bm25_rrf_component: float | None = None
    semantic_rrf_component: float | None = None
    text_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @property
    def evidence_key(self) -> EvidenceKey:
        """Return the identity used for scoring: credited gold key or own source."""
        return (
            self.tenant_id,
            self.document_id,
            self.document_version,
            self.credited_section_key or self.section_key,
        )


def gold_relevance(gold_evidence: Iterable[GoldEvidence]) -> dict[EvidenceKey, int]:
    """Map each canonical gold evidence key to its relevance grade."""
    return {
        (item.tenant_id, item.document_id, item.document_version, item.section_key):
            item.relevance_grade
        for item in gold_evidence
    }


def _normalize_text(text: str) -> str:
    """Normalize case and whitespace so quote matching ignores layout."""
    return " ".join(text.casefold().split())


def match_gold_evidence(
    gold_evidence: Iterable[GoldEvidence],
    *,
    tenant_id: str,
    document_id: str,
    document_version: str,
    section_key: str,
    passage_text: str,
) -> tuple[int, str | None]:
    """Return (grade, credited canonical section) for one retrieved passage.

    A passage is relevant only when it comes from the labeled document version,
    sits in the canonical or a reviewer-listed equivalent section, and contains
    the normalized source quote. This credits repeated statements of the same
    fact without crediting same-section chunks that lack it. Returns (0, None)
    when nothing matches.
    """
    normalized_passage = _normalize_text(passage_text)
    best: tuple[int, str | None] = (0, None)
    for item in gold_evidence:
        if (item.tenant_id, item.document_id, item.document_version) != (
            tenant_id, document_id, document_version,
        ):
            continue
        if section_key not in (item.section_key, *item.equivalent_sections):
            continue
        if _normalize_text(item.source_quote) not in normalized_passage:
            continue
        if item.relevance_grade > best[0]:
            best = (item.relevance_grade, item.section_key)
    return best


class QuestionResult(BaseModel):
    """Store inspectable results for one question and configuration."""

    model_config = ConfigDict(frozen=True)

    question_id: str
    configuration: EvaluationConfiguration
    candidates: tuple[CandidateRecord, ...]
    precision_at_5: float
    precision_at_10: float
    recall_at_5: float | None
    recall_at_10: float | None
    recall_at_20: float | None
    complete_evidence_coverage_at_5: bool | None
    complete_evidence_coverage_at_10: bool | None
    complete_evidence_coverage_at_20: bool | None
    reciprocal_rank_at_10: float | None
    ndcg_at_5: float | None
    ndcg_at_10: float | None
    latency_ms: float = Field(ge=0)
    error_code: str | None = None
    # Number of distinct gold evidence keys; None for no-answer questions. It
    # bounds precision: with g gold keys, P@K can never exceed min(g, K) / K.
    gold_evidence_count: int | None = Field(default=None, ge=1)


class AggregateRetrievalResult(BaseModel):
    """Aggregate a configuration while retaining explicit sample counts."""

    model_config = ConfigDict(frozen=True)

    configuration: EvaluationConfiguration
    question_count: int
    answerable_question_count: int
    failed_question_count: int
    precision_at_5: float | None
    precision_at_10: float | None
    recall_at_5: float | None
    recall_at_10: float | None
    recall_at_20: float | None
    complete_evidence_coverage_at_5: float | None
    complete_evidence_coverage_at_10: float | None
    complete_evidence_coverage_at_20: float | None
    mrr_at_10: float | None
    ndcg_at_5: float | None
    ndcg_at_10: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    # Precision above averages every completed question, including no-answer
    # ones (always 0), while recall averages answerable questions only. These
    # answerable-only values share recall's denominator, and the ceilings are
    # the best precision any ranking could reach given the gold counts.
    answerable_precision_at_5: float | None = None
    answerable_precision_at_10: float | None = None
    precision_ceiling_at_5: float | None = None
    precision_ceiling_at_10: float | None = None


def build_question_result(
    question_id: str,
    configuration: EvaluationConfiguration,
    candidates: tuple[CandidateRecord, ...],
    gold_relevance: dict[EvidenceKey, int],
    latency_ms: float,
    *,
    error_code: str | None = None,
) -> QuestionResult:
    """Calculate every required retrieval/ranking cutoff for one question."""
    ranked = tuple(
        RankedEvidence(item.evidence_key, item.relevance_grade) for item in candidates
    )
    at_5 = evaluate_ranking(ranked, gold_relevance, 5)
    at_10 = evaluate_ranking(ranked, gold_relevance, 10)
    at_20 = evaluate_ranking(ranked, gold_relevance, 20)
    return QuestionResult(
        question_id=question_id,
        configuration=configuration,
        candidates=candidates,
        precision_at_5=at_5.precision_at_k,
        precision_at_10=at_10.precision_at_k,
        recall_at_5=at_5.recall_at_k,
        recall_at_10=at_10.recall_at_k,
        recall_at_20=at_20.recall_at_k,
        complete_evidence_coverage_at_5=at_5.complete_evidence_coverage_at_k,
        complete_evidence_coverage_at_10=at_10.complete_evidence_coverage_at_k,
        complete_evidence_coverage_at_20=at_20.complete_evidence_coverage_at_k,
        reciprocal_rank_at_10=at_10.reciprocal_rank_at_k,
        ndcg_at_5=at_5.ndcg_at_k,
        ndcg_at_10=at_10.ndcg_at_k,
        latency_ms=latency_ms,
        error_code=error_code,
        gold_evidence_count=len(gold_relevance) or None,
    )


def aggregate_question_results(
    configuration: EvaluationConfiguration,
    results: Iterable[QuestionResult],
) -> AggregateRetrievalResult:
    """Aggregate only matching results and keep undefined denominators visible."""
    items = tuple(results)
    if any(item.configuration is not configuration for item in items):
        raise ValueError("Cannot aggregate mixed evaluation configurations")
    completed = tuple(item for item in items if item.error_code is None)
    answerable = tuple(item for item in completed if item.recall_at_5 is not None)
    percentiles = latency_percentiles(tuple(item.latency_ms for item in completed))

    def precision_ceiling(k: int) -> float | None:
        """Average the best reachable P@K; unknown if any gold count is missing."""
        counts = [item.gold_evidence_count for item in answerable]
        known = [count for count in counts if count is not None]
        if not counts or len(known) != len(counts):
            return None
        return sum(min(count, k) / k for count in known) / len(known)

    def coverage(name: str) -> float | None:
        values = [getattr(item, name) for item in answerable]
        return sum(bool(value) for value in values) / len(values) if values else None

    return AggregateRetrievalResult(
        configuration=configuration,
        question_count=len(items),
        answerable_question_count=len(answerable),
        failed_question_count=len(items) - len(completed),
        precision_at_5=mean_defined(tuple(item.precision_at_5 for item in completed)),
        precision_at_10=mean_defined(tuple(item.precision_at_10 for item in completed)),
        recall_at_5=mean_defined(tuple(item.recall_at_5 for item in answerable)),
        recall_at_10=mean_defined(tuple(item.recall_at_10 for item in answerable)),
        recall_at_20=mean_defined(tuple(item.recall_at_20 for item in answerable)),
        complete_evidence_coverage_at_5=coverage("complete_evidence_coverage_at_5"),
        complete_evidence_coverage_at_10=coverage("complete_evidence_coverage_at_10"),
        complete_evidence_coverage_at_20=coverage("complete_evidence_coverage_at_20"),
        mrr_at_10=mean_defined(tuple(item.reciprocal_rank_at_10 for item in answerable)),
        ndcg_at_5=mean_defined(tuple(item.ndcg_at_5 for item in answerable)),
        ndcg_at_10=mean_defined(tuple(item.ndcg_at_10 for item in answerable)),
        latency_p50_ms=percentiles[0] if percentiles else None,
        latency_p95_ms=percentiles[1] if percentiles else None,
        answerable_precision_at_5=mean_defined(tuple(item.precision_at_5 for item in answerable)),
        answerable_precision_at_10=mean_defined(
            tuple(item.precision_at_10 for item in answerable)
        ),
        precision_ceiling_at_5=precision_ceiling(5),
        precision_ceiling_at_10=precision_ceiling(10),
    )


class CandidateSetMismatchError(ValueError):
    """Signal that a reranked run did not receive the baseline's candidates.

    This is a run-level comparability failure, not a per-question error, so
    callers must abort instead of recording it as one failed question.
    """


def assert_fixed_candidate_set(
    baseline: tuple[CandidateRecord, ...],
    reranked: tuple[CandidateRecord, ...],
) -> None:
    """Reject unfair reranker comparisons that add or remove candidates."""
    baseline_ids = [item.chunk_id for item in baseline]
    reranked_ids = [item.chunk_id for item in reranked]
    if len(baseline_ids) != len(set(baseline_ids)):
        raise ValueError("Baseline candidates must have unique chunk IDs")
    if len(reranked_ids) != len(set(reranked_ids)):
        raise ValueError("Reranked candidates must have unique chunk IDs")
    if set(baseline_ids) != set(reranked_ids):
        raise CandidateSetMismatchError("Reranker comparison must preserve the candidate set")
