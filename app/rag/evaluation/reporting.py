"""Persist inspectable Phase 11 artifacts and render a conservative report."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field

from app.rag.evaluation.reviews import AnswerQualitySummary
from app.rag.evaluation.operations import OperationalSummary
from app.rag.evaluation.runner import (
    AggregateRetrievalResult,
    CandidateRecord,
    EvaluationConfiguration,
    QuestionResult,
)


class RerankerRuntime(BaseModel):
    """Record how the reranker actually ran, since ranking depends on it.

    Precision is the configured value; "auto" resolves to float32 on CPU and to
    the checkpoint dtype (bfloat16) on accelerators. truncated_inputs is None
    when the pipeline does not expose per-call reranker diagnostics.
    """

    model_config = ConfigDict(frozen=True)

    device: str
    precision: str
    batch_size: int = Field(gt=0)
    max_length: int = Field(gt=0)
    model_load_seconds: float = Field(ge=0)
    truncated_inputs: int | None = Field(default=None, ge=0)
    # The first question pays one-time inference warm-up, so its latency is
    # not representative of steady-state reranking.
    warmup_question_id: str | None = None


class RunMetadata(BaseModel):
    """Freeze the identities and configuration required to reproduce one run."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(min_length=1)
    timestamp: datetime
    dataset_id: str
    dataset_version: str
    dataset_split: str
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_snapshot: str
    index_version: str
    embedding_model: str
    reranker_model: str | None = None
    reranker_runtime: RerankerRuntime | None = None
    # A reranked run (D) names the hybrid run (C) whose persisted candidates it
    # was verified against, so the C/D comparison is traceable to exact inputs.
    baseline_run_id: str | None = None
    baseline_artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    llm_model: str | None = None
    prompt_version: str | None = None
    evaluation_configuration: str
    candidate_limit: int = Field(gt=0)
    test_environment: str


class EvaluationArtifact(BaseModel):
    """Contain machine-readable per-question, aggregate, and review results."""

    model_config = ConfigDict(frozen=True)

    metadata: RunMetadata
    per_question: tuple[QuestionResult, ...] = ()
    grounded_answers: tuple[dict[str, object], ...] = ()
    retrieval_aggregate: AggregateRetrievalResult | None = None
    answer_quality: AnswerQualitySummary | None = None
    operations: OperationalSummary | None = None
    security_results: Mapping[str, str] = Field(default_factory=dict)
    limitations: tuple[str, ...] = ()


def write_artifact(artifact: EvaluationArtifact, output_path: Path) -> None:
    """Atomically publish one JSON result after complete serialization succeeds."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        artifact.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)


def read_artifact(path: Path) -> EvaluationArtifact:
    """Load a previously validated machine-readable evaluation artifact."""
    return EvaluationArtifact.model_validate_json(path.read_text(encoding="utf-8"))


class CandidateBaseline(BaseModel):
    """Hold one verified hybrid run's candidates for a fixed-set reranker run."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    artifact_sha256: str
    candidates_by_question: Mapping[str, tuple[CandidateRecord, ...]]


def load_candidate_baseline(
    path: Path,
    *,
    dataset_sha256: str,
    dataset_split: str,
    corpus_snapshot: str,
    index_version: str,
    embedding_model: str,
    candidate_limit: int,
    question_ids: tuple[str, ...],
) -> CandidateBaseline:
    """Load a configuration C artifact that a D run must reproduce exactly.

    Raises ValueError when the artifact is not a hybrid run, was produced
    against different dataset, corpus, index, embedding, or candidate-limit
    settings, or lacks a successful result for any requested question. These
    checks make an unfair C/D comparison fail instead of being reported.
    """
    raw = path.read_bytes()
    artifact = EvaluationArtifact.model_validate_json(raw)
    metadata = artifact.metadata
    if metadata.evaluation_configuration != EvaluationConfiguration.HYBRID.value:
        raise ValueError("The candidate baseline must be a configuration C run")
    expected = {
        "dataset_sha256": dataset_sha256,
        "dataset_split": dataset_split,
        "corpus_snapshot": corpus_snapshot,
        "index_version": index_version,
        "embedding_model": embedding_model,
        "candidate_limit": candidate_limit,
    }
    for field, value in expected.items():
        if getattr(metadata, field) != value:
            raise ValueError(f"Candidate baseline {field} does not match this run")
    results = {item.question_id: item for item in artifact.per_question}
    candidates_by_question: dict[str, tuple[CandidateRecord, ...]] = {}
    for question_id in question_ids:
        result = results.get(question_id)
        # A failed or absent C question has no candidate set to hold fixed.
        if result is None or result.error_code is not None:
            raise ValueError(
                f"Candidate baseline has no successful result for {question_id}"
            )
        candidates_by_question[question_id] = result.candidates
    return CandidateBaseline(
        run_id=metadata.run_id,
        artifact_sha256=hashlib.sha256(raw).hexdigest(),
        candidates_by_question=candidates_by_question,
    )


def _metric(value: float | None) -> str:
    """Render absent measurements explicitly instead of inventing zero."""
    return "NOT MEASURED" if value is None else f"{value:.4f}"


def render_markdown_report(artifact: EvaluationArtifact) -> str:
    """Render a human-readable report whose claims trace to the JSON artifact."""
    metadata = artifact.metadata
    retrieval = artifact.retrieval_aggregate
    answer = artifact.answer_quality
    lines = [
        "# Month 2 RAG Evaluation Report",
        "",
        "## Run identity",
        "",
        f"- Run ID: `{metadata.run_id}`",
        f"- Timestamp: `{metadata.timestamp.isoformat()}`",
        f"- Dataset: `{metadata.dataset_id}` v`{metadata.dataset_version}`",
        f"- Split: `{metadata.dataset_split}`",
        f"- Configuration: `{metadata.evaluation_configuration}`",
        f"- Corpus snapshot: `{metadata.corpus_snapshot}`",
        f"- Index version: `{metadata.index_version}`",
        f"- Test environment: {metadata.test_environment}",
    ]
    if metadata.baseline_run_id is not None:
        lines.append(
            f"- Candidate baseline (C) run: `{metadata.baseline_run_id}` "
            f"(artifact SHA-256 `{metadata.baseline_artifact_sha256}`)"
        )
    runtime = metadata.reranker_runtime
    if runtime is not None:
        truncated = (
            "NOT MEASURED" if runtime.truncated_inputs is None
            else str(runtime.truncated_inputs)
        )
        lines.extend([
            f"- Reranker: `{metadata.reranker_model}` on `{runtime.device}`, "
            f"precision `{runtime.precision}`, batch size {runtime.batch_size}, "
            f"max length {runtime.max_length}",
            f"- Reranker model load seconds: {runtime.model_load_seconds:.2f}",
            f"- Reranker truncated inputs: {truncated}",
            f"- Warm-up question (latency includes first-inference cost): "
            f"`{runtime.warmup_question_id}`",
        ])
    lines.extend([
        "",
        "## Retrieval and ranking",
        "",
    ])
    if retrieval is None:
        lines.append("NOT MEASURED")
    else:
        lines.extend([
            f"- Questions: {retrieval.question_count}",
            f"- Failed questions: {retrieval.failed_question_count}",
            f"- Precision@5: {_metric(retrieval.precision_at_5)}",
            f"- Precision@10: {_metric(retrieval.precision_at_10)}",
            f"- Recall@5: {_metric(retrieval.recall_at_5)}",
            f"- Recall@10: {_metric(retrieval.recall_at_10)}",
            f"- Recall@20: {_metric(retrieval.recall_at_20)}",
            f"- Complete Evidence Coverage@5: {_metric(retrieval.complete_evidence_coverage_at_5)}",
            f"- MRR@10: {_metric(retrieval.mrr_at_10)}",
            f"- NDCG@5: {_metric(retrieval.ndcg_at_5)}",
            f"- NDCG@10: {_metric(retrieval.ndcg_at_10)}",
            f"- Latency p50 ms: {_metric(retrieval.latency_p50_ms)}",
            f"- Latency p95 ms: {_metric(retrieval.latency_p95_ms)}",
        ])
    lines.extend(["", "## Answer quality and grounding", ""])
    if answer is None:
        lines.append("PENDING HUMAN REVIEW")
    else:
        lines.extend([
            f"- Reviewed questions: {answer.reviewed_questions}",
            f"- Reviewed claims: {answer.reviewed_claims}",
            f"- Claim support rate: {_metric(answer.claim_support_rate)}",
            f"- Unsupported claim rate: {_metric(answer.unsupported_claim_rate)}",
            f"- Citation correctness: {_metric(answer.citation_correctness)}",
            f"- Abstention accuracy: {_metric(answer.abstention_accuracy)}",
            f"- Grounding failure rate: {_metric(answer.grounding_failure_rate)}",
        ])
    lines.extend(["", "## Security", ""])
    if artifact.security_results:
        lines.extend(f"- {name}: {result}" for name, result in artifact.security_results.items())
    else:
        lines.append("NOT MEASURED")
    lines.extend(["", "## Performance, usage, cost, and reliability", ""])
    if artifact.operations is None:
        lines.append("NOT MEASURED")
    else:
        operations = artifact.operations
        lines.extend([
            f"- Requests: {operations.request_count}",
            f"- Success rate: {_metric(operations.success_rate)}",
            f"- Failure rate: {_metric(operations.failure_rate)}",
            f"- Total tokens: {operations.total_tokens if operations.total_tokens is not None else 'NOT MEASURED'}",
            f"- Total estimated cost USD: {_metric(operations.total_estimated_cost_usd)}",
        ])
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in artifact.limitations or ("None recorded.",))
    return "\n".join(lines) + "\n"


def write_markdown_report(artifact: EvaluationArtifact, output_path: Path) -> None:
    """Write the deterministic Markdown projection of one JSON artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown_report(artifact), encoding="utf-8")


def new_timestamp() -> datetime:
    """Return an aware UTC timestamp for callers creating run metadata."""
    return datetime.now(timezone.utc)
