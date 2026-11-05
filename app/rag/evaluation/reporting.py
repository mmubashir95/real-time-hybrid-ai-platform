"""Persist inspectable Phase 11 artifacts and render a conservative report."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field

from app.rag.evaluation.reviews import AnswerQualitySummary
from app.rag.evaluation.operations import OperationalSummary
from app.rag.evaluation.runner import AggregateRetrievalResult, QuestionResult


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
        "",
        "## Retrieval and ranking",
        "",
    ]
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
