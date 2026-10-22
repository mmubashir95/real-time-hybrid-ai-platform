#!/usr/bin/env python3
"""Evaluate all retrieval modes against one frozen, reviewed question split."""

from __future__ import annotations

import argparse
import asyncio
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

import asyncpg
from sqlalchemy.engine import make_url


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.embeddings.encoder import SentenceTransformerEncoder  # noqa: E402
from app.rag.embeddings.models import EmbeddingConfig  # noqa: E402
from app.rag.embeddings.service import question_search_scope  # noqa: E402
from app.rag.persistence.lifecycle import EmbeddingProfile  # noqa: E402
from app.rag.retrieval import (  # noqa: E402
    DenseRetriever, HistoricalVersionScope, KeywordRetriever, RetrievalService,
    SearchContextOrigin, SearchFilters, SearchMode, SearchRequest, TrustedSearchContext,
)
from app.rag.retrieval.metrics import (  # noqa: E402
    EvaluationExample, latency_percentiles, mrr_at_k, ndcg_at_k, recall_at_k,
)


DEFAULT_QUESTIONS = PROJECT_ROOT / "data/rag/evaluation/question_bank_SYNTHETIC_REVIEW_REQUIRED.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/rag/evaluation/retrieval_report.json"


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse frozen dataset, retrieval configuration, and output options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--split", default="heldout_test")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config-fingerprint", required=True)
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", ""))
    parser.add_argument("--schema", default="public")
    parser.add_argument("--visibility", action="append", choices=("public", "internal"))
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--allow-unreviewed-labels", action="store_true",
        help="Run exploratory metrics despite labels marked as review-required.",
    )
    return parser.parse_args(arguments)


def _load_questions(path: Path, split: str, allow_unreviewed: bool) -> list[dict[str, str]]:
    """Load one split and prevent accidental publication of unreviewed metrics."""
    with path.open(encoding="utf-8", newline="") as source:
        rows = [row for row in csv.DictReader(source) if row["split"] == split]
    if not rows:
        raise ValueError(f"No questions found for split {split!r}")
    unreviewed = [row["question_id"] for row in rows if row["annotation_status"] != "reviewed"]
    if unreviewed and not allow_unreviewed:
        raise ValueError(
            f"{len(unreviewed)} labels are not reviewed; use --allow-unreviewed-labels "
            "only for explicitly exploratory results"
        )
    return rows


def _question_context(
    row: dict[str, str], visibilities: tuple[str, ...]
) -> TrustedSearchContext:
    """Build the trusted scope for one question, matching the Phase 5/6 convention.

    Only explicitly historical questions are authorized for their exact archived
    version; every other question searches active documents in its tenant, so
    gold labels never narrow ordinary candidate sets.
    """
    document_id, version = question_search_scope(row)
    historical = (
        HistoricalVersionScope(document_id, version)
        if document_id is not None and version is not None
        else None
    )
    return TrustedSearchContext(
        row["tenant_id"], visibilities, SearchContextOrigin.INTERNAL,
        historical, historical is not None,
    )


def _section_key(headings: tuple[str, ...]) -> str:
    """Resolve the source section marker used by the frozen label bank."""
    for heading in reversed(headings):
        if heading.startswith("[") and "]" in heading:
            return heading[1:heading.index("]")]
    return ""


def _source_key(passage: object) -> tuple[str, str, str, str]:
    """Use stable document provenance rather than mutable chunk identity."""
    return (
        passage.tenant_id, passage.document_id, passage.document_version,
        _section_key(passage.section),
    )


async def _evaluate(arguments: argparse.Namespace) -> dict[str, object]:
    """Execute identical questions and authorization scopes for every mode."""
    rows = _load_questions(arguments.questions, arguments.split, arguments.allow_unreviewed_labels)
    config = EmbeddingConfig(device=arguments.device)
    profile = EmbeddingProfile(
        config.model_id, config.model_revision, arguments.config_fingerprint,
        config.normalize_embeddings, config.encoding_config_version, config.query_instruction,
    )
    if not arguments.database_url:
        raise ValueError("DATABASE_URL or --database-url is required")
    url = make_url(arguments.database_url)
    dsn = url.set(drivername="postgresql").render_as_string(hide_password=False)
    encoder = SentenceTransformerEncoder(config)
    connection = await asyncpg.connect(dsn, timeout=10)
    try:
        service = RetrievalService(
            KeywordRetriever(connection, schema=arguments.schema),
            DenseRetriever(connection, encoder, schema=arguments.schema),
        )
        mode_examples: dict[SearchMode, list[EvaluationExample]] = {mode: [] for mode in SearchMode}
        mode_latencies: dict[SearchMode, list[float]] = {mode: [] for mode in SearchMode}
        empty_counts = {mode: 0 for mode in SearchMode}
        security_violations = {mode: 0 for mode in SearchMode}
        for row in rows:
            context = _question_context(
                row, tuple(arguments.visibility or ("public", "internal"))
            )
            no_answer = row["expected_no_answer"].strip().lower() == "true"
            gold = frozenset() if no_answer else frozenset((
                (
                    row["tenant_id"], row["gold_document_id"],
                    row["gold_version"], row["gold_section_key"],
                ),
            ))
            for mode in SearchMode:
                response = await service.search(SearchRequest(
                    row["query"], mode, SearchFilters(), 20, context, profile,
                    arguments.candidate_limit, arguments.candidate_limit,
                ))
                ranked = tuple(_source_key(passage) for passage in response.results)
                mode_examples[mode].append(EvaluationExample(ranked, gold))
                mode_latencies[mode].append(response.diagnostics.total_seconds)
                empty_counts[mode] += not response.results
                security_violations[mode] += sum(
                    passage.tenant_id != row["tenant_id"] for passage in response.results
                )
    finally:
        await connection.close()

    metrics: dict[str, object] = {}
    for mode in SearchMode:
        examples = tuple(mode_examples[mode])
        p50, p95 = latency_percentiles(tuple(mode_latencies[mode]))
        metrics[mode.value] = {
            "recall_at_5": recall_at_k(examples, 5),
            "recall_at_10": recall_at_k(examples, 10),
            "recall_at_20": recall_at_k(examples, 20),
            "mrr_at_10": mrr_at_k(examples, 10),
            "ndcg_at_10_binary": ndcg_at_k(examples, 10),
            "p50_latency_seconds": p50,
            "p95_latency_seconds": p95,
            "empty_result_rate": empty_counts[mode] / len(rows),
            "unauthorized_passages": security_violations[mode],
        }
    dataset_bytes = arguments.questions.read_bytes()
    return {
        "status": "EXPLORATORY_UNREVIEWED" if arguments.allow_unreviewed_labels else "REVIEWED",
        "configuration": {
            "evaluation_dataset": str(arguments.questions),
            "evaluation_dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
            "split": arguments.split,
            "questions": len(rows),
            "embedding": asdict(config),
            "embedding_config_fingerprint": arguments.config_fingerprint,
            "bm25": {"implementation": "local Okapi BM25", "k1": 1.5, "b": 0.75},
            "vector_index": "exact pgvector cosine distance",
            "candidate_limit": arguments.candidate_limit,
            "rrf_constant": 60,
            "filter_policy": (
                "verified tenant + permitted visibility + active versions; "
                "version_conflict questions use their exact archived version"
            ),
        },
        "metrics": metrics,
    }


def main(arguments: list[str] | None = None) -> int:
    """Write an inspectable JSON report or return a nonzero failure status."""
    parsed = parse_args(arguments)
    try:
        report = asyncio.run(_evaluate(parsed))
        parsed.output.parent.mkdir(parents=True, exist_ok=True)
        parsed.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (
        ValueError, TypeError, OSError, TimeoutError,
        asyncpg.PostgresError, RuntimeError,
    ) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1
    print(f"Evaluation report written to {parsed.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
