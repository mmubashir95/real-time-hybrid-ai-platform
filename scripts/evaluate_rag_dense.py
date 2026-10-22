#!/usr/bin/env python3
"""Run read-only dense-retrieval integrity, parity, quality, and latency checks."""

from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, TextIO

import asyncpg
import numpy as np
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.embeddings.encoder import SentenceTransformerEncoder  # noqa: E402
from app.rag.embeddings.models import EmbeddingConfig  # noqa: E402
from app.rag.embeddings.service import (  # noqa: E402
    load_embedding_artifacts,
    question_search_scope,
)
from app.rag.persistence.artifacts import (  # noqa: E402
    load_import_plan,
    resolve_embedding_run,
)
from app.rag.persistence.lifecycle import EmbeddingProfile  # noqa: E402
from app.rag.retrieval.dense import DenseRetriever  # noqa: E402
from app.rag.retrieval.models import (  # noqa: E402
    DenseSearchRequest,
    HistoricalVersionScope,
    SearchContextOrigin,
    TrustedSearchContext,
)
from app.rag.retrieval.verification import (  # noqa: E402
    artifact_run_fingerprint,
    compare_exact_ranking,
    database_versions,
    latency_summary,
    reconcile_import_plan,
    runtime_environment,
)


DEFAULT_MANIFEST = PROJECT_ROOT / "data/rag/manifest.csv"
DEFAULT_RAW = PROJECT_ROOT / "data/rag/raw"
DEFAULT_CHUNKS = PROJECT_ROOT / "data/rag/processed/chunks.jsonl"
DEFAULT_CHUNK_REPORT = PROJECT_ROOT / "data/rag/processed/chunking_report.json"
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "data/rag/processed/embeddings"
DEFAULT_QUESTIONS = (
    PROJECT_ROOT
    / "data/rag/evaluation/question_bank_SYNTHETIC_REVIEW_REQUIRED.csv"
)
DEFAULT_RESULTS = PROJECT_ROOT / "data/rag/evaluation/results"


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse explicit artifact, scope, database, and report configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", ""))
    parser.add_argument("--schema", default="public")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-directory", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--chunking-report", type=Path, default=DEFAULT_CHUNK_REPORT)
    parser.add_argument("--embedding-root", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--question-bank", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--visibility", action="append", choices=("public", "internal"),
        default=None, help="Production scope; repeat to permit multiple values.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args(arguments)


def _database_dsn(database_url: str) -> str:
    """Validate a PostgreSQL URL and return its asyncpg-compatible form."""
    if not database_url:
        raise ValueError("DATABASE_URL or --database-url is required")
    try:
        url = make_url(database_url)
    except ArgumentError as error:
        raise ValueError("DATABASE_URL is not valid") from error
    if url.drivername not in {"postgresql", "postgresql+asyncpg"}:
        raise ValueError("DATABASE_URL must use PostgreSQL with asyncpg")
    return url.set(drivername="postgresql").render_as_string(hide_password=False)


def _question_bank_identity(path: Path) -> str:
    """Hash the unchanged evaluation labels used by this run."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_questions(path: Path, split: str) -> list[dict[str, str]]:
    """Load and minimally validate one frozen evaluation partition."""
    with path.open(encoding="utf-8", newline="") as csv_file:
        rows = [dict(row) for row in csv.DictReader(csv_file) if row["split"] == split]
    if not rows:
        raise ValueError(f"No evaluation questions found for split: {split}")
    required = {
        "question_id", "query", "tenant_id", "expected_no_answer",
        "gold_document_id", "gold_version", "gold_section_key", "question_type",
    }
    for row in rows:
        if not required.issubset(row) or not row["question_id"].strip():
            raise ValueError("Evaluation question bank is missing required fields")
    return rows


def _profile(plan: Any) -> EmbeddingProfile:
    """Derive the exact database embedding profile from the validated plan."""
    embedding = plan.embeddings[0]
    return EmbeddingProfile(
        model_id=embedding.model_id,
        model_revision=embedding.model_revision,
        config_fingerprint=embedding.embedding_config_fingerprint,
        normalize_embeddings=embedding.normalize_embeddings,
        encoding_config_version=embedding.encoding_config_version,
        query_instruction=embedding.query_instruction,
    )


def _context(
    question: dict[str, str], visibilities: tuple[str, ...]
) -> TrustedSearchContext:
    """Build trusted test scope using the established historical-label rule."""
    document_id, version = question_search_scope(question)
    historical = (
        HistoricalVersionScope(document_id, version)
        if document_id is not None and version is not None else None
    )
    return TrustedSearchContext(
        tenant_id=question["tenant_id"],
        permitted_visibilities=visibilities,
        origin=SearchContextOrigin.INTERNAL,
        historical=historical,
        historical_access=historical is not None,
    )


def _is_hit(
    chunk_ids: tuple[str, ...] | list[str],
    records_by_id: dict[str, Any],
    question: dict[str, str],
    top_k: int,
) -> bool:
    """Apply the Phase 5 document+version+section anchor semantics."""
    return any(
        records_by_id[chunk_id].document_id == question["gold_document_id"]
        and records_by_id[chunk_id].document_version == question["gold_version"]
        and question["gold_section_key"]
        in " > ".join(records_by_id[chunk_id].heading_path)
        for chunk_id in chunk_ids[:top_k]
    )


def _gold_is_reachable(
    question: dict[str, str],
    gold_visibilities: dict[tuple[str, str, str], str],
    permitted_visibilities: tuple[str, ...],
) -> bool:
    """Return whether the question's gold version is visible under a scope.

    A question whose gold document is hidden by policy can never score a hit,
    so it measures the visibility rule rather than retrieval quality.
    """
    gold_visibility = gold_visibilities.get(
        (question["tenant_id"], question["gold_document_id"], question["gold_version"])
    )
    return gold_visibility in permitted_visibilities


async def _evaluate(arguments: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Execute one read-only evaluation and return summary and per-query records."""
    run_directory = resolve_embedding_run(
        arguments.embedding_root, arguments.run_directory
    )
    vectors, records, artifact_report = load_embedding_artifacts(run_directory)
    plan = load_import_plan(
        manifest_path=arguments.manifest,
        raw_directory=arguments.raw_directory,
        chunks_path=arguments.chunks,
        chunking_report_path=arguments.chunking_report,
        embedding_root=arguments.embedding_root,
        run_directory=run_directory,
    )
    questions = _load_questions(arguments.question_bank, arguments.split)
    profile = _profile(plan)
    config = EmbeddingConfig(device=arguments.device)
    config_identity = (
        config.model_id, config.model_revision, config.expected_dimensions,
        config.normalize_embeddings, config.encoding_config_version,
        config.query_instruction,
    )
    profile_identity = (
        profile.model_id, profile.model_revision, plan.dimensions,
        profile.normalize_embeddings, profile.encoding_config_version,
        profile.query_instruction,
    )
    if config_identity != profile_identity:
        raise ValueError("Runtime encoder configuration is incompatible with artifact run")

    connection = await asyncpg.connect(_database_dsn(arguments.database_url), timeout=10)
    try:
        integrity = await reconcile_import_plan(
            connection, plan, schema=arguments.schema
        )
        if integrity.status != "PASS":
            raise RuntimeError("Database integrity reconciliation failed")
        versions = await database_versions(connection)
        encoder = SentenceTransformerEncoder(config)
        retriever = DenseRetriever(connection, None, schema=arguments.schema)
        records_by_id = {record.chunk_id: record for record in records}
        gold_visibilities = {
            (record.tenant_id, record.document_id, record.document_version):
                record.access_scope
            for record in records
        }
        production_visibilities = tuple(arguments.visibility or ["public"])
        outcomes: list[dict[str, Any]] = []
        evaluated = no_answer = matched_5 = matched_10 = production_5 = production_10 = 0
        production_reachable = 0
        exact_matches = explained_ties = 0
        unexplained_differences: list[str] = []
        encoding_samples: list[float] = []
        sql_samples: list[float] = []
        total_samples: list[float] = []

        for question in questions:
            if question["expected_no_answer"].strip().lower() == "true":
                no_answer += 1
                outcomes.append({"question_id": question["question_id"], "status": "SKIPPED",
                                 "reason": "expected-no-answer excluded from recall"})
                continue
            encoding_started = asyncio.get_running_loop().time()
            query_vector = np.asarray(encoder.encode_query(question["query"]), dtype=np.float32)
            encoding_seconds = asyncio.get_running_loop().time() - encoding_started
            matched_context = _context(question, ("public", "internal"))
            comparison = await compare_exact_ranking(
                retriever,
                query_vector=query_vector,
                vectors=vectors,
                records=records,
                context=matched_context,
                profile=profile,
                top_k=arguments.top_k,
            )
            production_context = _context(question, production_visibilities)
            if production_context == matched_context:
                production_ids = comparison.database_chunk_ids
                production_database_seconds = comparison.database_seconds
                production_total_seconds = comparison.total_seconds
            else:
                production_response = await retriever.search(
                    DenseSearchRequest(
                        context=production_context, profile=profile,
                        query_vector=query_vector, top_k=arguments.top_k,
                    )
                )
                production_ids = tuple(
                    item.chunk_id for item in production_response.results
                )
                production_database_seconds = production_response.database_seconds
                production_total_seconds = production_response.total_seconds
            evaluated += 1
            exact_matches += int(comparison.exact_order_match)
            explained_ties += int(
                comparison.difference_classification == "explained_numerical_tie"
            )
            if comparison.difference_classification == "unexplained":
                unexplained_differences.append(question["question_id"])
            matched_5 += int(
                _is_hit(comparison.database_chunk_ids, records_by_id, question, 5)
            )
            matched_10 += int(
                _is_hit(comparison.database_chunk_ids, records_by_id, question, 10)
            )
            production_5 += int(
                _is_hit(production_ids, records_by_id, question, 5)
            )
            production_10 += int(
                _is_hit(production_ids, records_by_id, question, 10)
            )
            gold_reachable = _gold_is_reachable(
                question, gold_visibilities, production_visibilities
            )
            production_reachable += int(gold_reachable)
            encoding_samples.append(encoding_seconds)
            sql_samples.append(production_database_seconds)
            total_samples.append(encoding_seconds + production_total_seconds)
            outcomes.append({
                "question_id": question["question_id"],
                "status": (
                    "PASS" if comparison.difference_classification != "unexplained"
                    else "FAIL"
                ),
                "matched_scope": comparison.to_dict(),
                "production_chunk_ids": list(production_ids),
                "production_gold_reachable": gold_reachable,
            })
    finally:
        await connection.close()

    recall = lambda hits: hits / evaluated if evaluated else 0.0
    # Unreachable questions cannot hit, so all production hits are reachable
    # hits; only the denominator changes for policy-independent quality.
    reachable_recall = (
        lambda hits: hits / production_reachable if production_reachable else 0.0
    )
    report = {
        "status": "PASS" if not unexplained_differences else "FAIL",
        "execution": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_run": str(run_directory),
            "artifact_fingerprint": artifact_run_fingerprint(run_directory, artifact_report),
            "question_bank": str(arguments.question_bank),
            "question_bank_sha256": _question_bank_identity(arguments.question_bank),
            "split": arguments.split,
            "schema": arguments.schema,
            "similarity_metric": "cosine",
            "top_k": arguments.top_k,
            "tie_breaking": "ascending cosine distance, then chunk_id",
            "score_absolute_tolerance": 1e-6,
            "production_visibilities": list(production_visibilities),
        },
        "model": {
            "model_id": profile.model_id, "revision": profile.model_revision,
            "dimensions": plan.dimensions,
            "config_fingerprint": profile.config_fingerprint,
            "normalize_embeddings": profile.normalize_embeddings,
            "encoding_config_version": profile.encoding_config_version,
        },
        "database_integrity": integrity.to_dict(),
        "transaction_verification": {
            "status": "NOT VERIFIED",
            "reason": "Mutation checks run only in disposable integration-test schemas.",
        },
        "ranking_parity": {
            "queries_compared": evaluated, "exact_matches": exact_matches,
            "explained_tie_differences": explained_ties,
            "unexplained_differences": unexplained_differences,
        },
        "retrieval_quality": {
            "evaluated_questions": evaluated, "no_answer_questions": no_answer,
            "historical_phase5_reference": {
                "status": "NOT VERIFIED",
                "recall_at_5": 0.2075,
                "recall_at_10": 0.3396,
                "reported_question_count": 106,
                "note": "Prompt-supplied reference; not treated as a current run result.",
            },
            "matched_scope": {"hits_at_5": matched_5, "hits_at_10": matched_10,
                              "recall_at_5": recall(matched_5),
                              "recall_at_10": recall(matched_10)},
            "production_scope": {"hits_at_5": production_5,
                                 "hits_at_10": production_10,
                                 "recall_at_5": recall(production_5),
                                 "recall_at_10": recall(production_10),
                                 "reachable_questions": production_reachable,
                                 "unreachable_questions": evaluated - production_reachable,
                                 "reachable_recall_at_5": reachable_recall(production_5),
                                 "reachable_recall_at_10": reachable_recall(production_10)},
            "match_semantics": "document+version+section anchor",
        },
        "security_filtering": {
            "status": "NOT VERIFIED",
            "reason": (
                "Adversarial tenant/lifecycle/visibility fixtures are executed by "
                "tests.test_rag_dense_retrieval against disposable PostgreSQL."
            ),
        },
        "performance": {
            "model_loading_seconds": encoder.model_load_seconds,
            "query_encoding": latency_summary(encoding_samples),
            "sql_retrieval": latency_summary(sql_samples),
            "end_to_end": latency_summary(total_samples),
        },
        "environment": {**runtime_environment(), **versions},
        "limitations": [
            "Tenant IDs are trusted development/test scope, not user authentication.",
            "The synthetic question bank remains review-required.",
            "Recall measures retrieval of a section anchor, not answer accuracy.",
        ],
    }
    return report, outcomes


def _write_reports(
    report: dict[str, Any], outcomes: list[dict[str, Any]], results_root: Path
) -> Path:
    """Write a new immutable result directory without overwriting prior runs."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_directory = results_root / f"dense-eval-{timestamp}"
    run_directory.mkdir(parents=True, exist_ok=False)
    (run_directory / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [json.dumps(item, sort_keys=True) for item in outcomes]
    (run_directory / "retrieval_results.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    ranking = {
        "status": report["status"],
        "ranking_parity": report["ranking_parity"],
    }
    (run_directory / "ranking_comparison.json").write_text(
        json.dumps(ranking, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run_directory


def main(
    arguments: list[str] | None = None,
    output: TextIO = sys.stdout,
    error_output: TextIO = sys.stderr,
) -> int:
    """Run evaluation, write reports, and return a meaningful process status."""
    parsed = parse_args(arguments)
    try:
        report, outcomes = asyncio.run(_evaluate(parsed))
        result_directory = _write_reports(report, outcomes, parsed.results_root)
    except (ValueError, OSError, TimeoutError, asyncpg.PostgresError, RuntimeError) as error:
        print(f"Dense evaluation failed: {error}", file=error_output)
        print("FINAL STATUS: FAILED", file=output)
        return 1
    quality = report["retrieval_quality"]
    print("DENSE RETRIEVAL EVALUATION", file=output)
    print(f"Integrity: {report['database_integrity']['status']}", file=output)
    print(f"Ranking parity: {report['ranking_parity']['exact_matches']}/"
          f"{report['ranking_parity']['queries_compared']}", file=output)
    print(f"Matched Recall@5: {quality['matched_scope']['recall_at_5']:.4f}", file=output)
    print(f"Matched Recall@10: {quality['matched_scope']['recall_at_10']:.4f}", file=output)
    production = quality["production_scope"]
    print(f"Production reachable questions: {production['reachable_questions']}/"
          f"{quality['evaluated_questions']}", file=output)
    print(f"Production reachable Recall@5: {production['reachable_recall_at_5']:.4f}",
          file=output)
    print(f"Production reachable Recall@10: {production['reachable_recall_at_10']:.4f}",
          file=output)
    print(f"Report directory: {result_directory}", file=output)
    print(f"FINAL STATUS: {report['status']}", file=output)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
