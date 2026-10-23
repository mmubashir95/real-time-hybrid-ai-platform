#!/usr/bin/env python3
"""Compare frozen hybrid candidates with Qwen3 reranking on one label split."""

from __future__ import annotations

import argparse
import asyncio
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
from app.rag.persistence.lifecycle import EmbeddingProfile  # noqa: E402
from app.rag.reranking import Qwen3Reranker, RerankerConfig, RerankingService  # noqa: E402
from app.rag.retrieval import (  # noqa: E402
    DenseRetriever,
    KeywordRetriever,
    RetrievalService,
    SearchFilters,
    SearchMode,
    SearchRequest,
)
from app.rag.retrieval.metrics import (  # noqa: E402
    EvaluationExample,
    latency_percentiles,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from scripts.evaluate_rag_retrieval import (  # noqa: E402
    DEFAULT_QUESTIONS,
    _load_questions,
    _question_context,
    _source_key,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "data/rag/evaluation/reranking_report.json"


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse frozen dataset, retrieval, and reranker benchmark settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--split", default="heldout_test")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config-fingerprint", required=True)
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", ""))
    parser.add_argument("--schema", default="public")
    parser.add_argument("--visibility", action="append", choices=("public", "internal"))
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--reranker-device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--reranker-precision", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-unreviewed-labels", action="store_true")
    return parser.parse_args(arguments)


def _metric_summary(
    examples: tuple[EvaluationExample, ...], latencies: tuple[float, ...]
) -> dict[str, float]:
    """Calculate the Phase 8 quality and operational metric set."""
    p50, p95 = latency_percentiles(latencies)
    return {
        "recall_at_5": recall_at_k(examples, 5),
        "mrr_at_10": mrr_at_k(examples, 10),
        "ndcg_at_5_binary": ndcg_at_k(examples, 5),
        "ndcg_at_10_binary": ndcg_at_k(examples, 10),
        "p50_latency_seconds": p50,
        "p95_latency_seconds": p95,
    }


async def _evaluate(arguments: argparse.Namespace) -> dict[str, object]:
    """Retrieve once per query and compare two rankings of identical candidates."""
    rows = _load_questions(arguments.questions, arguments.split, arguments.allow_unreviewed_labels)
    if arguments.top_k > arguments.candidate_limit:
        raise ValueError("--top-k must not exceed --candidate-limit")
    if not arguments.database_url:
        raise ValueError("DATABASE_URL or --database-url is required")

    embedding = EmbeddingConfig(device=arguments.embedding_device)
    profile = EmbeddingProfile(
        embedding.model_id,
        embedding.model_revision,
        arguments.config_fingerprint,
        embedding.normalize_embeddings,
        embedding.encoding_config_version,
        embedding.query_instruction,
    )
    reranker_config = RerankerConfig(
        batch_size=arguments.batch_size,
        max_length=arguments.max_length,
        device=arguments.reranker_device,
        precision=arguments.reranker_precision,
        local_files_only=arguments.local_files_only,
    )
    url = make_url(arguments.database_url)
    dsn = url.set(drivername="postgresql").render_as_string(hide_password=False)
    connection = await asyncpg.connect(dsn, timeout=10)
    try:
        retrieval = RetrievalService(
            KeywordRetriever(connection, schema=arguments.schema),
            DenseRetriever(connection, SentenceTransformerEncoder(embedding), schema=arguments.schema),
        )
        qwen_scorer = Qwen3Reranker(reranker_config)
        reranker = RerankingService(qwen_scorer)
        baseline_examples: list[EvaluationExample] = []
        reranked_examples: list[EvaluationExample] = []
        candidate_examples: list[EvaluationExample] = []
        retrieval_latencies: list[float] = []
        combined_latencies: list[float] = []
        frozen_candidates: list[dict[str, object]] = []
        truncation_count = 0
        unauthorized_passages = 0

        for row in rows:
            context = _question_context(row, tuple(arguments.visibility or ("public", "internal")))
            response = await retrieval.search(SearchRequest(
                row["query"], SearchMode.HYBRID, SearchFilters(),
                arguments.candidate_limit, context, profile,
                arguments.candidate_limit, arguments.candidate_limit,
            ))
            # This is the single frozen candidate set used for both rankings.
            reranked = await asyncio.to_thread(
                reranker.rerank, row["query"], response.results, arguments.top_k
            )
            no_answer = row["expected_no_answer"].strip().lower() == "true"
            gold = frozenset() if no_answer else frozenset((
                (row["tenant_id"], row["gold_document_id"], row["gold_version"], row["gold_section_key"]),
            ))
            baseline_sources = tuple(_source_key(item) for item in response.results)
            reranked_sources = tuple(_source_key(item.passage) for item in reranked.results)
            candidate_examples.append(EvaluationExample(baseline_sources, gold))
            baseline_examples.append(EvaluationExample(baseline_sources[:arguments.top_k], gold))
            reranked_examples.append(EvaluationExample(reranked_sources, gold))
            retrieval_latencies.append(response.diagnostics.total_seconds)
            combined_latencies.append(response.diagnostics.total_seconds + reranked.diagnostics.inference_seconds)
            truncation_count += reranked.diagnostics.truncated_inputs
            unauthorized_passages += sum(item.tenant_id != row["tenant_id"] for item in response.results)
            frozen_candidates.append({
                "question_id": row["question_id"],
                "chunk_ids": [item.chunk_id for item in response.results],
                "hybrid_ranks": [item.final_rank for item in response.results],
                "reranked_chunk_ids": [item.passage.chunk_id for item in reranked.results],
            })
    finally:
        await connection.close()

    candidate_tuple = tuple(candidate_examples)
    return {
        "status": "EXPLORATORY_UNREVIEWED" if arguments.allow_unreviewed_labels else "REVIEWED",
        "configuration": {
            "evaluation_dataset": str(arguments.questions),
            "evaluation_dataset_sha256": hashlib.sha256(arguments.questions.read_bytes()).hexdigest(),
            "split": arguments.split,
            "questions": len(rows),
            "embedding": asdict(embedding),
            "embedding_config_fingerprint": arguments.config_fingerprint,
            "bm25": {"k1": 1.5, "b": 0.75},
            "rrf_constant": 60,
            "candidate_limit": arguments.candidate_limit,
            "top_k": arguments.top_k,
            "reranker": asdict(reranker_config),
            "scoring_implementation": "Qwen yes-token logit minus no-token logit",
        },
        "metrics": {
            "hybrid_only": _metric_summary(tuple(baseline_examples), tuple(retrieval_latencies)),
            "hybrid_plus_qwen3": _metric_summary(tuple(reranked_examples), tuple(combined_latencies)),
            "candidate_recall_at_20": recall_at_k(candidate_tuple, min(20, arguments.candidate_limit)),
            "unauthorized_passages": unauthorized_passages,
            "truncated_inputs": truncation_count,
            "peak_memory": "NOT MEASURED",
            "model_load_latency_seconds": qwen_scorer.model_load_seconds,
        },
        "frozen_candidate_sets": frozen_candidates,
    }


def main(arguments: list[str] | None = None) -> int:
    """Write a reproducible report or return a controlled failure code."""
    parsed = parse_args(arguments)
    try:
        report = asyncio.run(_evaluate(parsed))
        parsed.output.parent.mkdir(parents=True, exist_ok=True)
        parsed.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (ValueError, TypeError, OSError, TimeoutError, asyncpg.PostgresError, RuntimeError) as error:
        print(f"Reranking evaluation failed: {error}", file=sys.stderr)
        return 1
    print(f"Reranking evaluation report written to {parsed.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
