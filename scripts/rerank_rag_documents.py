#!/usr/bin/env python3
"""Inspect authorized Phase 7 hybrid candidates and Phase 8 Qwen3 reranking."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import sys
from typing import TextIO

import asyncpg
from sqlalchemy.engine import make_url


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.embeddings.encoder import SentenceTransformerEncoder  # noqa: E402
from app.rag.embeddings.models import EmbeddingConfig  # noqa: E402
from app.rag.persistence.lifecycle import EmbeddingProfile  # noqa: E402
from app.rag.reranking import (  # noqa: E402
    Qwen3Reranker,
    RerankerConfig,
    RerankerInferenceError,
    RerankingService,
    RetrievalRerankingService,
)
from app.rag.retrieval import (  # noqa: E402
    DenseRetriever,
    KeywordRetriever,
    RetrievalService,
    SearchContextOrigin,
    SearchFilters,
    SearchMode,
    SearchRequest,
    TrustedSearchContext,
)


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse trusted development scope and bounded inference configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--tenant", required=True, help="Trusted local/evaluation tenant scope.")
    parser.add_argument("--visibility", action="append", choices=("public", "internal"))
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--config-fingerprint", required=True)
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", ""))
    parser.add_argument("--schema", default="public")
    parser.add_argument("--document-id")
    parser.add_argument("--section")
    parser.add_argument("--effective-date-from")
    parser.add_argument("--effective-date-to")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--reranker-device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--reranker-precision", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-rerank", action="store_true", help="Display the hybrid baseline only.")
    return parser.parse_args(arguments)


def _database_dsn(value: str) -> str:
    """Return an asyncpg DSN after validating the database scheme."""
    if not value:
        raise ValueError("DATABASE_URL or --database-url is required")
    url = make_url(value)
    if url.drivername not in {"postgresql", "postgresql+asyncpg"}:
        raise ValueError("DATABASE_URL must use PostgreSQL")
    return url.set(drivername="postgresql").render_as_string(hide_password=False)


def _build_request(arguments: argparse.Namespace) -> tuple[SearchRequest, EmbeddingConfig]:
    """Build a hybrid request whose tenant scope is trusted by this local CLI."""
    embedding = EmbeddingConfig(device=arguments.embedding_device)
    profile = EmbeddingProfile(
        embedding.model_id,
        embedding.model_revision,
        arguments.config_fingerprint,
        embedding.normalize_embeddings,
        embedding.encoding_config_version,
        embedding.query_instruction,
    )
    context = TrustedSearchContext(
        arguments.tenant,
        tuple(arguments.visibility or ("public",)),
        SearchContextOrigin.INTERNAL,
    )
    request = SearchRequest(
        arguments.query,
        SearchMode.HYBRID,
        SearchFilters(
            document_id=arguments.document_id,
            section=arguments.section,
            effective_date_from=arguments.effective_date_from,
            effective_date_to=arguments.effective_date_to,
        ),
        arguments.candidate_k,
        context,
        profile,
        arguments.candidate_k,
        arguments.candidate_k,
    )
    if arguments.top_k > arguments.candidate_k:
        raise ValueError("--top-k must not exceed --candidate-k")
    return request, embedding


def _print_retrieval(response, output: TextIO) -> None:
    """Print the original hybrid order without exposing any extra content."""
    print("\nRETRIEVAL", file=output)
    print(f"  Hybrid retrieval latency: {response.diagnostics.total_seconds:.6f}s", file=output)
    print(f"  Candidate count: {len(response.results)}", file=output)
    for passage in response.results:
        print(
            f"  {passage.final_rank}. {passage.chunk_id} "
            f"fusion={passage.fusion_score}",
            file=output,
        )


def _print_reranking(response, output: TextIO) -> None:
    """Print reranker measurements and source-traceable final results."""
    diagnostics = response.diagnostics
    print("\nRERANKING", file=output)
    print(f"  Model load latency: {diagnostics.model_load_seconds:.6f}s", file=output)
    print(f"  Model inference latency: {diagnostics.inference_seconds:.6f}s", file=output)
    print(f"  Batch size: {diagnostics.batch_size}", file=output)
    print(f"  Truncated inputs: {diagnostics.truncated_inputs}", file=output)
    print("\nRESULTS", file=output)
    for result in response.results:
        passage = result.passage
        preview = " ".join(passage.text.split())[:240]
        print(f"\n{result.rerank_rank}. {passage.chunk_id}", file=output)
        print(f"   Document: {passage.document_id} v{passage.document_version}", file=output)
        print(f"   Section: {' > '.join(passage.section) or '(root)'}", file=output)
        print(f"   Pages: {passage.start_page or '-'}-{passage.end_page or '-'}", file=output)
        print(f"   Original hybrid rank: {result.retrieval_rank}", file=output)
        print(f"   Reranker score: {result.rerank_score}", file=output)
        print(f"   Input tokens/truncated: {result.input_tokens}/{result.passage_truncated}", file=output)
        print(f"   Passage preview: {preview}", file=output)


async def _run(arguments: argparse.Namespace, output: TextIO) -> None:
    """Run one authorized candidate retrieval and optional pretrained rerank."""
    request, embedding = _build_request(arguments)
    connection = await asyncpg.connect(_database_dsn(arguments.database_url), timeout=10)
    retrieval = RetrievalService(
        KeywordRetriever(connection, schema=arguments.schema),
        DenseRetriever(connection, SentenceTransformerEncoder(embedding), schema=arguments.schema),
    )
    try:
        if arguments.no_rerank:
            retrieval_response = await retrieval.search(request)
            rerank_response = None
        else:
            config = RerankerConfig(
                batch_size=arguments.batch_size,
                max_length=arguments.max_length,
                device=arguments.reranker_device,
                precision=arguments.reranker_precision,
                local_files_only=arguments.local_files_only,
            )
            retrieval_response, rerank_response = await RetrievalRerankingService(
                retrieval,
                RerankingService(Qwen3Reranker(config)),
            ).search(request, top_k=arguments.top_k)
    finally:
        await connection.close()

    print("=" * 64, file=output)
    print("PHASE 8 — PRETRAINED RERANKING", file=output)
    print("=" * 64, file=output)
    print(f"Query: {request.query}", file=output)
    print(f"Authorized tenant: {request.context.tenant_id}", file=output)
    print(f"Embedding model: {embedding.model_id}@{embedding.model_revision}", file=output)
    print(f"Candidate count requested: {request.k}", file=output)
    print(f"Final top K: {arguments.top_k}", file=output)
    _print_retrieval(retrieval_response, output)
    if rerank_response is not None:
        first = rerank_response.results[0] if rerank_response.results else None
        print(f"Reranker model: {first.reranker_model_id if first else RerankerConfig().model_id}", file=output)
        print(f"Reranker revision: {first.reranker_model_revision if first else RerankerConfig().model_revision}", file=output)
        _print_reranking(rerank_response, output)
    print("\nFINAL STATUS: PASSED", file=output)


def main(
    arguments: list[str] | None = None,
    output: TextIO = sys.stdout,
    error_output: TextIO = sys.stderr,
) -> int:
    """Run the CLI and convert expected failures to a nonzero exit code."""
    parsed = parse_args(arguments)
    try:
        asyncio.run(_run(parsed, output))
    except (
        ValueError,
        TypeError,
        OSError,
        TimeoutError,
        asyncpg.PostgresError,
        RerankerInferenceError,
        RuntimeError,
    ) as error:
        print(f"Reranking failed: {error}", file=error_output)
        print("FINAL STATUS: FAILED", file=output)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
