#!/usr/bin/env python3
"""Inspect authorized keyword, semantic, or hybrid RAG retrieval results."""

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
from app.rag.retrieval import (  # noqa: E402
    DenseRetriever,
    HistoricalVersionScope,
    KeywordRetriever,
    RetrievalService,
    SearchContextOrigin,
    SearchFilters,
    SearchMode,
    SearchRequest,
    TrustedSearchContext,
)


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse trusted development scope and bounded retrieval options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode", choices=tuple(mode.value for mode in SearchMode), default="hybrid")
    parser.add_argument("--tenant", required=True, help="Trusted local/evaluation tenant scope.")
    parser.add_argument("--visibility", action="append", choices=("public", "internal"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--keyword-candidates", type=int, default=20)
    parser.add_argument("--semantic-candidates", type=int, default=20)
    parser.add_argument(
        "--config-fingerprint",
        help="Required for semantic and hybrid modes; ignored by keyword mode.",
    )
    parser.add_argument("--document-id")
    parser.add_argument("--section")
    parser.add_argument("--effective-date-from")
    parser.add_argument("--effective-date-to")
    parser.add_argument("--historical-document-id")
    parser.add_argument("--historical-version")
    parser.add_argument("--authorize-historical", action="store_true")
    parser.add_argument("--schema", default="public")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", ""))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(arguments)


def _database_dsn(value: str) -> str:
    """Return an asyncpg DSN after validating the database scheme."""
    if not value:
        raise ValueError("DATABASE_URL or --database-url is required")
    url = make_url(value)
    if url.drivername not in {"postgresql", "postgresql+asyncpg"}:
        raise ValueError("DATABASE_URL must use PostgreSQL")
    return url.set(drivername="postgresql").render_as_string(hide_password=False)


def _historical_scope(arguments: argparse.Namespace) -> HistoricalVersionScope | None:
    """Require an explicit acknowledgement and exact archived version scope."""
    identifiers = (arguments.historical_document_id, arguments.historical_version)
    if identifiers == (None, None):
        if arguments.authorize_historical:
            raise ValueError("Historical authorization requires exact document and version")
        return None
    if None in identifiers or not arguments.authorize_historical:
        raise ValueError("Historical retrieval requires both identifiers and --authorize-historical")
    return HistoricalVersionScope(*identifiers)


async def _run(arguments: argparse.Namespace, output: TextIO) -> None:
    """Construct the trusted context, run retrieval, and print safe diagnostics."""
    config = EmbeddingConfig(device=arguments.device)
    mode = SearchMode(arguments.mode)
    if mode is not SearchMode.KEYWORD and not arguments.config_fingerprint:
        raise ValueError("--config-fingerprint is required for semantic and hybrid modes")
    profile = (
        EmbeddingProfile(
            config.model_id, config.model_revision, arguments.config_fingerprint,
            config.normalize_embeddings, config.encoding_config_version,
            config.query_instruction,
        )
        if arguments.config_fingerprint else None
    )
    historical = _historical_scope(arguments)
    context = TrustedSearchContext(
        arguments.tenant, tuple(arguments.visibility or ("public",)),
        SearchContextOrigin.INTERNAL, historical, historical is not None,
    )
    filters = SearchFilters(
        document_id=arguments.document_id, section=arguments.section,
        effective_date_from=arguments.effective_date_from,
        effective_date_to=arguments.effective_date_to,
    )
    request = SearchRequest(
        arguments.query, mode, filters, arguments.k, context, profile,
        arguments.keyword_candidates, arguments.semantic_candidates,
    )
    # Keyword-only mode avoids loading the transformer entirely.
    encoder = None if request.mode is SearchMode.KEYWORD else SentenceTransformerEncoder(config)
    connection = await asyncpg.connect(_database_dsn(arguments.database_url), timeout=10)
    try:
        response = await RetrievalService(
            KeywordRetriever(connection, schema=arguments.schema),
            DenseRetriever(connection, encoder, schema=arguments.schema),
        ).search(request)
    finally:
        await connection.close()

    print("=" * 56, file=output)
    print("RAG RETRIEVAL", file=output)
    print("=" * 56, file=output)
    print(f"Mode: {request.mode.value}\nQuery: {request.query}\nTenant: {context.tenant_id}\nTop K: {request.k}", file=output)
    diagnostics = response.diagnostics
    print("\nRETRIEVAL", file=output)
    print(f"  BM25 candidates: {diagnostics.keyword_candidates}", file=output)
    print(f"  Semantic candidates: {diagnostics.semantic_candidates}", file=output)
    print(f"  Fused candidates: {diagnostics.fused_candidates}", file=output)
    print(f"  Returned passages: {diagnostics.returned_count}", file=output)
    print("\nRESULTS", file=output)
    for passage in response.results:
        preview = " ".join(passage.text.split())[:240]
        print(f"\n{passage.final_rank}. {passage.chunk_id}", file=output)
        print(f"   Document: {passage.document_id} v{passage.document_version}", file=output)
        print(f"   Section: {' > '.join(passage.section) or '(root)'}", file=output)
        print(f"   Pages: {passage.start_page or '-'}-{passage.end_page or '-'}", file=output)
        print(f"   BM25 rank/score: {passage.bm25_rank}/{passage.bm25_score}", file=output)
        print(f"   Semantic rank/score: {passage.semantic_rank}/{passage.semantic_score}", file=output)
        print(f"   Fusion score: {passage.fusion_score}", file=output)
        print(f"   Preview: {preview}", file=output)
    print("\nPERFORMANCE", file=output)
    print(f"  Keyword: {diagnostics.keyword_seconds:.6f}s", file=output)
    print(f"  Query embedding: {diagnostics.query_embedding_seconds:.6f}s", file=output)
    print(f"  Vector search: {diagnostics.vector_search_seconds:.6f}s", file=output)
    print(f"  Fusion: {diagnostics.fusion_seconds:.6f}s", file=output)
    print(f"  Total: {diagnostics.total_seconds:.6f}s", file=output)
    print("\nFINAL STATUS: PASSED", file=output)


def main(
    arguments: list[str] | None = None,
    output: TextIO = sys.stdout,
    error_output: TextIO = sys.stderr,
) -> int:
    """Run the CLI and convert expected operational errors to exit code one."""
    parsed = parse_args(arguments)
    try:
        asyncio.run(_run(parsed, output))
    except (ValueError, TypeError, OSError, TimeoutError, asyncpg.PostgresError, RuntimeError) as error:
        print(f"Retrieval failed: {error}", file=error_output)
        print("FINAL STATUS: FAILED", file=output)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
