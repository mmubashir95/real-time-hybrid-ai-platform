#!/usr/bin/env python3
"""Run internal exact dense search against an authorized development scope."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import sys
from typing import TextIO

import asyncpg
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.embeddings.encoder import SentenceTransformerEncoder  # noqa: E402
from app.rag.embeddings.models import EmbeddingConfig  # noqa: E402
from app.rag.embeddings.service import load_evaluation_question  # noqa: E402
from app.rag.persistence.lifecycle import EmbeddingProfile  # noqa: E402
from app.rag.retrieval import (  # noqa: E402
    DenseRetriever,
    DenseSearchRequest,
    HistoricalVersionScope,
    TrustedSearchContext,
)


DEFAULT_QUESTION_BANK = (
    PROJECT_ROOT
    / "data"
    / "rag"
    / "evaluation"
    / "question_bank_SYNTHETIC_REVIEW_REQUIRED.csv"
)


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse internal scope, model profile, and query selection arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    query = parser.add_mutually_exclusive_group(required=True)
    query.add_argument("--question")
    query.add_argument("--question-id")
    parser.add_argument("--question-bank", type=Path, default=DEFAULT_QUESTION_BANK)
    parser.add_argument("--tenant", help="Trusted development/test tenant scope.")
    parser.add_argument(
        "--visibility",
        action="append",
        choices=("public", "internal"),
        default=None,
        help="Permitted visibility; repeat to permit more than one.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--config-fingerprint", required=True)
    parser.add_argument("--historical-document-id")
    parser.add_argument("--historical-version")
    parser.add_argument("--schema", default="public")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", ""))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(arguments)


def _database_dsn(database_url: str) -> str:
    """Validate a PostgreSQL URL and return an asyncpg-compatible DSN."""
    if not database_url:
        raise ValueError("DATABASE_URL or --database-url is required")
    try:
        url = make_url(database_url)
    except ArgumentError as error:
        raise ValueError("DATABASE_URL is not valid") from error
    if url.drivername not in {"postgresql", "postgresql+asyncpg"}:
        raise ValueError("DATABASE_URL must use PostgreSQL with asyncpg")
    return url.set(drivername="postgresql").render_as_string(hide_password=False)


def _resolve_query(arguments: argparse.Namespace) -> tuple[str, str]:
    """Return question and tenant, using a labeled row when one was selected."""
    if arguments.question_id:
        row = load_evaluation_question(arguments.question_bank, arguments.question_id)
        if arguments.tenant and arguments.tenant != row["tenant_id"]:
            raise ValueError("--tenant conflicts with the evaluation question tenant")
        return row["query"], row["tenant_id"]
    if not arguments.tenant:
        raise ValueError("--tenant is required with --question")
    return arguments.question, arguments.tenant


def _historical_scope(arguments: argparse.Namespace) -> HistoricalVersionScope | None:
    """Require both identifiers for an explicitly authorized historical query."""
    provided = (
        arguments.historical_document_id is not None,
        arguments.historical_version is not None,
    )
    if provided == (False, False):
        return None
    if provided != (True, True):
        raise ValueError(
            "Historical retrieval requires --historical-document-id and --historical-version"
        )
    return HistoricalVersionScope(
        arguments.historical_document_id, arguments.historical_version
    )


async def _run(arguments: argparse.Namespace, output: TextIO) -> None:
    """Load one model instance, execute search, and print measured results."""
    question, tenant_id = _resolve_query(arguments)
    config = EmbeddingConfig(device=arguments.device)
    profile = EmbeddingProfile(
        model_id=config.model_id,
        model_revision=config.model_revision,
        config_fingerprint=arguments.config_fingerprint,
        normalize_embeddings=config.normalize_embeddings,
        encoding_config_version=config.encoding_config_version,
        query_instruction=config.query_instruction,
    )
    context = TrustedSearchContext(
        tenant_id=tenant_id,
        permitted_visibilities=tuple(arguments.visibility or ["public"]),
        historical=_historical_scope(arguments),
    )
    request = DenseSearchRequest(
        question=question,
        context=context,
        profile=profile,
        top_k=arguments.top_k,
    )
    encoder = SentenceTransformerEncoder(config)
    connection = await asyncpg.connect(_database_dsn(arguments.database_url), timeout=10)
    try:
        response = await DenseRetriever(
            connection, encoder, schema=arguments.schema
        ).search(request)
    finally:
        await connection.close()

    print("DENSE SEMANTIC SEARCH\n", file=output)
    print(f"Question: {question}", file=output)
    print(f"Tenant: {tenant_id}", file=output)
    print(f"Visibility: {', '.join(context.permitted_visibilities)}", file=output)
    version_scope = (
        f"{context.historical.document_id}/{context.historical.version}"
        if context.historical
        else "active"
    )
    print(f"Document versions: {version_scope}", file=output)
    print(f"Top-K: {arguments.top_k}", file=output)
    print(f"Model: {profile.model_id}@{profile.model_revision}", file=output)
    print(f"Candidates in scope: {response.candidate_count}\n", file=output)
    print("RESULTS", file=output)
    for rank, result in enumerate(response.results, start=1):
        section = " > ".join(result.heading_path) or "(root)"
        print(f"\n{rank}. Chunk ID: {result.chunk_id}", file=output)
        print(f"   Document: {result.document_title} ({result.document_id})", file=output)
        print(f"   Version: {result.document_version}", file=output)
        print(f"   Section: {section}", file=output)
        print(f"   Similarity: {result.similarity:.6f}", file=output)
        print(f"   Source: {result.source_path}", file=output)
    print("\nPERFORMANCE", file=output)
    print(f"  Model loading: {encoder.model_load_seconds:.6f}s", file=output)
    print(f"  Query encoding: {response.query_encoding_seconds:.6f}s", file=output)
    print(f"  Database retrieval: {response.database_seconds:.6f}s", file=output)
    print(f"  Total retrieval: {response.total_seconds:.6f}s", file=output)
    print("\nRESULT STATUS: SUCCESS", file=output)


def main(
    arguments: list[str] | None = None,
    output: TextIO = sys.stdout,
    error_output: TextIO = sys.stderr,
) -> int:
    """Run the internal CLI and convert expected failures to a nonzero status."""
    parsed = parse_args(arguments)
    try:
        asyncio.run(_run(parsed, output))
    except (ValueError, OSError, TimeoutError, asyncpg.PostgresError, RuntimeError) as error:
        print(f"Dense search failed: {error}", file=error_output)
        print("RESULT STATUS: FAILED", file=output)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
