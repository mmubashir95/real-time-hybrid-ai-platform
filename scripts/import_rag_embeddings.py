#!/usr/bin/env python3
"""Validate and transactionally import published RAG embeddings into PostgreSQL."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import TextIO

import asyncpg
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.persistence import (  # noqa: E402
    ImportPlan,
    RagImportConflict,
    import_rag_embeddings,
    load_import_plan,
)


DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "rag" / "manifest.csv"
DEFAULT_RAW_DIRECTORY = PROJECT_ROOT / "data" / "rag" / "raw"
DEFAULT_CHUNKS = PROJECT_ROOT / "data" / "rag" / "processed" / "chunks.jsonl"
DEFAULT_CHUNKING_REPORT = (
    PROJECT_ROOT / "data" / "rag" / "processed" / "chunking_report.json"
)
DEFAULT_EMBEDDING_ROOT = (
    PROJECT_ROOT / "data" / "rag" / "processed" / "embeddings"
)


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse source, database, schema, and validation-only options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-directory", type=Path, default=DEFAULT_RAW_DIRECTORY)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument(
        "--chunking-report", type=Path, default=DEFAULT_CHUNKING_REPORT
    )
    parser.add_argument(
        "--embedding-root", type=Path, default=DEFAULT_EMBEDDING_ROOT
    )
    parser.add_argument(
        "--run-directory",
        type=Path,
        help="Use an explicit immutable run instead of embedding-root/latest.json.",
    )
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", ""))
    parser.add_argument("--schema", default="public")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate and map every artifact without opening a database connection.",
    )
    return parser.parse_args(arguments)


def _database_urls(database_url: str) -> tuple[str, str]:
    """Return an asyncpg DSN and a password-redacted display URL."""
    try:
        url = make_url(database_url)
    except ArgumentError as error:
        raise ValueError("DATABASE_URL is not valid") from error
    if url.drivername not in {"postgresql", "postgresql+asyncpg"}:
        raise ValueError("DATABASE_URL must use PostgreSQL with asyncpg")
    dsn = url.set(drivername="postgresql").render_as_string(hide_password=False)
    display = url.set(drivername="postgresql").render_as_string(hide_password=True)
    return dsn, display


def _print_input(plan: ImportPlan, validation_seconds: float, output: TextIO) -> None:
    """Print source scope without exposing document content or vector values."""
    print("\nINPUT", file=output)
    print(f"  Source run: {plan.run_directory}", file=output)
    print(f"  Documents: {len(plan.documents)}", file=output)
    print(f"  Versions: {len(plan.versions)}", file=output)
    print(f"  Chunks: {len(plan.chunks)}", file=output)
    print(f"  Embeddings: {len(plan.embeddings)}", file=output)
    print(f"  Model ID: {plan.model_id}", file=output)
    print(f"  Revision: {plan.model_revision}", file=output)
    print(f"  Dimensions: {plan.dimensions}", file=output)
    print(f"  Artifact validation seconds: {validation_seconds:.3f}", file=output)


async def _execute_import(
    arguments: argparse.Namespace,
    plan: ImportPlan,
    output: TextIO,
) -> float:
    """Connect to the selected database, run one transaction, and print counts."""
    if not arguments.database_url:
        raise ValueError("DATABASE_URL or --database-url is required for import")
    dsn, display_url = _database_urls(arguments.database_url)
    print(f"\nTARGET\n  Database: {display_url}\n  Schema: {arguments.schema}", file=output)
    connection = await asyncpg.connect(dsn, timeout=10)
    try:
        result = await import_rag_embeddings(
            connection, plan, schema=arguments.schema
        )
    finally:
        await connection.close()
    counts = result.counts
    print("\nRESULT", file=output)
    for label, value in (
        ("Documents inserted", counts.documents_inserted),
        ("Documents reused", counts.documents_reused),
        ("Versions inserted", counts.versions_inserted),
        ("Versions reused", counts.versions_reused),
        ("Chunks inserted", counts.chunks_inserted),
        ("Chunks reused", counts.chunks_reused),
        ("Embeddings inserted", counts.embeddings_inserted),
        ("Embeddings reused", counts.embeddings_reused),
    ):
        print(f"  {label}: {value}", file=output)
    print("  Conflicts: 0", file=output)
    print("  Transaction committed: yes", file=output)
    print(f"  Database import seconds: {result.database_seconds:.3f}", file=output)
    return result.database_seconds


def main(
    arguments: list[str] | None = None,
    output: TextIO = sys.stdout,
    error_output: TextIO = sys.stderr,
) -> int:
    """Validate input, optionally import it, and return a process exit status."""
    parsed = parse_args(arguments)
    started_at = perf_counter()
    print("=" * 56, file=output)
    print("                RAG EMBEDDING IMPORT", file=output)
    print("=" * 56, file=output)
    try:
        validation_started_at = perf_counter()
        plan = load_import_plan(
            manifest_path=parsed.manifest,
            raw_directory=parsed.raw_directory,
            chunks_path=parsed.chunks,
            chunking_report_path=parsed.chunking_report,
            embedding_root=parsed.embedding_root,
            run_directory=parsed.run_directory,
        )
        validation_seconds = perf_counter() - validation_started_at
        _print_input(plan, validation_seconds, output)
        if parsed.check_only:
            print("\n  Transaction committed: no (validation-only mode)", file=output)
        else:
            asyncio.run(_execute_import(parsed, plan, output))
    except (
        RagImportConflict,
        ValueError,
        FileNotFoundError,
        OSError,
        TimeoutError,
        asyncpg.PostgresError,
    ) as error:
        print(f"Import failed: {error}", file=error_output)
        print(f"\nTOTAL SECONDS: {perf_counter() - started_at:.3f}", file=output)
        print("FINAL STATUS: FAILED", file=output)
        return 1
    print(f"\nTOTAL SECONDS: {perf_counter() - started_at:.3f}", file=output)
    print("FINAL STATUS: PASSED", file=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
