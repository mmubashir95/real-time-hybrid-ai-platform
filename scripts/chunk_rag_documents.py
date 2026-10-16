#!/usr/bin/env python3
"""Run discovery, parsing, normalization, and deterministic RAG chunking.

Successful runs atomically replace the generated JSONL and validation report.
The command stops at text chunks; it performs no embedding, indexing, database,
or retrieval work.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, TextIO

# Direct script execution places scripts/ rather than the repository root on
# sys.path. Resolve the project root so imports and defaults are cwd-independent.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.ingestion.chunking import (  # noqa: E402
    ChunkingConfig,
    chunk_documents,
    count_tokens,
)
from app.rag.ingestion.discovery import validate_manifest  # noqa: E402
from app.rag.ingestion.document_parser import parse_documents  # noqa: E402
from app.rag.ingestion.models import DocumentChunk  # noqa: E402
from app.rag.ingestion.normalization import normalize_documents  # noqa: E402
from scripts.discover_rag_documents import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    DEFAULT_RAW_DIRECTORY,
)


DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "rag" / "processed"
CHUNKS_FILENAME = "chunks.jsonl"
REPORT_FILENAME = "chunking_report.json"


def serialize_chunks(chunks: tuple[DocumentChunk, ...]) -> bytes:
    """Serialize chunks as stable, compact UTF-8 JSON Lines."""
    lines = [
        json.dumps(
            chunk.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for chunk in chunks
    ]
    text = "\n".join(lines) + ("\n" if lines else "")
    return text.encode("utf-8")


def _error_record(error: object, stage: str) -> dict[str, str]:
    """Convert an ingestion error model into one stable report record."""
    return {
        "document_id": str(getattr(error, "document_id", None) or "unknown"),
        "tenant_id": str(getattr(error, "tenant_id", None) or "unknown"),
        "document_version": str(getattr(error, "version", None) or "unknown"),
        "source_path": str(getattr(error, "source_path", "unknown")),
        "processing_stage": stage,
        "error_type": str(getattr(error, "error_type", type(error).__name__)),
        "error_message": str(
            getattr(error, "error_message", getattr(error, "message", error))
        ),
    }


def build_chunking_report(
    *,
    documents_discovered: int,
    documents_parsed: int,
    documents_normalized: int,
    chunks: tuple[DocumentChunk, ...],
    processed_documents: int,
    documents_with_chunks: int,
    zero_chunk_documents: tuple[str, ...],
    normalized_blocks_processed: int,
    empty_blocks_skipped: int,
    oversized_paragraphs: int,
    token_fallback_splits: int,
    duplicated_overlap_tokens: int,
    failures: list[dict[str, str]],
    config: ChunkingConfig,
) -> dict[str, Any]:
    """Build the deterministic full-corpus validation report."""
    token_counts = [chunk.token_count for chunk in chunks]
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    document_counts = Counter(
        f"{chunk.tenant_id}/{chunk.document_id}/{chunk.document_version}"
        for chunk in chunks
    )
    tenant_counts = Counter(chunk.tenant_id for chunk in chunks)
    format_counts = Counter(chunk.source_type for chunk in chunks)
    duplicate_chunk_ids = len(chunk_ids) - len(set(chunk_ids))
    chunks_above_maximum = sum(
        chunk.token_count > config.max_tokens for chunk in chunks
    )
    inaccurate_token_counts = sum(
        chunk.token_count != count_tokens(chunk.text) for chunk in chunks
    )
    empty_chunks = sum(not chunk.text.strip() for chunk in chunks)
    chunks_bytes = serialize_chunks(chunks)
    failed_document_keys = {
        (
            failure["tenant_id"],
            failure["document_id"],
            failure["document_version"],
        )
        for failure in failures
    }

    passed = (
        not failures
        and not zero_chunk_documents
        and processed_documents == documents_discovered
        and duplicate_chunk_ids == 0
        and chunks_above_maximum == 0
        and inaccurate_token_counts == 0
        and empty_chunks == 0
    )
    return {
        "final_status": "PASSED" if passed else "FAILED",
        "documents_discovered": documents_discovered,
        "documents_parsed": documents_parsed,
        "documents_normalized": documents_normalized,
        "documents_processed": processed_documents,
        "documents_failed": len(failed_document_keys),
        "documents_with_chunks": documents_with_chunks,
        "documents_with_zero_chunks": len(zero_chunk_documents),
        "zero_chunk_document_details": list(zero_chunk_documents),
        "normalized_blocks_processed": normalized_blocks_processed,
        "empty_blocks_skipped": empty_blocks_skipped,
        "total_chunks": len(chunks),
        "min_chunk_tokens": min(token_counts, default=0),
        "avg_chunk_tokens": round(
            sum(token_counts) / len(token_counts), 2
        ) if token_counts else 0,
        "max_chunk_tokens": max(token_counts, default=0),
        "chunks_exceeding_maximum": chunks_above_maximum,
        "empty_chunks": empty_chunks,
        "inaccurate_token_counts": inaccurate_token_counts,
        "duplicate_chunk_ids": duplicate_chunk_ids,
        "oversized_paragraphs": oversized_paragraphs,
        "token_level_fallback_splits": token_fallback_splits,
        "duplicated_overlap_tokens": duplicated_overlap_tokens,
        "chunk_counts_per_document": dict(sorted(document_counts.items())),
        "chunk_counts_per_tenant": dict(sorted(tenant_counts.items())),
        "chunk_counts_per_format": dict(sorted(format_counts.items())),
        "tokenizer": {
            "name": config.tokenizer_name,
            "version": config.tokenizer_version,
        },
        "chunking_configuration": config.to_dict(),
        "chunks_jsonl_sha256": hashlib.sha256(chunks_bytes).hexdigest(),
        "processing_failures": failures,
    }


def run_pipeline(
    manifest_path: Path,
    raw_directory: Path,
    config: ChunkingConfig,
) -> tuple[tuple[DocumentChunk, ...], dict[str, Any]]:
    """Run all in-memory stages once and return chunks plus validation report."""
    failures: list[dict[str, str]] = []
    validation = validate_manifest(manifest_path, raw_directory)
    failures.extend(_error_record(error, "validation") for error in validation.errors)

    parsing = parse_documents(validation.documents)
    failures.extend(_error_record(error, "parsing") for error in parsing.errors)

    normalization = normalize_documents(parsing.documents)
    failures.extend(
        _error_record(error, "normalization") for error in normalization.errors
    )

    chunking = chunk_documents(normalization.documents, config)
    failures.extend(_error_record(error, "chunking") for error in chunking.errors)

    report = build_chunking_report(
        documents_discovered=validation.total_manifest_records,
        documents_parsed=len(parsing.documents),
        documents_normalized=len(normalization.documents),
        chunks=chunking.chunks,
        processed_documents=chunking.processed_documents,
        documents_with_chunks=chunking.documents_with_chunks,
        zero_chunk_documents=chunking.zero_chunk_documents,
        normalized_blocks_processed=chunking.stats.normalized_blocks_processed,
        empty_blocks_skipped=chunking.stats.empty_blocks_skipped,
        oversized_paragraphs=chunking.stats.oversized_paragraphs,
        token_fallback_splits=chunking.stats.token_fallback_splits,
        duplicated_overlap_tokens=chunking.stats.duplicated_overlap_tokens,
        failures=failures,
        config=config,
    )
    return chunking.chunks, report


def _atomic_write(path: Path, content: bytes) -> None:
    """Replace one artifact only after its complete temporary file is written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            temporary_path = Path(temporary_file.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_artifacts(
    output_directory: Path,
    chunks: tuple[DocumentChunk, ...],
    report: dict[str, Any],
) -> tuple[Path, Path]:
    """Atomically write complete artifacts only for a fully passing report."""
    if report.get("final_status") != "PASSED":
        raise ValueError("Refusing to write final artifacts for a failed run")

    chunks_path = output_directory / CHUNKS_FILENAME
    report_path = output_directory / REPORT_FILENAME
    report_bytes = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write(chunks_path, serialize_chunks(chunks))
    _atomic_write(report_path, report_bytes)
    return chunks_path, report_path


def print_report(report: dict[str, Any], output: TextIO | None = None) -> None:
    """Print a concise calculated Phase 4 terminal summary."""
    if output is None:
        output = sys.stdout
    print("=" * 48, file=output)
    print("           RAG DOCUMENT CHUNKING", file=output)
    print("=" * 48, file=output)
    for key, label in (
        ("documents_discovered", "Documents discovered"),
        ("documents_processed", "Documents processed"),
        ("documents_failed", "Documents failed"),
        ("documents_with_chunks", "Documents with chunks"),
        ("documents_with_zero_chunks", "Zero-chunk documents"),
        ("normalized_blocks_processed", "Normalized blocks"),
        ("empty_blocks_skipped", "Empty blocks skipped"),
        ("total_chunks", "Total chunks"),
        ("min_chunk_tokens", "Minimum chunk tokens"),
        ("avg_chunk_tokens", "Average chunk tokens"),
        ("max_chunk_tokens", "Maximum chunk tokens"),
        ("chunks_exceeding_maximum", "Chunks above maximum"),
        ("duplicate_chunk_ids", "Duplicate chunk IDs"),
    ):
        print(f"{label + ':':<30}{report[key]}", file=output)

    if report["processing_failures"]:
        print("\nFAILURES", file=output)
        for failure in report["processing_failures"]:
            print(
                f"[{failure['processing_stage']}] "
                f"{failure['tenant_id']}/{failure['document_id']}/"
                f"{failure['document_version']}: {failure['error_type']}: "
                f"{failure['error_message']}",
                file=output,
            )
    print("=" * 48, file=output)
    print(f"FINAL STATUS: {report['final_status']}", file=output)
    print("=" * 48, file=output)


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse cwd-independent paths and configurable token budgets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--raw-directory", type=Path, default=DEFAULT_RAW_DIRECTORY)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--target-tokens", type=int, default=400)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--overlap-tokens", type=int, default=50)
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    """Run Phase 4, print its report, and write artifacts on full success."""
    args = parse_args(arguments)
    config = ChunkingConfig(
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
    )
    try:
        chunks, report = run_pipeline(args.manifest, args.raw_directory, config)
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    print_report(report)
    if report["final_status"] != "PASSED":
        return 1

    chunks_path, report_path = write_artifacts(
        args.output_directory, chunks, report
    )
    print(f"Chunks written: {chunks_path}")
    print(f"Report written: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
