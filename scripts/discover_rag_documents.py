#!/usr/bin/env python3
"""Discover and validate RAG source documents declared in a CSV manifest.

Validation rules live in ``app.rag.ingestion.discovery``; this command only
supplies project-root defaults, prints the report, and sets the exit code.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import TextIO

# Running ``python scripts/discover_rag_documents.py`` puts scripts/ rather than
# the project root on sys.path. Add the stable repository root before app imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.ingestion.discovery import (  # noqa: E402
    ValidationResult,
    validate_manifest,
)


DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "rag" / "manifest.csv"
DEFAULT_RAW_DIRECTORY = PROJECT_ROOT / "data" / "rag" / "raw"


def print_report(result: ValidationResult, output: TextIO | None = None) -> None:
    """Render a human-readable validation summary."""
    if output is None:
        output = sys.stdout

    format_counts = Counter(document.format for document in result.documents)
    tenant_counts = Counter(document.tenant_id for document in result.documents)

    print("=" * 40, file=output)
    print("       RAG DOCUMENT VALIDATION", file=output)
    print("=" * 40, file=output)
    print("\nDOCUMENT INVENTORY", file=output)
    print("-" * 40, file=output)
    print(f"Total manifest records: {result.total_manifest_records}", file=output)
    print(f"Valid documents: {len(result.documents)}", file=output)
    print(f"Invalid documents: {result.invalid_documents}", file=output)

    print("\nDOCUMENT FORMATS", file=output)
    print("-" * 40, file=output)
    for format_name, label in (
        ("pdf", "PDF"),
        ("html", "HTML"),
        ("md", "Markdown"),
        ("txt", "TXT"),
    ):
        print(f"{label + ':':<10}{format_counts[format_name]}", file=output)

    print("\nTENANTS", file=output)
    print("-" * 40, file=output)
    if tenant_counts:
        for tenant_id in sorted(tenant_counts):
            print(f"{tenant_id + ':':<15}{tenant_counts[tenant_id]}", file=output)
    else:
        print("None", file=output)

    print("\nVALIDATION RESULTS", file=output)
    print("-" * 40, file=output)
    categories = (
        ("missing_files", "Missing files"),
        ("unsupported_formats", "Unsupported formats"),
        ("checksum_mismatches", "Checksum mismatches"),
        ("duplicate_document_versions", "Duplicate document versions"),
        ("other", "Other validation errors"),
    )
    for category, label in categories:
        print(f"{label + ':':<29}{result.error_count(category)}", file=output)

    if result.errors:
        print("\nERROR DETAILS", file=output)
        print("-" * 40, file=output)
        for error in result.errors:
            row = f" row {error.manifest_row}" if error.manifest_row else ""
            print(f"[FAIL] {error.document_label}{row}", file=output)
            print(f"Reason: {error.message}", file=output)

    print("\n" + "=" * 40, file=output)
    print(f"FINAL STATUS: {'PASSED' if result.passed else 'FAILED'}", file=output)
    print("=" * 40, file=output)


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse optional paths while keeping project-root defaults cwd-independent."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--raw-directory", type=Path, default=DEFAULT_RAW_DIRECTORY)
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    """Run validation, print its report, and return a shell-friendly exit code."""
    args = parse_args(arguments)
    result = validate_manifest(args.manifest, args.raw_directory)
    print_report(result)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
