#!/usr/bin/env python3
"""Validate and parse the RAG corpus, then print extraction diagnostics.

Phase 1 remains the security and integrity boundary. This command parses only
the validated records it returns and never discovers source paths independently.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import TextIO

# Running ``python scripts/parse_rag_documents.py`` puts scripts/ rather than the
# project root on sys.path. Add the stable repository root before app imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.ingestion.document_parser import parse_documents  # noqa: E402
from app.rag.ingestion.models import ParsedDocument, ParsingResult  # noqa: E402
from scripts.discover_rag_documents import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    DEFAULT_RAW_DIRECTORY,
    print_report as print_validation_report,
    validate_manifest,
)


def print_parsing_report(
    result: ParsingResult, output: TextIO | None = None
) -> None:
    """Render dynamic parsing, format, tenant, and content statistics."""
    if output is None:
        output = sys.stdout

    format_counts = Counter(document.format for document in result.documents)
    tenant_counts = Counter(document.tenant_id for document in result.documents)
    print("=" * 40, file=output)
    print("        RAG DOCUMENT PARSING", file=output)
    print("=" * 40, file=output)
    print("\nDOCUMENT INVENTORY", file=output)
    print("-" * 40, file=output)
    print(f"Total documents:          {result.total_documents}", file=output)
    print(f"Successfully parsed:      {len(result.documents)}", file=output)
    print(f"Failed:                   {len(result.errors)}", file=output)

    print("\nDOCUMENT FORMATS", file=output)
    print("-" * 40, file=output)
    for format_name, label in (
        ("pdf", "PDF"),
        ("html", "HTML"),
        ("md", "Markdown"),
        ("txt", "TXT"),
    ):
        print(f"{label + ':':<26}{format_counts[format_name]}", file=output)

    print("\nCONTENT EXTRACTION", file=output)
    print("-" * 40, file=output)
    print(f"Total content blocks:     {result.total_content_blocks}", file=output)
    print(f"Total extracted words:    {result.total_extracted_words}", file=output)
    print(f"Empty documents:          {result.empty_documents}", file=output)
    print(f"Empty PDF pages:          {result.empty_pdf_pages}", file=output)
    print(f"Extraction failures:      {len(result.errors)}", file=output)

    print("\nTENANTS", file=output)
    print("-" * 40, file=output)
    for tenant_id in sorted(tenant_counts):
        print(f"{tenant_id + ':':<26}{tenant_counts[tenant_id]}", file=output)

    if result.errors:
        print("\nPARSING FAILURES", file=output)
        print("-" * 40, file=output)
        for error in result.errors:
            print(
                f"[FAIL] {error.tenant_id}/{error.document_id}/{error.version}",
                file=output,
            )
            print(f"Source: {error.source_path}", file=output)
            print(
                f"Reason: {error.error_type}: {error.error_message}", file=output
            )

    print("\n" + "=" * 40, file=output)
    print(f"FINAL STATUS: {'PASSED' if result.passed else 'FAILED'}", file=output)
    print("=" * 40, file=output)


def find_inspection_document(
    result: ParsingResult,
    tenant_id: str,
    document_id: str,
    version: str,
) -> ParsedDocument | None:
    """Find one exact tenant/document/version tuple in successful results."""
    return next(
        (
            document
            for document in result.documents
            if document.tenant_id == tenant_id
            and document.document_id == document_id
            and document.version == version
        ),
        None,
    )


def print_document(document: ParsedDocument, output: TextIO | None = None) -> None:
    """Print extracted blocks for local, tenant-qualified inspection."""
    if output is None:
        output = sys.stdout
    print(f"Document ID: {document.document_id}", file=output)
    print(f"Tenant: {document.tenant_id}", file=output)
    print(f"Version: {document.version}", file=output)
    print(f"Format: {document.format}", file=output)
    for block in document.content:
        if block.page is not None:
            label = f"Page {block.page}"
        elif block.section:
            label = f"Section: {block.section}"
        else:
            label = f"Block {block.block_index}"
        print(f"\n{label}:\n{block.text}", file=output)


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse paths and optional exact document-inspection identifiers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--raw-directory", type=Path, default=DEFAULT_RAW_DIRECTORY)
    parser.add_argument("--inspect-document")
    parser.add_argument("--tenant")
    parser.add_argument("--version")
    args = parser.parse_args(arguments)
    inspection_values = (args.inspect_document, args.tenant, args.version)
    if any(inspection_values) and not all(inspection_values):
        parser.error(
            "--inspect-document, --tenant, and --version must be provided together"
        )
    return args


def main(arguments: list[str] | None = None) -> int:
    """Validate, parse, report, optionally inspect, and return an exit code."""
    args = parse_args(arguments)
    validation_result = validate_manifest(args.manifest, args.raw_directory)
    if not validation_result.passed:
        print_validation_report(validation_result)
        return 1

    parsing_result = parse_documents(validation_result.documents)
    print_parsing_report(parsing_result)

    if args.inspect_document:
        selected = find_inspection_document(
            parsing_result, args.tenant, args.inspect_document, args.version
        )
        if selected is None:
            print(
                "\nRequested document was not found among successful parsing results.",
                file=sys.stderr,
            )
            return 1
        print("\nINSPECTED DOCUMENT\n" + "-" * 40)
        print_document(selected)

    return 0 if parsing_result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
