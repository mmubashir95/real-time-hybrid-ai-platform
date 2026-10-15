#!/usr/bin/env python3
"""Validate, parse, and normalize the RAG corpus, then report Phase 3 results.

The command composes the existing Phase 1 and Phase 2 services. Normalization
receives only parsed objects and never reads or writes source documents itself.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TextIO

# Direct script execution places scripts/ on sys.path. Use the resolved project
# root so default paths and imports remain valid from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.ingestion.discovery import validate_manifest  # noqa: E402
from app.rag.ingestion.document_parser import parse_documents  # noqa: E402
from app.rag.ingestion.models import (  # noqa: E402
    NormalizationResult,
    ParsedDocument,
)
from app.rag.ingestion.normalization import normalize_documents  # noqa: E402
from scripts.discover_rag_documents import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    DEFAULT_RAW_DIRECTORY,
    print_report as print_validation_report,
)
from scripts.parse_rag_documents import print_parsing_report  # noqa: E402


def print_normalization_report(
    result: NormalizationResult,
    output: TextIO | None = None,
) -> None:
    """Render calculated cleanup and integrity metrics for one batch."""
    if output is None:
        output = sys.stdout
    stats = result.stats

    print("=" * 40, file=output)
    print("       RAG TEXT NORMALIZATION", file=output)
    print("=" * 40, file=output)
    print("\nDOCUMENT INVENTORY", file=output)
    print("-" * 40, file=output)
    print(f"Total parsed documents:         {result.total_documents}", file=output)
    print(f"Successfully normalized:        {len(result.documents)}", file=output)
    print(f"Failed:                          {len(result.errors)}", file=output)

    print("\nCONTENT", file=output)
    print("-" * 40, file=output)
    print(f"Input content blocks:            {stats.input_content_blocks}", file=output)
    print(
        f"Output content blocks:           {stats.output_content_blocks}",
        file=output,
    )
    print(f"Input words:                     {stats.input_words}", file=output)
    print(f"Output words:                    {stats.output_words}", file=output)

    print("\nCLEANUP", file=output)
    print("-" * 40, file=output)
    print(
        f"Markdown front matter removed:   {stats.markdown_front_matters_removed}",
        file=output,
    )
    print(f"PDF headers removed:             {stats.pdf_headers_removed}", file=output)
    print(f"PDF footers removed:             {stats.pdf_footers_removed}", file=output)
    print(
        f"PDF page labels removed:         {stats.pdf_page_labels_removed}",
        file=output,
    )
    print(
        f"BOM artifacts removed:           {stats.bom_artifacts_removed}",
        file=output,
    )
    print(
        f"Whitespace-adjusted blocks:      {stats.whitespace_adjustments}",
        file=output,
    )

    print("\nVALIDATION", file=output)
    print("-" * 40, file=output)
    print(
        f"Large word-count reductions:     {stats.large_word_reductions}",
        file=output,
    )
    print(
        f"Duplicate blocks introduced:     {stats.duplicate_blocks_introduced}",
        file=output,
    )

    if result.warnings:
        print("\nWARNINGS", file=output)
        print("-" * 40, file=output)
        for warning in result.warnings:
            print(f"[WARN] {warning}", file=output)

    if result.errors:
        print("\nNORMALIZATION FAILURES", file=output)
        print("-" * 40, file=output)
        for error in result.errors:
            print(
                f"[FAIL] {error.tenant_id}/{error.document_id}/{error.version}",
                file=output,
            )
            print(f"Source: {error.source_path}", file=output)
            print(f"Reason: {error.error_type}: {error.error_message}", file=output)

    print("\n" + "=" * 40, file=output)
    print(f"FINAL STATUS: {'PASSED' if result.passed else 'FAILED'}", file=output)
    print("=" * 40, file=output)


def find_inspection_document(
    result: NormalizationResult,
    tenant_id: str,
    document_id: str,
    version: str,
) -> ParsedDocument | None:
    """Find one exact tenant/document/version among normalized successes."""
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
    """Print metadata and normalized blocks for local human inspection."""
    if output is None:
        output = sys.stdout
    print(f"Document ID: {document.document_id}", file=output)
    print(f"Tenant: {document.tenant_id}", file=output)
    print(f"Version: {document.version}", file=output)
    print(f"Status: {document.status}", file=output)
    print(f"Format: {document.format}", file=output)
    print(f"Source: {document.source_path}", file=output)
    for block in document.content:
        print(f"\nBlock index: {block.block_index}", file=output)
        if block.page is not None:
            print(f"Page: {block.page}", file=output)
        if block.section:
            print(f"Section: {block.section}", file=output)
        print(f"Normalized text:\n{block.text}", file=output)


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse corpus paths and optional tenant-qualified inspection values."""
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
    """Run all three ingestion phases and return zero only on full success."""
    args = parse_args(arguments)
    validation_result = validate_manifest(args.manifest, args.raw_directory)
    if not validation_result.passed:
        print_validation_report(validation_result)
        return 1

    parsing_result = parse_documents(validation_result.documents)
    if not parsing_result.passed:
        print_parsing_report(parsing_result)
        return 1

    normalization_result = normalize_documents(parsing_result.documents)
    print_normalization_report(normalization_result)

    if args.inspect_document:
        selected = find_inspection_document(
            normalization_result, args.tenant, args.inspect_document, args.version
        )
        if selected is None:
            print(
                "\nRequested document was not found among normalized results.",
                file=sys.stderr,
            )
            return 1
        print("\nINSPECTED DOCUMENT\n" + "-" * 40)
        print_document(selected)

    return 0 if normalization_result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
