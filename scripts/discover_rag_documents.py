#!/usr/bin/env python3
"""Discover and validate RAG source documents declared in a CSV manifest.

This module is intentionally independent from FastAPI and the database. It turns
untrusted manifest rows into validated, typed document records that later RAG
ingestion stages can safely consume.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "rag" / "manifest.csv"
DEFAULT_RAW_DIRECTORY = PROJECT_ROOT / "data" / "rag" / "raw"
HASH_CHUNK_SIZE = 1024 * 1024

REQUIRED_COLUMNS = (
    "document_id",
    "tenant_id",
    "version",
    "effective_date",
    "status",
    "visibility",
    "synthetic",
    "title",
    "topic_family",
    "path",
    "format",
    "word_count",
    "pdf_pages",
    "sha256",
    "section_count",
    "scope_note",
)
REQUIRED_VALUE_FIELDS = tuple(
    column for column in REQUIRED_COLUMNS if column != "pdf_pages"
)
# Non-PDF rows legitimately leave ``pdf_pages`` blank, so it is a required
# column but not a universally required value.
SUPPORTED_EXTENSIONS = {
    "pdf": ".pdf",
    "html": ".html",
    "md": ".md",
    "txt": ".txt",
}


@dataclass(frozen=True)
class ValidatedDocument:
    """A manifest record whose metadata and source file passed validation."""

    document_id: str
    tenant_id: str
    version: str
    status: str
    format: str
    title: str
    resolved_file_path: Path
    sha256: str
    metadata: dict[str, str]
    manifest_row: int


@dataclass(frozen=True)
class ValidationError:
    """One actionable problem found while validating a manifest record."""

    category: str
    message: str
    manifest_row: int | None = None
    document_id: str | None = None
    tenant_id: str | None = None
    version: str | None = None

    @property
    def document_label(self) -> str:
        """Build a compact identifier for terminal error messages."""
        parts = [self.tenant_id, self.document_id, self.version]
        return "/".join(part for part in parts if part) or "manifest"


@dataclass(frozen=True)
class ValidationResult:
    """Structured batch result for reporting and later ingestion stages."""

    total_manifest_records: int
    documents: tuple[ValidatedDocument, ...]
    errors: tuple[ValidationError, ...]

    @property
    def passed(self) -> bool:
        """Return whether the complete validation batch has no errors."""
        return not self.errors

    @property
    def invalid_documents(self) -> int:
        """Count distinct manifest rows that produced validation errors."""
        invalid_rows = {
            error.manifest_row
            for error in self.errors
            if error.manifest_row is not None
        }
        return len(invalid_rows)

    def error_count(self, category: str) -> int:
        """Count individual errors assigned to one reporting category."""
        return sum(error.category == category for error in self.errors)


def calculate_sha256(file_path: Path) -> str:
    """Calculate a file digest without loading the entire file into memory."""
    digest = hashlib.sha256()
    with file_path.open("rb") as source:
        while chunk := source.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_document_path(
    manifest_path: str,
    tenant_id: str,
    raw_directory: Path,
) -> tuple[Path | None, str | None]:
    """Map documents/<tenant>/... to raw/<tenant>/... within safe boundaries."""
    # Tenant IDs become directory names, so accepting a separator here would
    # let manifest metadata choose a directory outside its tenant boundary.
    if not tenant_id or tenant_id in {".", ".."} or any(
        separator in tenant_id for separator in ("/", "\\")
    ):
        return None, "Tenant ID must be a single safe path component"

    if not manifest_path or "\\" in manifest_path:
        return None, "Document path must be a non-empty POSIX-style relative path"

    declared_path = PurePosixPath(manifest_path)
    if declared_path.is_absolute() or PureWindowsPath(manifest_path).is_absolute():
        return None, "Absolute document paths are not allowed"
    if ".." in declared_path.parts:
        return None, "Document path traversal is not allowed"
    if len(declared_path.parts) < 3 or declared_path.parts[0] != "documents":
        return None, "Document path must start with documents/<tenant>/"
    if declared_path.parts[1] != tenant_id:
        return None, "Document path tenant does not match tenant_id"

    # resolve() follows symlinks before the containment checks. This prevents a
    # symlink located under raw/ from redirecting validation to an outside file.
    raw_root = raw_directory.resolve()
    tenant_root = (raw_root / tenant_id).resolve()
    resolved_path = (tenant_root.joinpath(*declared_path.parts[2:])).resolve()

    try:
        tenant_root.relative_to(raw_root)
        resolved_path.relative_to(tenant_root)
    except ValueError:
        return None, "Document path resolves outside the allowed tenant directory"

    return resolved_path, None


def _record_error(
    errors: list[ValidationError],
    category: str,
    message: str,
    row_number: int,
    record: dict[str, str],
) -> None:
    """Append one consistently labelled record error to the batch error list."""
    errors.append(
        ValidationError(
            category=category,
            message=message,
            manifest_row=row_number,
            document_id=record.get("document_id") or None,
            tenant_id=record.get("tenant_id") or None,
            version=record.get("version") or None,
        )
    )


def _validate_record(
    record: dict[str, str],
    row_number: int,
    raw_directory: Path,
    seen_versions: set[tuple[str, str, str]],
) -> tuple[ValidatedDocument | None, list[ValidationError]]:
    """Validate one normalized manifest row without stopping the batch."""
    errors: list[ValidationError] = []
    missing_fields = [field for field in REQUIRED_VALUE_FIELDS if not record[field]]
    for field in missing_fields:
        _record_error(
            errors,
            "other",
            f"Required field is empty: {field}",
            row_number,
            record,
        )

    tenant_id = record["tenant_id"]
    document_id = record["document_id"]
    version = record["version"]
    # A document may have many versions. Only the complete tenant/document/version
    # tuple must be unique within a manifest.
    version_key = (tenant_id, document_id, version)
    if all(version_key):
        if version_key in seen_versions:
            _record_error(
                errors,
                "duplicate_document_versions",
                "Duplicate (tenant_id, document_id, version) record",
                row_number,
                record,
            )
        else:
            seen_versions.add(version_key)

    declared_format = record["format"].lower()
    expected_extension = SUPPORTED_EXTENSIONS.get(declared_format)
    if record["format"] and expected_extension is None:
        _record_error(
            errors,
            "unsupported_formats",
            f"Unsupported format: {record['format']}",
            row_number,
            record,
        )

    resolved_path, path_error = resolve_document_path(
        record["path"], tenant_id, raw_directory
    )
    if path_error:
        _record_error(errors, "other", path_error, row_number, record)
    elif resolved_path is not None:
        if expected_extension and resolved_path.suffix.lower() != expected_extension:
            _record_error(
                errors,
                "other",
                f"File extension does not match declared format {declared_format}",
                row_number,
                record,
            )

        if not resolved_path.exists():
            _record_error(
                errors, "missing_files", "File not found", row_number, record
            )
        elif not resolved_path.is_file():
            _record_error(
                errors,
                "other",
                "Resolved path is not a regular file",
                row_number,
                record,
            )
        else:
            expected_sha256 = record["sha256"].lower()
            if expected_sha256 and (
                len(expected_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in expected_sha256
                )
            ):
                _record_error(
                    errors,
                    "other",
                    "SHA-256 value must contain exactly 64 hexadecimal characters",
                    row_number,
                    record,
                )
            elif expected_sha256:
                try:
                    actual_sha256 = calculate_sha256(resolved_path)
                except OSError as error:
                    _record_error(
                        errors,
                        "other",
                        f"File could not be read: {error}",
                        row_number,
                        record,
                    )
                else:
                    if actual_sha256 != expected_sha256:
                        _record_error(
                            errors,
                            "checksum_mismatches",
                            "SHA-256 mismatch",
                            row_number,
                            record,
                        )

    # Only fully valid rows are exposed to downstream ingestion stages. Errors
    # remain available separately in ValidationResult for complete reporting.
    if errors or resolved_path is None:
        return None, errors

    return (
        ValidatedDocument(
            document_id=document_id,
            tenant_id=tenant_id,
            version=version,
            status=record["status"],
            format=declared_format,
            title=record["title"],
            resolved_file_path=resolved_path,
            sha256=record["sha256"].lower(),
            metadata=dict(record),
            manifest_row=row_number,
        ),
        errors,
    )


def validate_manifest(manifest_path: Path, raw_directory: Path) -> ValidationResult:
    """Validate every manifest record and accumulate all discovered problems."""
    errors: list[ValidationError] = []
    documents: list[ValidatedDocument] = []
    total_records = 0

    try:
        manifest_file = manifest_path.open("r", encoding="utf-8", newline="")
    except OSError as error:
        return ValidationResult(
            total_manifest_records=0,
            documents=(),
            errors=(ValidationError("other", f"Manifest could not be read: {error}"),),
        )

    with manifest_file:
        reader = csv.DictReader(manifest_file)
        fieldnames = reader.fieldnames or []
        missing_columns = [
            column for column in REQUIRED_COLUMNS if column not in fieldnames
        ]
        if missing_columns:
            return ValidationResult(
                total_manifest_records=0,
                documents=(),
                errors=(
                    ValidationError(
                        "other",
                        "Manifest is missing required columns: "
                        + ", ".join(missing_columns),
                    ),
                ),
            )

        seen_versions: set[tuple[str, str, str]] = set()
        # CSV line 1 is the header, so data-row numbering starts at 2. Keeping
        # physical row numbers makes malformed records straightforward to find.
        for row_number, raw_record in enumerate(reader, start=2):
            total_records += 1
            if None in raw_record or any(
                value is None for value in raw_record.values()
            ):
                errors.append(
                    ValidationError(
                        "other",
                        "Malformed CSV record has too many or too few values",
                        manifest_row=row_number,
                    )
                )
                continue

            record = {key: value.strip() for key, value in raw_record.items()}
            document, record_errors = _validate_record(
                record, row_number, raw_directory, seen_versions
            )
            errors.extend(record_errors)
            if document is not None:
                documents.append(document)

    if total_records == 0:
        errors.append(ValidationError("other", "Manifest contains no document records"))

    return ValidationResult(
        total_manifest_records=total_records,
        documents=tuple(documents),
        errors=tuple(errors),
    )


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
