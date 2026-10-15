"""Behavior tests for secure RAG manifest discovery and validation.

All document and manifest fixtures live in temporary directories so this suite
cannot modify the repository's real RAG corpus.
"""

import csv
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.discover_rag_documents import (
    REQUIRED_COLUMNS,
    main,
    print_report,
    validate_manifest,
)


class DocumentDiscoveryTests(unittest.TestCase):
    """Verify successful discovery, accumulated failures, and CLI exit behavior."""

    def setUp(self) -> None:
        """Create an isolated manifest and raw-document root for each test."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.raw_directory = self.root / "raw"
        self.manifest_path = self.root / "manifest.csv"

    def make_record(self, **changes: str) -> dict[str, str]:
        """Build a valid baseline manifest row with optional field overrides."""
        content = changes.pop("content", "valid document")
        record = {
            "document_id": "DOC-001",
            "tenant_id": "asterdesk",
            "version": "1.0",
            "effective_date": "2026-01-01",
            "status": "active",
            "visibility": "public",
            "synthetic": "true",
            "title": "Test document",
            "topic_family": "support",
            "path": "documents/asterdesk/DOC-001_v1-0.txt",
            "format": "txt",
            "word_count": "2",
            "pdf_pages": "",
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
            "section_count": "1",
            "scope_note": "synthetic test fixture",
        }
        record.update(changes)
        return record

    def write_document(
        self, record: dict[str, str], content: str = "valid document"
    ) -> Path:
        """Create a physical raw document matching a test manifest row."""
        relative_parts = Path(record["path"]).parts[2:]
        document_path = self.raw_directory / record["tenant_id"] / Path(
            *relative_parts
        )
        document_path.parent.mkdir(parents=True, exist_ok=True)
        document_path.write_text(content, encoding="utf-8")
        return document_path

    def write_manifest(
        self,
        records: list[dict[str, str]],
        fieldnames: tuple[str, ...] = REQUIRED_COLUMNS,
    ) -> None:
        """Write test records using the production manifest column ordering."""
        with self.manifest_path.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(
                target, fieldnames=fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(records)

    def test_valid_manifest_and_document_succeeds(self) -> None:
        """A complete record with matching content should produce one document."""
        record = self.make_record()
        self.write_document(record)
        self.write_manifest([record])

        result = validate_manifest(self.manifest_path, self.raw_directory)

        self.assertTrue(result.passed)
        self.assertEqual(len(result.documents), 1)
        self.assertEqual(result.documents[0].document_id, "DOC-001")
        self.assertTrue(result.documents[0].resolved_file_path.is_absolute())

    def test_missing_document_is_recorded_and_batch_continues(self) -> None:
        """A missing file should not prevent a later valid row from succeeding."""
        missing = self.make_record(document_id="DOC-MISSING")
        valid = self.make_record(
            document_id="DOC-VALID", path="documents/asterdesk/valid.txt"
        )
        self.write_document(valid)
        self.write_manifest([missing, valid])

        result = validate_manifest(self.manifest_path, self.raw_directory)

        self.assertEqual(result.error_count("missing_files"), 1)
        self.assertEqual(
            [document.document_id for document in result.documents], ["DOC-VALID"]
        )

    def test_sha256_mismatch_is_recorded(self) -> None:
        """Content that differs from its declared digest should be rejected."""
        record = self.make_record(sha256="0" * 64)
        self.write_document(record)
        self.write_manifest([record])

        result = validate_manifest(self.manifest_path, self.raw_directory)

        self.assertEqual(result.error_count("checksum_mismatches"), 1)

    def test_unsupported_file_format_is_recorded(self) -> None:
        """A format outside the supported allowlist should be reported."""
        record = self.make_record(format="docx", path="documents/asterdesk/file.docx")
        self.write_document(record)
        self.write_manifest([record])

        result = validate_manifest(self.manifest_path, self.raw_directory)

        self.assertEqual(result.error_count("unsupported_formats"), 1)

    def test_duplicate_document_version_is_recorded(self) -> None:
        """Repeated tenant, document, and version keys should be duplicates."""
        first = self.make_record()
        second = self.make_record(path="documents/asterdesk/duplicate.txt")
        self.write_document(first)
        self.write_document(second)
        self.write_manifest([first, second])

        result = validate_manifest(self.manifest_path, self.raw_directory)

        self.assertEqual(result.error_count("duplicate_document_versions"), 1)
        self.assertEqual(len(result.documents), 1)

    def test_two_versions_of_one_document_are_accepted(self) -> None:
        """Distinct versions of the same tenant document should remain valid."""
        first = self.make_record()
        second = self.make_record(
            version="2.0", path="documents/asterdesk/DOC-001_v2-0.txt"
        )
        self.write_document(first)
        self.write_document(second)
        self.write_manifest([first, second])

        result = validate_manifest(self.manifest_path, self.raw_directory)

        self.assertTrue(result.passed)
        self.assertEqual(len(result.documents), 2)

    def test_missing_required_manifest_column_fails_clearly(self) -> None:
        """A missing schema column should fail before processing data rows."""
        columns = tuple(column for column in REQUIRED_COLUMNS if column != "sha256")
        self.write_manifest([], columns)

        result = validate_manifest(self.manifest_path, self.raw_directory)

        self.assertFalse(result.passed)
        self.assertIn("sha256", result.errors[0].message)

    def test_missing_required_field_value_is_recorded(self) -> None:
        """A blank value in a required field should identify that field."""
        record = self.make_record(title="")
        self.write_document(record)
        self.write_manifest([record])

        result = validate_manifest(self.manifest_path, self.raw_directory)

        self.assertTrue(any("title" in error.message for error in result.errors))

    def test_manifest_path_traversal_is_rejected(self) -> None:
        """Parent-directory components should never reach filesystem access."""
        record = self.make_record(path="documents/asterdesk/../outside.txt")
        self.write_manifest([record])

        result = validate_manifest(self.manifest_path, self.raw_directory)

        self.assertTrue(any("traversal" in error.message for error in result.errors))

    def test_absolute_document_path_is_rejected(self) -> None:
        """Manifest rows should not be allowed to select absolute paths."""
        record = self.make_record(path="/tmp/document.txt")
        self.write_manifest([record])

        result = validate_manifest(self.manifest_path, self.raw_directory)

        self.assertTrue(any("Absolute" in error.message for error in result.errors))

    def test_wrong_tenant_in_document_path_is_rejected(self) -> None:
        """The declared path tenant must match the row's tenant metadata."""
        record = self.make_record(path="documents/brookhaven/document.txt")
        self.write_manifest([record])

        result = validate_manifest(self.manifest_path, self.raw_directory)

        self.assertTrue(any("tenant" in error.message for error in result.errors))

    def test_symlink_cannot_escape_raw_directory(self) -> None:
        """A symlink target outside the raw root should fail containment checks."""
        outside = self.root / "outside.txt"
        outside.write_text("valid document", encoding="utf-8")
        tenant_directory = self.raw_directory / "asterdesk"
        tenant_directory.mkdir(parents=True)
        (tenant_directory / "escape.txt").symlink_to(outside)
        record = self.make_record(path="documents/asterdesk/escape.txt")
        self.write_manifest([record])

        result = validate_manifest(self.manifest_path, self.raw_directory)

        self.assertTrue(any("outside" in error.message for error in result.errors))

    def test_empty_manifest_is_handled_explicitly(self) -> None:
        """A header-only manifest should return a clear batch-level error."""
        self.write_manifest([])

        result = validate_manifest(self.manifest_path, self.raw_directory)

        self.assertFalse(result.passed)
        self.assertEqual(result.total_manifest_records, 0)
        self.assertIn("no document records", result.errors[0].message)

    def test_unreadable_file_is_recorded(self) -> None:
        """A file read failure should become an error instead of aborting."""
        record = self.make_record()
        self.write_document(record)
        self.write_manifest([record])

        with patch(
            "scripts.discover_rag_documents.calculate_sha256",
            side_effect=PermissionError("permission denied"),
        ):
            result = validate_manifest(self.manifest_path, self.raw_directory)

        self.assertTrue(
            any("could not be read" in error.message for error in result.errors)
        )

    def test_multiple_invalid_documents_report_all_errors(self) -> None:
        """Independent row failures should all appear in the final report."""
        missing = self.make_record(document_id="DOC-MISSING")
        bad_hash = self.make_record(
            document_id="DOC-HASH",
            path="documents/asterdesk/hash.txt",
            sha256="0" * 64,
        )
        self.write_document(bad_hash)
        self.write_manifest([missing, bad_hash])

        result = validate_manifest(self.manifest_path, self.raw_directory)

        self.assertEqual(len(result.errors), 2)
        report = io.StringIO()
        print_report(result, report)
        self.assertIn("DOC-MISSING", report.getvalue())
        self.assertIn("DOC-HASH", report.getvalue())

    def test_successful_validation_has_zero_exit_code(self) -> None:
        """The command entry point should return zero after a valid batch."""
        record = self.make_record()
        self.write_document(record)
        self.write_manifest([record])

        with patch("sys.stdout", new_callable=io.StringIO):
            exit_code = main(
                [
                    "--manifest",
                    str(self.manifest_path),
                    "--raw-directory",
                    str(self.raw_directory),
                ]
            )

        self.assertEqual(exit_code, 0)

    def test_failed_validation_has_nonzero_exit_code(self) -> None:
        """The command entry point should return nonzero after any failure."""
        self.write_manifest([self.make_record()])

        with patch("sys.stdout", new_callable=io.StringIO):
            exit_code = main(
                [
                    "--manifest",
                    str(self.manifest_path),
                    "--raw-directory",
                    str(self.raw_directory),
                ]
            )

        self.assertNotEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
