"""Behavior tests for standardized, multi-format RAG document parsing."""

from __future__ import annotations

import csv
import hashlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from app.rag.ingestion.discovery import REQUIRED_COLUMNS, ValidatedDocument
from app.rag.ingestion.document_parser import parse_document, parse_documents
from app.rag.ingestion.parsers.html_parser import parse_html
from app.rag.ingestion.parsers.markdown_parser import parse_markdown
from app.rag.ingestion.parsers.pdf_parser import parse_pdf
from app.rag.ingestion.parsers.text_parser import parse_text
from scripts.parse_rag_documents import main, print_document, print_parsing_report


class DocumentParsingTests(unittest.TestCase):
    """Verify format extraction, metadata, failures, and CLI behavior."""

    def setUp(self) -> None:
        """Create an isolated source directory for each parsing test."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    def write_bytes(self, name: str, content: bytes) -> Path:
        """Write a binary fixture and return its resolved path."""
        path = self.root / name
        path.write_bytes(content)
        return path.resolve()

    def write_text(self, name: str, content: str) -> Path:
        """Write a UTF-8 fixture and return its resolved path."""
        path = self.root / name
        path.write_text(content, encoding="utf-8")
        return path.resolve()

    def make_pdf(self, name: str, pages: list[str]) -> Path:
        """Generate a small deterministic PDF fixture with the requested pages."""
        path = self.root / name
        with fitz.open() as pdf:
            for text in pages:
                page = pdf.new_page()
                if text:
                    page.insert_text((72, 72), text)
            pdf.save(path)
        return path.resolve()

    def make_document(
        self,
        path: Path,
        format_name: str,
        **changes: str,
    ) -> ValidatedDocument:
        """Create a Phase 1-style validated record around a test fixture."""
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata = {
            "document_id": "DOC-001",
            "tenant_id": "asterdesk",
            "version": "1.0",
            "status": "active",
            "format": format_name,
            "title": "Test document",
            "path": f"documents/asterdesk/{path.name}",
            "effective_date": "2026-01-01",
            "visibility": "public",
            "synthetic": "true",
            "topic_family": "support",
            "sha256": content_hash,
            "word_count": "2",
            "pdf_pages": "1" if format_name == "pdf" else "",
            "section_count": "1",
            "scope_note": "test fixture",
        }
        metadata.update(changes)
        return ValidatedDocument(
            document_id=metadata["document_id"],
            tenant_id=metadata["tenant_id"],
            version=metadata["version"],
            status=metadata["status"],
            format=format_name,
            title=metadata["title"],
            resolved_file_path=path,
            sha256=content_hash,
            metadata=metadata,
            manifest_row=2,
        )

    def test_text_preserves_line_breaks(self) -> None:
        """TXT extraction should retain meaningful source line structure."""
        blocks = parse_text(self.write_text("source.txt", "First\n\nSecond\n"))
        self.assertEqual(blocks[0].text, "First\n\nSecond")
        self.assertIsNone(blocks[0].page)
        self.assertIsNone(blocks[0].section)

    def test_text_section_markers_become_ordered_sections(self) -> None:
        """Bracketed section headings should attribute TXT text to sections."""
        blocks = parse_text(
            self.write_text(
                "source.txt",
                "Preamble line\n\n[SCOPE] Who is covered\nScope body\n"
                "Mentions [DECISION] inline only\n[DECISION] Rules\nRule body\n",
            )
        )
        self.assertEqual(
            [block.section for block in blocks],
            [None, "[SCOPE] Who is covered", "[DECISION] Rules"],
        )
        self.assertEqual(blocks[0].text, "Preamble line")
        self.assertIn("Mentions [DECISION] inline only", blocks[1].text)
        self.assertTrue(blocks[2].text.startswith("[DECISION] Rules\n"))
        self.assertEqual([block.block_index for block in blocks], [0, 1, 2])

    def test_empty_text_is_rejected(self) -> None:
        """Whitespace-only TXT input should be an explicit failure."""
        with self.assertRaisesRegex(ValueError, "no extractable text"):
            parse_text(self.write_text("empty.txt", " \n\t"))

    def test_invalid_text_encoding_is_reported_by_batch(self) -> None:
        """Invalid UTF-8 should become a controlled document-level error."""
        path = self.write_bytes("invalid.txt", b"\xff\xfe")
        result = parse_documents([self.make_document(path, "txt")])
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0].error_type, "UnicodeDecodeError")

    def test_markdown_headings_sections_lists_and_order(self) -> None:
        """Markdown sections should retain headings, lists, and source order."""
        path = self.write_text(
            "source.md", "Intro\n# First\n- one\n- two\n## Second\nFinal"
        )
        blocks = parse_markdown(path)
        self.assertEqual([block.section for block in blocks], [None, "First", "Second"])
        self.assertIn("# First", blocks[1].text)
        self.assertIn("- two", blocks[1].text)
        self.assertIn("Final", blocks[2].text)

    def test_markdown_heading_inside_code_fence_is_not_a_section(self) -> None:
        """Code examples that resemble headings must remain ordinary content."""
        path = self.write_text("code.md", "# Real\n```\n# code\n```\ntext")
        blocks = parse_markdown(path)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].section, "Real")
        self.assertIn("# code", blocks[0].text)

    def test_html_extracts_structure_without_scripts_or_duplicates(self) -> None:
        """HTML should retain useful structures once and remove active content."""
        path = self.write_text(
            "source.html",
            "<style>.x{}</style><script>bad()</script><h1>Policy</h1>"
            "<div><p>Nested <strong>text</strong>.</p></div>"
            "<ul><li>First</li><li>Second</li></ul>"
            "<table><tr><th>Name</th><td>Value</td></tr></table>",
        )
        blocks = parse_html(path)
        combined = "\n".join(block.text for block in blocks)
        self.assertEqual(combined.count("Nested text ."), 1)
        self.assertIn("Policy", combined)
        self.assertIn("First", combined)
        self.assertIn("Name Value", combined)
        self.assertNotIn("bad", combined)
        self.assertNotIn(".x", combined)
        self.assertTrue(all(block.section == "Policy" for block in blocks))

    def test_html_keeps_text_outside_recognized_content_tags(self) -> None:
        """Loose and mixed-content text must not be silently discarded."""
        path = self.write_text(
            "loose.html",
            "<body><header><strong>Banner</strong></header><h2>Rules</h2>"
            "<div>Div-only rule.</div>"
            "<ul><li>Parent text<ul><li>Child</li></ul></li></ul>"
            "<table><tr><td>Cell<p>Inner</p></td><td>Other cell</td></tr></table>"
            "<!-- hidden comment --></body>",
        )
        texts = [block.text for block in parse_html(path)]
        self.assertEqual(
            texts,
            [
                "Banner",
                "Rules",
                "Div-only rule.",
                "Parent text",
                "Child",
                "Cell",
                "Inner",
                "Other cell",
            ],
        )

    def test_html_without_readable_text_is_rejected(self) -> None:
        """Markup with only head, script, and style content should fail."""
        path = self.write_text(
            "empty.html",
            "<html><head><title>Title</title></head>"
            "<body><script>run()</script><style>.x{}</style></body></html>",
        )
        with self.assertRaisesRegex(ValueError, "no extractable text"):
            parse_html(path)

    def test_html_plain_fragment_uses_fallback_text(self) -> None:
        """HTML with no semantic block tags should still yield readable text."""
        blocks = parse_html(self.write_text("fragment.html", "<span>Useful</span>"))
        self.assertEqual(blocks[0].text, "Useful")

    def test_pdf_preserves_pages_and_reports_empty_page(self) -> None:
        """PDF blocks should use 1-based pages and retain empty page records."""
        blocks = parse_pdf(self.make_pdf("source.pdf", ["First", "", "Third"]))
        self.assertEqual([block.page for block in blocks], [1, 2, 3])
        self.assertEqual(blocks[1].text, "")
        self.assertEqual([block.block_index for block in blocks], [0, 1, 2])

    def test_pdf_sections_split_pages_and_continue_across_pages(self) -> None:
        """PDF blocks should carry both their page and their open section."""
        path = self.make_pdf(
            "sections.pdf",
            ["Title\n[SCOPE] Who is covered\nScope body", "More scope", ""],
        )
        blocks = parse_pdf(path)
        self.assertEqual(
            [(block.page, block.section) for block in blocks],
            [
                (1, None),
                (1, "[SCOPE] Who is covered"),
                (2, "[SCOPE] Who is covered"),
                (3, "[SCOPE] Who is covered"),
            ],
        )
        self.assertEqual(blocks[0].text, "Title")
        self.assertEqual(blocks[2].text, "More scope")
        self.assertEqual(blocks[3].text, "")
        self.assertEqual([block.block_index for block in blocks], [0, 1, 2, 3])

    def test_pdf_with_no_extractable_text_is_rejected(self) -> None:
        """An image-only or blank PDF should not silently parse as success."""
        with self.assertRaisesRegex(ValueError, "OCR is not enabled"):
            parse_pdf(self.make_pdf("blank.pdf", ["", ""]))

    def test_malformed_pdf_is_rejected(self) -> None:
        """Malformed PDF bytes should produce a controlled extraction error."""
        with self.assertRaisesRegex(ValueError, "could not be read"):
            parse_pdf(self.write_bytes("bad.pdf", b"not a pdf"))

    def test_dispatcher_preserves_metadata_and_archived_status(self) -> None:
        """The common model should keep source identity and manifest metadata."""
        path = self.write_text("source.txt", "Readable content")
        source = self.make_document(
            path,
            "txt",
            status="archived",
            version="2.0",
            visibility="internal",
        )
        parsed = parse_document(source)
        self.assertEqual(parsed.document_id, source.document_id)
        self.assertEqual(parsed.tenant_id, source.tenant_id)
        self.assertEqual(parsed.version, "2.0")
        self.assertEqual(parsed.status, "archived")
        self.assertEqual(parsed.visibility, "internal")
        self.assertEqual(parsed.to_dict()["content"][0]["text"], "Readable content")

    def test_unsupported_format_is_rejected(self) -> None:
        """The dispatcher should defensively reject an unknown declared format."""
        source = self.make_document(self.write_text("source.xml", "text"), "xml")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            parse_document(source)

    def test_batch_accumulates_multiple_failures_and_keeps_success(self) -> None:
        """Independent failures should not prevent later documents from parsing."""
        empty_one = self.make_document(self.write_text("one.txt", ""), "txt")
        empty_two = self.make_document(
            self.write_text("two.txt", ""), "txt", document_id="DOC-002"
        )
        valid = self.make_document(
            self.write_text("valid.txt", "good"), "txt", document_id="DOC-003"
        )
        result = parse_documents([empty_one, empty_two, valid])
        self.assertEqual(len(result.errors), 2)
        self.assertEqual([item.document_id for item in result.documents], ["DOC-003"])
        self.assertFalse(result.passed)

    def test_multiple_versions_remain_separate(self) -> None:
        """Parsing must not merge versions of the same document identifier."""
        path = self.write_text("source.txt", "good")
        first = self.make_document(path, "txt", version="1.0", status="archived")
        second = self.make_document(path, "txt", version="2.0", status="active")
        result = parse_documents([first, second])
        self.assertEqual([item.version for item in result.documents], ["1.0", "2.0"])

    def test_report_contains_calculated_diagnostics(self) -> None:
        """The report should calculate content, empty-page, and tenant totals."""
        source = self.make_document(self.make_pdf("source.pdf", ["words here", ""]), "pdf")
        report = io.StringIO()
        print_parsing_report(parse_documents([source]), report)
        output = report.getvalue()
        self.assertIn("Total content blocks:     2", output)
        self.assertIn("Total extracted words:    2", output)
        self.assertIn("Empty PDF pages:          1", output)
        self.assertIn("asterdesk:", output)

    def test_document_inspection_prints_page_labels(self) -> None:
        """Manual PDF inspection should clearly attribute text to source pages."""
        parsed = parse_document(
            self.make_document(self.make_pdf("source.pdf", ["Page text"]), "pdf")
        )
        output = io.StringIO()
        print_document(parsed, output)
        self.assertIn("Tenant: asterdesk", output.getvalue())
        self.assertIn("Page 1:\nPage text", output.getvalue())

    def test_successful_cli_returns_zero(self) -> None:
        """A successful validation/parsing run should use shell exit code zero."""
        source = self.make_document(self.write_text("source.txt", "good"), "txt")
        with (
            patch("scripts.parse_rag_documents.validate_manifest") as validate,
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            validate.return_value.passed = True
            validate.return_value.documents = (source,)
            self.assertEqual(main([]), 0)

    def test_failed_cli_returns_nonzero(self) -> None:
        """A parsing failure should use a nonzero shell exit code."""
        source = self.make_document(self.write_text("empty.txt", ""), "txt")
        with (
            patch("scripts.parse_rag_documents.validate_manifest") as validate,
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            validate.return_value.passed = True
            validate.return_value.documents = (source,)
            self.assertEqual(main([]), 1)

    def test_parsing_does_not_modify_source_file(self) -> None:
        """Reading a source must leave its exact bytes unchanged."""
        path = self.write_text("source.txt", "unchanged\ncontent")
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        parse_document(self.make_document(path, "txt"))
        after = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(after, before)

    def test_default_cli_paths_do_not_depend_on_current_directory(self) -> None:
        """Project-root defaults should remain valid after changing directories."""
        previous_directory = Path.cwd()
        try:
            os.chdir(self.root)
            with (
                patch("scripts.parse_rag_documents.validate_manifest") as validate,
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                validate.return_value.passed = True
                validate.return_value.documents = ()
                self.assertEqual(main([]), 0)
                manifest_path, raw_directory = validate.call_args.args
        finally:
            os.chdir(previous_directory)

        self.assertTrue(manifest_path.is_absolute())
        self.assertTrue(raw_directory.is_absolute())


class ParsingCommandEndToEndTests(unittest.TestCase):
    """Run the parsing CLI with real Phase 1 validation on temporary corpora."""

    def setUp(self) -> None:
        """Create isolated raw tenant directories and a manifest location."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.raw_directory = self.root / "raw"
        self.manifest_path = self.root / "manifest.csv"
        (self.raw_directory / "brookhaven").mkdir(parents=True)
        self.secret_path = self.raw_directory / "brookhaven" / "secret.txt"
        self.secret_path.write_text("Brookhaven confidential text", encoding="utf-8")

    def write_manifest(self, file_path: Path, declared_name: str) -> None:
        """Declare one asterdesk TXT document whose checksum matches file_path."""
        record = {column: "x" for column in REQUIRED_COLUMNS}
        record.update(
            document_id="AST-001",
            tenant_id="asterdesk",
            version="1.0",
            format="txt",
            path=f"documents/asterdesk/{declared_name}",
            pdf_pages="",
            sha256=hashlib.sha256(file_path.read_bytes()).hexdigest(),
        )
        with self.manifest_path.open("w", encoding="utf-8", newline="") as manifest:
            writer = csv.DictWriter(manifest, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader()
            writer.writerow(record)

    def run_command(self) -> tuple[int, str]:
        """Run the CLI against the temporary corpus and capture its output."""
        output = io.StringIO()
        with patch("sys.stdout", output):
            exit_code = main(
                [
                    "--manifest",
                    str(self.manifest_path),
                    "--raw-directory",
                    str(self.raw_directory),
                ]
            )
        return exit_code, output.getvalue()

    def test_valid_corpus_is_validated_and_parsed(self) -> None:
        """A valid tenant-owned document should parse with exit code zero."""
        tenant_directory = self.raw_directory / "asterdesk"
        tenant_directory.mkdir()
        source = tenant_directory / "policy.txt"
        source.write_text("[SCOPE] Coverage\nAsterDesk text", encoding="utf-8")
        self.write_manifest(source, "policy.txt")

        exit_code, output = self.run_command()

        self.assertEqual(exit_code, 0)
        self.assertIn("Successfully parsed:      1", output)
        self.assertIn("FINAL STATUS: PASSED", output)

    def test_tenant_directory_symlink_is_rejected_before_parsing(self) -> None:
        """A tenant folder pointing at another tenant must never be parsed."""
        (self.raw_directory / "asterdesk").symlink_to(
            self.raw_directory / "brookhaven", target_is_directory=True
        )
        self.write_manifest(self.secret_path, "secret.txt")

        exit_code, output = self.run_command()

        self.assertEqual(exit_code, 1)
        self.assertIn("Tenant directory resolves outside", output)
        self.assertNotIn("RAG DOCUMENT PARSING", output)

    def test_file_symlink_to_other_tenant_is_rejected_before_parsing(self) -> None:
        """A file symlink into another tenant must never be parsed."""
        tenant_directory = self.raw_directory / "asterdesk"
        tenant_directory.mkdir()
        (tenant_directory / "link.txt").symlink_to(self.secret_path)
        self.write_manifest(self.secret_path, "link.txt")

        exit_code, output = self.run_command()

        self.assertEqual(exit_code, 1)
        self.assertIn("outside the allowed tenant directory", output)
        self.assertNotIn("RAG DOCUMENT PARSING", output)


if __name__ == "__main__":
    unittest.main()
