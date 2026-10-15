"""Behavior and integration tests for Phase 3 RAG text normalization."""

from __future__ import annotations

import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.rag.ingestion.models import ContentBlock, ParsedDocument
from app.rag.ingestion.normalization import normalize_document, normalize_documents
from scripts.normalize_rag_documents import (
    find_inspection_document,
    main,
    print_document,
    print_normalization_report,
)


class DocumentNormalizationTests(unittest.TestCase):
    """Verify safe cleanup, provenance, idempotency, and batch behavior."""

    def make_document(
        self,
        text: str,
        format_name: str = "txt",
        *,
        blocks: tuple[ContentBlock, ...] | None = None,
        **changes: object,
    ) -> ParsedDocument:
        """Build a complete parsed-document fixture with realistic metadata."""
        values: dict[str, object] = {
            "document_id": "AST-TEST-001",
            "tenant_id": "asterdesk",
            "version": "2.0",
            "status": "active",
            "title": "Normalization fixture",
            "format": format_name,
            "source_path": "documents/asterdesk/test.txt",
            "effective_date": "2026-07-01",
            "visibility": "public",
            "synthetic": "true",
            "topic_family": "support",
            "sha256": "a" * 64,
            "word_count": "20",
            "pdf_pages": "1" if format_name == "pdf" else "",
            "section_count": "1",
            "scope_note": "test",
            "content": blocks
            if blocks is not None
            else (ContentBlock(text=text, block_index=0),),
        }
        values.update(changes)
        return ParsedDocument(**values)  # type: ignore[arg-type]

    def test_whitespace_unicode_and_paragraphs_are_normalized(self) -> None:
        """Common cleanup should be deterministic without flattening paragraphs."""
        source = self.make_document(
            "\ufeff  Café\u0301\u00a0  costs   $500.\r\n\r\n\r\n"
            "It is NOT permitted unless approved.  \rNext"
        )
        normalized = normalize_document(source)
        self.assertEqual(
            normalized.content[0].text,
            "Café́ costs $500.\n\nIt is NOT permitted unless approved.\nNext",
        )
        self.assertEqual(source.content[0].text[0], "\ufeff")

    def test_unicode_nfc_preserves_non_english_math_currency_and_date(self) -> None:
        """NFC must retain semantic symbols, identifiers, and international text."""
        source = self.make_document(
            "Cafe\u0301 العربية €40 ≤ 50 on 2026-07-01."
        )
        text = normalize_document(source).content[0].text
        self.assertEqual(text, "Café العربية €40 ≤ 50 on 2026-07-01.")

    def test_markdown_front_matter_is_removed_from_first_block_only(self) -> None:
        """Bounded leading metadata should disappear while body rules remain."""
        blocks = (
            ContentBlock(
                text="---\ndocument_id: AST-TEST-001\ntenant_id: asterdesk\n---\nBody",
                block_index=4,
                section=None,
            ),
            ContentBlock(
                text="## Rules\nLater\n---\nexample: retained\n---",
                block_index=8,
                section="Rules",
            ),
        )
        source = self.make_document("", "md", blocks=blocks)
        normalized = normalize_document(source)
        self.assertEqual(normalized.content[0].text, "Body")
        self.assertIn("example: retained", normalized.content[1].text)
        self.assertEqual([block.block_index for block in normalized.content], [4, 8])
        self.assertEqual(normalized.content[1].section, "Rules")

    def test_markdown_horizontal_rule_and_malformed_front_matter_remain(self) -> None:
        """Unbounded or non-YAML delimiter content must not be mistaken for metadata."""
        horizontal_rule = self.make_document("---\nordinary prose\n---\nBody", "md")
        no_closing_rule = self.make_document("---\ntitle: Test\nBody", "md")
        self.assertTrue(
            normalize_document(horizontal_rule).content[0].text.startswith("---")
        )
        self.assertTrue(
            normalize_document(no_closing_rule).content[0].text.startswith("---")
        )

    def test_markdown_code_and_indentation_are_preserved(self) -> None:
        """Fenced and indentation-sensitive Markdown should not be flattened."""
        source = self.make_document(
            "# Example\n\n```python\nvalue  =  3\n```\n"
            "    indented  code\n  - nested   item",
            "md",
        )
        text = normalize_document(source).content[0].text
        self.assertIn("value  =  3", text)
        self.assertIn("    indented code", text)
        self.assertIn("  - nested item", text)

    def test_pdf_removes_only_matching_banner_and_page_labels(self) -> None:
        """PDF rules should be metadata-verified and preserve meaningful edge lines."""
        blocks = (
            ContentBlock(
                text=(
                    "SYNTHETIC | AST-TEST-001 v2.0 | asterdesk | active\n"
                    "Page 2\nRefund Policy\n$500 is NOT refundable unless approved.\n"
                    "Page 2 of 5"
                ),
                block_index=7,
                page=2,
                section="[DECISION] Rules",
            ),
        )
        source = self.make_document("", "pdf", blocks=blocks)
        result = normalize_documents([source])
        text = result.documents[0].content[0].text
        self.assertEqual(text, "Refund Policy\n$500 is NOT refundable unless approved.")
        self.assertEqual(result.stats.pdf_headers_removed, 1)
        self.assertEqual(result.stats.pdf_page_labels_removed, 1)
        self.assertEqual(result.stats.pdf_footers_removed, 1)
        self.assertEqual(result.documents[0].content[0].page, 2)
        self.assertEqual(result.documents[0].content[0].section, "[DECISION] Rules")

    def test_pdf_does_not_remove_unverified_or_meaningful_repeated_lines(self) -> None:
        """A generic first/last-line rule must never remove policy content."""
        source = self.make_document(
            "SYNTHETIC | OTHER v2.0 | asterdesk | active\n"
            "Page 9\nWARNING: Never share tokens.\nWARNING: Never share tokens.",
            "pdf",
            blocks=(
                ContentBlock(
                    text=(
                        "SYNTHETIC | OTHER v2.0 | asterdesk | active\n"
                        "Page 9\nWARNING: Never share tokens.\n"
                        "WARNING: Never share tokens."
                    ),
                    block_index=0,
                    page=2,
                ),
            ),
        )
        text = normalize_document(source).content[0].text
        self.assertIn("SYNTHETIC | OTHER", text)
        self.assertIn("Page 9", text)
        self.assertEqual(text.count("WARNING"), 2)

    def test_html_content_values_and_sections_remain(self) -> None:
        """HTML-derived headings, lists, and table values should only lose spacing."""
        blocks = (
            ContentBlock("Pricing", 0, section="Pricing"),
            ContentBlock("Plan    Cost", 1, section="Pricing"),
            ContentBlock("- Basic   $30", 2, section="Pricing"),
        )
        normalized = normalize_document(self.make_document("", "html", blocks=blocks))
        self.assertEqual(
            [block.section for block in normalized.content], ["Pricing"] * 3
        )
        self.assertEqual(normalized.content[0].text, "Pricing")
        self.assertEqual(normalized.content[1].text, "Plan Cost")
        self.assertEqual(normalized.content[2].text, "- Basic $30")

    def test_all_metadata_page_section_and_indexes_are_preserved(self) -> None:
        """Normalization must not alter identity, versions, or source provenance."""
        block = ContentBlock("  text  ", 11, page=3, section="[SCOPE] Coverage")
        source = self.make_document(
            "",
            "pdf",
            blocks=(block,),
            status="archived",
            version="1.0",
            visibility="internal",
        )
        normalized = normalize_document(source)
        before = source.to_dict()
        after = normalized.to_dict()
        before.pop("content")
        after.pop("content")
        self.assertEqual(after, before)
        self.assertEqual(normalized.content[0].page, 3)
        self.assertEqual(normalized.content[0].section, "[SCOPE] Coverage")
        self.assertEqual(normalized.content[0].block_index, 11)

    def test_versions_and_tenants_stay_separate_and_ordered(self) -> None:
        """Batch processing must never merge versions or reassign tenants."""
        archived = self.make_document("old", version="1.0", status="archived")
        active = self.make_document("new", version="2.0", status="active")
        other = self.make_document(
            "other", document_id="BKH-TEST-001", tenant_id="brookhaven"
        )
        result = normalize_documents([archived, active, other])
        self.assertEqual(
            [(item.tenant_id, item.version) for item in result.documents],
            [("asterdesk", "1.0"), ("asterdesk", "2.0"), ("brookhaven", "2.0")],
        )

    def test_normalization_is_idempotent_for_every_supported_format(self) -> None:
        """A second pass should produce the same complete document value."""
        for format_name in ("pdf", "html", "md", "txt"):
            with self.subTest(format_name=format_name):
                source = self.make_document(
                    "  Customers    must act.\n\n\nNow.  ", format_name
                )
                once = normalize_document(source)
                self.assertEqual(normalize_document(once), once)

    def test_batch_accumulates_errors_and_preserves_successes(self) -> None:
        """One malformed item should not discard successful later documents."""
        valid_one = self.make_document(" one ", document_id="DOC-1")
        invalid = object()
        valid_two = self.make_document(" two ", document_id="DOC-2")
        result = normalize_documents(
            [valid_one, invalid, valid_two]  # type: ignore[list-item]
        )
        self.assertEqual(
            [item.document_id for item in result.documents], ["DOC-1", "DOC-2"]
        )
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0].error_type, "TypeError")
        self.assertFalse(result.passed)

    def test_empty_batch_is_explicit_success_with_zero_counts(self) -> None:
        """An empty iterable should return a structured, deterministic result."""
        result = normalize_documents([])
        self.assertTrue(result.passed)
        self.assertEqual(result.total_documents, 0)
        self.assertEqual(result.stats.input_words, 0)

    def test_report_uses_measured_counts_and_cleanup_metrics(self) -> None:
        """CLI reporting should expose calculated values rather than constants."""
        result = normalize_documents([self.make_document("\ufeff two   words")])
        output = io.StringIO()
        print_normalization_report(result, output)
        report = output.getvalue()
        self.assertIn("Successfully normalized:        1", report)
        self.assertIn("BOM artifacts removed:           1", report)
        self.assertIn("FINAL STATUS: PASSED", report)

    def test_document_inspection_and_exact_lookup(self) -> None:
        """Inspection should show normalized content and complete source context."""
        result = normalize_documents([self.make_document(" clean   text ")])
        selected = find_inspection_document(
            result, "asterdesk", "AST-TEST-001", "2.0"
        )
        self.assertIsNotNone(selected)
        output = io.StringIO()
        print_document(selected, output)  # type: ignore[arg-type]
        printed = output.getvalue()
        self.assertIn("Status: active", printed)
        self.assertIn("Source: documents/asterdesk/test.txt", printed)
        self.assertIn("Normalized text:\nclean text", printed)


class NormalizationCommandTests(unittest.TestCase):
    """Verify phase composition, exit codes, inspection, and cwd independence."""

    def make_document(self) -> ParsedDocument:
        """Create the successful parser output used by mocked pipeline stages."""
        return DocumentNormalizationTests().make_document(" clean   text ")

    def test_successful_cli_returns_zero_and_runs_all_phases(self) -> None:
        """Successful validation, parsing, and normalization should return zero."""
        document = self.make_document()
        with (
            patch("scripts.normalize_rag_documents.validate_manifest") as validate,
            patch("scripts.normalize_rag_documents.parse_documents") as parse,
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            validate.return_value.passed = True
            validate.return_value.documents = (object(),)
            parse.return_value.passed = True
            parse.return_value.documents = (document,)
            self.assertEqual(main([]), 0)

    def test_required_stage_failure_returns_nonzero(self) -> None:
        """A Phase 1 failure should stop later processing and return nonzero."""
        with (
            patch("scripts.normalize_rag_documents.validate_manifest") as validate,
            patch("scripts.normalize_rag_documents.print_validation_report"),
            patch("scripts.normalize_rag_documents.parse_documents") as parse,
        ):
            validate.return_value.passed = False
            self.assertEqual(main([]), 1)
            parse.assert_not_called()

    def test_missing_inspection_document_returns_nonzero(self) -> None:
        """An exact inspection miss should be visible through the exit status."""
        document = self.make_document()
        with (
            patch("scripts.normalize_rag_documents.validate_manifest") as validate,
            patch("scripts.normalize_rag_documents.parse_documents") as parse,
            patch("sys.stdout", new_callable=io.StringIO),
            patch("sys.stderr", new_callable=io.StringIO),
        ):
            validate.return_value.passed = True
            validate.return_value.documents = (object(),)
            parse.return_value.passed = True
            parse.return_value.documents = (document,)
            self.assertEqual(
                main(
                    [
                        "--inspect-document",
                        "missing",
                        "--tenant",
                        "asterdesk",
                        "--version",
                        "2.0",
                    ]
                ),
                1,
            )

    def test_default_paths_are_absolute_from_another_working_directory(self) -> None:
        """Project defaults should not depend on the caller's current directory."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            previous_directory = Path.cwd()
            try:
                os.chdir(temporary_directory)
                with (
                    patch(
                        "scripts.normalize_rag_documents.validate_manifest"
                    ) as validate,
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


if __name__ == "__main__":
    unittest.main()
