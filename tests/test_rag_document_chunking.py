"""Behavior tests for Phase 4 production document chunking."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from app.rag.ingestion.chunking import (
    ChunkingConfig,
    chunk_document,
    chunk_documents,
    count_tokens,
)
from app.rag.ingestion.models import ContentBlock, ParsedDocument
from app.rag.ingestion.parsers.html_parser import parse_html
from app.rag.ingestion.parsers.markdown_parser import parse_markdown
from scripts.chunk_rag_documents import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_RAW_DIRECTORY,
    build_chunking_report,
    run_pipeline,
    serialize_chunks,
    write_artifacts,
)


def make_document(
    *blocks: ContentBlock,
    document_id: str = "DOC-001",
    tenant_id: str = "asterdesk",
    version: str = "1.0",
    visibility: str = "internal",
) -> ParsedDocument:
    """Build the smallest realistic normalized-document fixture."""
    return ParsedDocument(
        document_id=document_id,
        tenant_id=tenant_id,
        version=version,
        status="active",
        title="Chunking fixture",
        format="md",
        source_path="documents/asterdesk/chunking-fixture.md",
        effective_date="2026-01-01",
        visibility=visibility,
        synthetic="true",
        topic_family="support",
        sha256="fixture-sha256",
        word_count="0",
        pdf_pages="",
        section_count="1",
        scope_note="test fixture",
        content=tuple(blocks),
    )


class ChunkingConfigurationTests(unittest.TestCase):
    """Verify invalid budgets fail before any document is processed."""

    def test_invalid_token_limits_are_rejected(self) -> None:
        """Budgets must be positive, ordered, and capable of progress."""
        invalid_values = (
            {"target_tokens": 0},
            {"max_tokens": 0},
            {"target_tokens": 11, "max_tokens": 10},
            {"overlap_tokens": -1},
            {"max_tokens": 10, "overlap_tokens": 10},
            {"config_version": " "},
        )
        for values in invalid_values:
            with self.subTest(values=values), self.assertRaises(ValueError):
                ChunkingConfig(**values)


class DocumentChunkingTests(unittest.TestCase):
    """Verify hierarchy, recursive boundaries, overlap, and metadata."""

    def test_empty_blocks_are_skipped_without_emitting_empty_chunks(self) -> None:
        """Normalized provenance placeholders must not become retrieval text."""
        document = make_document(
            ContentBlock("", 0, page=1),
            ContentBlock("Useful policy text.", 1, page=2),
        )
        result = chunk_documents((document,), ChunkingConfig(overlap_tokens=0))
        self.assertTrue(result.passed)
        self.assertEqual([chunk.text for chunk in result.chunks], ["Useful policy text."])
        self.assertEqual(result.stats.empty_blocks_skipped, 1)

    def test_small_section_groups_paragraphs_and_preserves_metadata(self) -> None:
        """Paragraphs fitting the target stay together with source provenance."""
        document = make_document(
            ContentBlock(
                "First paragraph.\n\nSecond paragraph.",
                4,
                page=2,
                section="Refunds",
                heading_path=("Policies", "Refunds"),
            ),
            visibility="tenant",
        )
        chunks = chunk_document(document, ChunkingConfig(overlap_tokens=0))
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.heading_path, ("Policies", "Refunds"))
        self.assertEqual(chunk.source_block_ids, (4,))
        self.assertEqual((chunk.start_page, chunk.end_page), (2, 2))
        self.assertEqual(chunk.access_scope, "tenant")
        self.assertEqual(chunk.token_count, count_tokens(chunk.text))

    def test_section_boundaries_are_never_combined_or_overlapped(self) -> None:
        """Even generous budgets must keep unrelated top-level sections apart."""
        document = make_document(
            ContentBlock("Refund terms.", 0, section="Refunds"),
            ContentBlock("Shipping terms.", 1, section="Shipping"),
        )
        chunks = chunk_document(
            document,
            ChunkingConfig(target_tokens=50, max_tokens=60, overlap_tokens=10),
        )
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].heading_path, ("Refunds",))
        self.assertEqual(chunks[1].heading_path, ("Shipping",))
        self.assertNotIn("Refund", chunks[1].text)

    def test_oversized_paragraph_prefers_sentence_boundaries(self) -> None:
        """A long paragraph splits into complete sentences when they fit."""
        sentences = [f"Sentence {index} explains one policy detail." for index in range(18)]
        text = " ".join(sentences)
        document = make_document(ContentBlock(text, 0, section="Policy"))
        config = ChunkingConfig(target_tokens=30, max_tokens=35, overlap_tokens=0)
        chunks = chunk_document(document, config)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.token_count <= 35 for chunk in chunks))
        self.assertEqual(" ".join(chunk.text for chunk in chunks), text)

    def test_oversized_sentence_uses_lossless_token_fallback(self) -> None:
        """Uninterrupted content still terminates and remains under the maximum."""
        text = "x" * 1000
        document = make_document(ContentBlock(text, 0, section="Identifiers"))
        config = ChunkingConfig(target_tokens=20, max_tokens=25, overlap_tokens=0)
        chunks = chunk_document(document, config)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunk.text for chunk in chunks), text)
        self.assertTrue(all(chunk.token_count <= 25 for chunk in chunks))

    def test_overlap_is_bounded_and_does_not_create_overlap_only_chunks(self) -> None:
        """Adjacent same-section chunks may share context without exceeding max."""
        paragraphs = [
            f"Paragraph {index} contains enough policy context for testing."
            for index in range(12)
        ]
        document = make_document(
            ContentBlock("\n\n".join(paragraphs), 0, section="Policy")
        )
        config = ChunkingConfig(target_tokens=25, max_tokens=35, overlap_tokens=8)
        result = chunk_documents((document,), config)
        self.assertTrue(result.passed)
        self.assertGreater(len(result.chunks), 1)
        self.assertGreater(result.stats.duplicated_overlap_tokens, 0)
        self.assertTrue(all(chunk.token_count <= 35 for chunk in result.chunks))
        self.assertTrue(all(chunk.text.strip() for chunk in result.chunks))

    def test_chunk_ids_are_stable_and_isolated_by_identity_and_config(self) -> None:
        """SHA-256 IDs repeat exactly but change across relevant inputs."""
        document = make_document(ContentBlock("Stable policy text.", 0))
        config = ChunkingConfig(overlap_tokens=0)
        original = chunk_document(document, config)[0].chunk_id
        self.assertEqual(original, chunk_document(document, config)[0].chunk_id)
        self.assertNotEqual(
            original,
            chunk_document(replace(document, version="2.0"), config)[0].chunk_id,
        )
        changed_config = ChunkingConfig(overlap_tokens=0, config_version="phase4-v2")
        self.assertNotEqual(
            original, chunk_document(document, changed_config)[0].chunk_id
        )

    def test_batch_accumulates_a_document_failure_and_continues(self) -> None:
        """One malformed entry must not hide chunks from later valid documents."""
        valid = make_document(ContentBlock("Valid content.", 0))
        result = chunk_documents((object(), valid))  # type: ignore[arg-type]
        self.assertEqual(result.total_documents, 2)
        self.assertEqual(result.processed_documents, 1)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(len(result.chunks), 1)


class HeadingHierarchyTests(unittest.TestCase):
    """Verify source parsers preserve trustworthy heading ancestry."""

    def setUp(self) -> None:
        """Create an isolated parser-fixture directory."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    def test_markdown_tracks_nested_and_skipped_heading_levels(self) -> None:
        """Only actual Markdown ancestors should appear in heading paths."""
        path = self.root / "nested.md"
        path.write_text("# Top\nA\n### Deep\nB\n# Next\nC", encoding="utf-8")
        blocks = parse_markdown(path)
        self.assertEqual(
            [block.heading_path for block in blocks],
            [("Top",), ("Top", "Deep"), ("Next",)],
        )

    def test_html_tracks_nested_heading_levels(self) -> None:
        """HTML blocks should receive the current real heading hierarchy."""
        path = self.root / "nested.html"
        path.write_text(
            "<h1>Top</h1><p>A</p><h2>Child</h2><p>B</p>",
            encoding="utf-8",
        )
        blocks = parse_html(path)
        self.assertEqual(blocks[-1].heading_path, ("Top", "Child"))


class ChunkArtifactTests(unittest.TestCase):
    """Verify stable serialization, guarded writes, and corpus integration."""

    def test_serialization_and_artifact_writes_are_deterministic(self) -> None:
        """Equivalent chunks produce byte-identical JSONL and valid JSON output."""
        config = ChunkingConfig(overlap_tokens=0)
        chunks = chunk_document(
            make_document(ContentBlock("Serializable policy.", 0)), config
        )
        first = serialize_chunks(chunks)
        self.assertEqual(first, serialize_chunks(chunks))
        report = build_chunking_report(
            documents_discovered=1,
            documents_parsed=1,
            documents_normalized=1,
            chunks=chunks,
            processed_documents=1,
            documents_with_chunks=1,
            zero_chunk_documents=(),
            normalized_blocks_processed=1,
            empty_blocks_skipped=0,
            oversized_paragraphs=0,
            token_fallback_splits=0,
            duplicated_overlap_tokens=0,
            failures=[],
            config=config,
        )
        with tempfile.TemporaryDirectory() as directory:
            chunks_path, report_path = write_artifacts(Path(directory), chunks, report)
            self.assertEqual(chunks_path.read_bytes(), first)
            self.assertEqual(json.loads(report_path.read_text()), report)

    def test_failed_report_is_not_written(self) -> None:
        """Failed validation must never be published as final artifacts."""
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "failed run"):
                write_artifacts(Path(directory), (), {"final_status": "FAILED"})
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_full_corpus_is_complete_bounded_unique_and_deterministic(self) -> None:
        """All manifest documents should produce stable production-safe chunks."""
        config = ChunkingConfig()
        first_chunks, first_report = run_pipeline(
            DEFAULT_MANIFEST_PATH, DEFAULT_RAW_DIRECTORY, config
        )
        second_chunks, second_report = run_pipeline(
            DEFAULT_MANIFEST_PATH, DEFAULT_RAW_DIRECTORY, config
        )
        self.assertEqual(first_report["final_status"], "PASSED")
        self.assertEqual(first_report["documents_discovered"], 79)
        self.assertEqual(first_report["documents_processed"], 79)
        self.assertEqual(first_report["documents_with_zero_chunks"], 0)
        self.assertEqual(first_report["chunks_exceeding_maximum"], 0)
        self.assertEqual(first_report["duplicate_chunk_ids"], 0)
        self.assertEqual(serialize_chunks(first_chunks), serialize_chunks(second_chunks))
        self.assertEqual(first_report, second_report)


if __name__ == "__main__":
    unittest.main()
