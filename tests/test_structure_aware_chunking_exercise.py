"""Behavior tests for the structure-aware recursive chunking exercise."""

from __future__ import annotations

import unittest

from experiments.chunking.structure_aware_chunking import (
    EXPERIMENT_CONFIGS,
    SAMPLE_DOCUMENT,
    ChunkingConfig,
    calculate_experiment_results,
    chunk_document,
    count_tokens,
    group_section_paragraphs,
    identify_sections,
    stage_3a_chunks,
)


class Stage3ASectionTests(unittest.TestCase):
    """Verify Markdown hierarchy and paragraph-aware word grouping."""

    def test_sample_headings_and_hierarchy_are_identified(self) -> None:
        """Nested headings should retain their nearest real ancestors."""
        sections = identify_sections(SAMPLE_DOCUMENT)
        self.assertEqual(
            [section.heading for section in sections],
            [
                "Refund Policy",
                "Refund Processing",
                "Subscription Policy",
                "Shipping Policy",
            ],
        )
        self.assertEqual(
            sections[1].heading_path,
            ("Refund Policy", "Refund Processing"),
        )

    def test_skipped_heading_levels_do_not_invent_parents(self) -> None:
        """A level-three heading may follow level one without a fake level two."""
        sections = identify_sections("# Parent\n\nIntro\n\n### Child\n\nDetails")
        self.assertEqual(sections[1].heading_path, ("Parent", "Child"))
        self.assertEqual(sections[1].heading_lines, ("# Parent", "### Child"))

    def test_heading_does_not_require_surrounding_blank_lines(self) -> None:
        """Markdown heading lines should be structural even beside body lines."""
        sections = identify_sections("# First\nBody one.\n## Child\nBody two.")
        self.assertEqual([section.heading for section in sections], ["First", "Child"])
        self.assertEqual(sections[0].paragraphs, ("Body one.",))
        self.assertEqual(sections[1].paragraphs, ("Body two.",))

    def test_preamble_before_first_heading_is_preserved(self) -> None:
        """Text before a heading should remain in an explicit preamble section."""
        sections = identify_sections("Opening paragraph.\n\n# Rules\n\nRule body.")
        self.assertIsNone(sections[0].heading)
        self.assertEqual(sections[0].paragraphs, ("Opening paragraph.",))
        self.assertEqual(sections[1].paragraphs, ("Rule body.",))

    def test_empty_paragraphs_do_not_create_sections_or_chunks(self) -> None:
        """Repeated blank lines should act only as boundaries."""
        sections = identify_sections("\n\n# Rules\n\n\nBody.\n\n")
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0].paragraphs, ("Body.",))
        self.assertEqual(len(stage_3a_chunks("\n\n")), 0)

    def test_paragraph_grouping_preserves_order_and_intact_paragraphs(self) -> None:
        """Stage 3A should group only complete paragraphs from one section."""
        section = identify_sections(
            "# Rules\n\none two three\n\nfour five\n\nsix seven"
        )[0]
        chunks = group_section_paragraphs(section, word_limit=5)
        self.assertEqual(chunks, ["one two three\n\nfour five", "six seven"])

    def test_oversized_stage_3a_paragraph_is_not_omitted(self) -> None:
        """Stage 3A keeps a large paragraph intact for Stage 3B to split later."""
        section = identify_sections("# Rules\n\none two three four five six")[0]
        self.assertEqual(
            group_section_paragraphs(section, word_limit=3),
            ["one two three four five six"],
        )

    def test_top_level_sections_are_never_combined(self) -> None:
        """Each Stage 3A output should retain exactly one section association."""
        grouped = stage_3a_chunks(SAMPLE_DOCUMENT, word_limit=100)
        self.assertEqual(len(grouped), 4)
        self.assertEqual(
            [section.heading for section, _ in grouped],
            [
                "Refund Policy",
                "Refund Processing",
                "Subscription Policy",
                "Shipping Policy",
            ],
        )

    def test_invalid_word_limit_is_rejected(self) -> None:
        """A non-positive grouping limit cannot produce meaningful progress."""
        section = identify_sections("# Rules\n\nBody")[0]
        with self.assertRaisesRegex(ValueError, "must be positive"):
            group_section_paragraphs(section, 0)


class Stage3BRecursiveSplittingTests(unittest.TestCase):
    """Verify token counting, recursive boundaries, limits, and preservation."""

    def chunk(
        self,
        text: str,
        target: int,
        maximum: int,
    ):
        """Chunk a test document without overlap using stable fixture metadata."""
        return chunk_document(
            text,
            "TEST-DOC",
            "1.0",
            "synthetic-test",
            ChunkingConfig(target, maximum, 0),
        )

    def test_token_count_uses_model_encoding_not_word_count(self) -> None:
        """Punctuation and uncommon words demonstrate token/word differences."""
        text = "tokenization isn't one-token-per-word: antidisestablishmentarianism"
        self.assertNotEqual(count_tokens(text), len(text.split()))

    def test_empty_and_whitespace_documents_create_no_chunks(self) -> None:
        """Documents without meaningful content should have empty results."""
        for text in ("", " \n\t "):
            with self.subTest(text=repr(text)):
                self.assertEqual(self.chunk(text, 20, 30).chunks, ())

    def test_document_without_headings_is_preserved(self) -> None:
        """Plain paragraphs should chunk under an empty heading path."""
        result = self.chunk("First paragraph.\n\nSecond paragraph.", 30, 40)
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.chunks[0].heading_path, ())
        self.assertIn("First paragraph.", result.chunks[0].text)
        self.assertIn("Second paragraph.", result.chunks[0].text)

    def test_small_section_and_exact_target_remain_together(self) -> None:
        """Content at or below its target should not be split unnecessarily."""
        text = "one two three four"
        exact_target = count_tokens(text)
        result = self.chunk(text, exact_target, exact_target + 5)
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.chunks[0].text, text)

    def test_content_above_target_splits_at_paragraph_boundary(self) -> None:
        """Separate paragraphs should be preferred over smaller boundaries."""
        first = "First paragraph has several words for grouping."
        second = "Second paragraph also has several words for grouping."
        target = count_tokens(first) + 1
        result = self.chunk(f"{first}\n\n{second}", target, target + 20)
        self.assertEqual([chunk.text for chunk in result.chunks], [first, second])

    def test_oversized_paragraph_uses_sentence_boundaries(self) -> None:
        """A long paragraph should split between sentences before token fallback."""
        sentence_one = "Alpha beta gamma delta epsilon."
        sentence_two = "Zeta eta theta iota kappa."
        maximum = max(count_tokens(sentence_one), count_tokens(sentence_two))
        result = self.chunk(
            f"{sentence_one} {sentence_two}",
            target=maximum,
            maximum=maximum,
        )
        self.assertEqual(
            [chunk.text for chunk in result.chunks],
            [sentence_one, sentence_two],
        )
        self.assertEqual(result.oversized_paragraphs, 1)
        self.assertEqual(result.token_fallback_splits, 0)

    def test_oversized_sentence_uses_token_fallback(self) -> None:
        """An uninterrupted sentence should use bounded token slices and terminate."""
        sentence = " ".join(f"word{index}" for index in range(120))
        result = self.chunk(sentence, target=25, maximum=30)
        self.assertGreater(len(result.chunks), 1)
        self.assertGreater(result.token_fallback_splits, 1)
        self.assertTrue(all(chunk.token_count <= 30 for chunk in result.chunks))

    def test_recursive_output_preserves_word_order_without_overlap(self) -> None:
        """Recursive splitting should neither lose nor reorder body words."""
        text = (
            "First sentence has useful content. Second sentence is also useful. "
            + " ".join(f"tail{index}" for index in range(80))
        )
        result = self.chunk(text, target=20, maximum=25)
        output_words = [word for chunk in result.chunks for word in chunk.text.split()]
        self.assertEqual(output_words, text.split())

    def test_nested_heading_paths_and_hard_limits_are_preserved(self) -> None:
        """Every nested chunk should carry its path and satisfy the maximum."""
        text = "# Parent\n\nIntro.\n\n## Child\n\n" + "word " * 100
        result = self.chunk(text, target=25, maximum=30)
        child_chunks = [
            chunk
            for chunk in result.chunks
            if chunk.heading_path == ("Parent", "Child")
        ]
        self.assertTrue(child_chunks)
        self.assertTrue(all(chunk.token_count <= 30 for chunk in result.chunks))

    def test_invalid_token_limits_raise_clear_errors(self) -> None:
        """Invalid target, maximum, and overlap values should fail immediately."""
        invalid_configs = (
            ChunkingConfig(0, 10, 0),
            ChunkingConfig(20, 10, 0),
            ChunkingConfig(10, 10, -1),
            ChunkingConfig(10, 10, 10),
        )
        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    chunk_document("Body", "DOC", "1", "source", config)


class Stage3CMetadataAndOverlapTests(unittest.TestCase):
    """Verify controlled overlap, metadata, IDs, ordering, and experiments."""

    def test_overlap_is_added_only_within_the_same_section(self) -> None:
        """Distinct section markers must never leak through overlap."""
        first_body = " ".join(["FIRST_ONLY"] + [f"a{i}." for i in range(100)])
        second_body = " ".join(["SECOND_ONLY"] + [f"b{i}." for i in range(100)])
        text = f"# First\n\n{first_body}\n\n# Second\n\n{second_body}"
        result = chunk_document(
            text,
            "DOC",
            "1",
            "source",
            ChunkingConfig(30, 40, 8),
        )
        for chunk in result.chunks:
            if chunk.heading_path == ("First",):
                self.assertNotIn("SECOND_ONLY", chunk.text)
            if chunk.heading_path == ("Second",):
                self.assertNotIn("FIRST_ONLY", chunk.text)

    def test_overlap_never_creates_empty_or_overlap_only_chunks(self) -> None:
        """Every emitted chunk should contain new source content and stay bounded."""
        text = "# Long\n\n" + " ".join(f"word{index}" for index in range(120))
        without_overlap = chunk_document(
            text, "DOC", "1", "source", ChunkingConfig(25, 35, 0)
        )
        with_overlap = chunk_document(
            text, "DOC", "1", "source", ChunkingConfig(25, 35, 5)
        )
        self.assertEqual(len(with_overlap.chunks), len(without_overlap.chunks))
        self.assertTrue(all(chunk.text.strip() for chunk in with_overlap.chunks))
        self.assertTrue(all(chunk.token_count <= 35 for chunk in with_overlap.chunks))
        self.assertGreater(with_overlap.duplicated_overlap_tokens, 0)

    def test_metadata_order_and_actual_token_counts_are_correct(self) -> None:
        """Chunks should preserve supplied identity and calculated source context."""
        result = chunk_document(
            SAMPLE_DOCUMENT,
            "LEARN-001",
            "2.0",
            "synthetic-reference",
            ChunkingConfig(50, 70, 5),
        )
        self.assertEqual(
            [chunk.chunk_index for chunk in result.chunks],
            list(range(len(result.chunks))),
        )
        for chunk in result.chunks:
            self.assertEqual(chunk.document_id, "LEARN-001")
            self.assertEqual(chunk.document_version, "2.0")
            self.assertEqual(chunk.source_reference, "synthetic-reference")
            self.assertEqual(chunk.token_count, count_tokens(chunk.text))

    def test_chunk_ids_and_outputs_are_deterministic(self) -> None:
        """Identical identity, text, and configuration should reproduce IDs."""
        arguments = (
            SAMPLE_DOCUMENT,
            "LEARN-001",
            "1.0",
            "synthetic-reference",
            ChunkingConfig(40, 60, 5),
        )
        self.assertEqual(chunk_document(*arguments), chunk_document(*arguments))

    def test_document_versions_cannot_share_chunk_ids(self) -> None:
        """Version identity should be part of every deterministic chunk digest."""
        first = chunk_document(
            SAMPLE_DOCUMENT, "DOC", "1.0", "source", ChunkingConfig(50, 70, 0)
        )
        second = chunk_document(
            SAMPLE_DOCUMENT, "DOC", "2.0", "source", ChunkingConfig(50, 70, 0)
        )
        self.assertTrue(
            set(chunk.chunk_id for chunk in first.chunks).isdisjoint(
                chunk.chunk_id for chunk in second.chunks
            )
        )

    def test_experiments_use_requested_configs_and_measured_limits(self) -> None:
        """All experiment results should be calculated and respect hard maxima."""
        results = calculate_experiment_results()
        self.assertEqual(
            [(name, config) for name, config, _ in results],
            list(EXPERIMENT_CONFIGS),
        )
        for _, config, result in results:
            self.assertGreater(len(result.chunks), 1)
            self.assertTrue(
                all(chunk.token_count <= config.max_tokens for chunk in result.chunks)
            )
            self.assertTrue(result.section_boundaries_preserved)


if __name__ == "__main__":
    unittest.main()
