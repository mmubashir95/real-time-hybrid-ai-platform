"""Tests for the isolated simple word-chunking learning exercise."""

from __future__ import annotations

import unittest

from experiments.chunking.simple_chunking import (
    EXPERIMENTS,
    calculate_duplicate_words,
    chunk_text,
    chunk_text_with_overlap,
    document,
)


class FixedSizeChunkingTests(unittest.TestCase):
    """Verify fixed-size boundaries, validation, order, and preservation."""

    def test_empty_input_returns_empty_list(self) -> None:
        """An empty document should not create an empty chunk."""
        self.assertEqual(chunk_text(""), [])

    def test_whitespace_only_input_returns_empty_list(self) -> None:
        """Whitespace without words should not create a chunk."""
        self.assertEqual(chunk_text(" \n\t  "), [])

    def test_text_smaller_than_chunk_size_returns_one_chunk(self) -> None:
        """A short document should remain together in one chunk."""
        self.assertEqual(chunk_text("one two three", 5), ["one two three"])

    def test_text_equal_to_chunk_size_returns_one_chunk(self) -> None:
        """A document exactly matching the boundary should not create a remainder."""
        self.assertEqual(chunk_text("one two three", 3), ["one two three"])

    def test_text_larger_than_chunk_size_returns_multiple_chunks(self) -> None:
        """Words beyond a boundary should continue in a smaller final chunk."""
        self.assertEqual(
            chunk_text("one two three four five", 2),
            ["one two", "three four", "five"],
        )

    def test_non_positive_chunk_sizes_are_rejected(self) -> None:
        """Zero and negative sizes cannot advance through a document."""
        for invalid_size in (0, -1):
            with self.subTest(chunk_size=invalid_size):
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    chunk_text("one two", invalid_size)

    def test_all_words_keep_their_original_order(self) -> None:
        """Flattening fixed chunks should exactly reproduce the source word list."""
        source_words = "alpha beta gamma delta epsilon zeta eta".split()
        chunks = chunk_text(" ".join(source_words), 3)
        reconstructed_words = [word for chunk in chunks for word in chunk.split()]
        self.assertEqual(reconstructed_words, source_words)


class OverlappingChunkingTests(unittest.TestCase):
    """Verify overlap validation, shared boundaries, and loop termination."""

    def test_empty_and_whitespace_input_return_empty_lists(self) -> None:
        """Valid overlap settings should still produce no chunks without words."""
        self.assertEqual(chunk_text_with_overlap(""), [])
        self.assertEqual(chunk_text_with_overlap(" \n\t "), [])

    def test_invalid_chunk_size_is_rejected(self) -> None:
        """Overlap chunking also requires a positive chunk size."""
        with self.assertRaisesRegex(ValueError, "must be positive"):
            chunk_text_with_overlap("one two", chunk_size=0)

    def test_invalid_overlaps_are_rejected(self) -> None:
        """Overlap must be non-negative and smaller than the chunk size."""
        for invalid_overlap in (-1, 5, 6):
            with self.subTest(overlap=invalid_overlap):
                with self.assertRaisesRegex(ValueError, "overlap must be"):
                    chunk_text_with_overlap(
                        "one two three", chunk_size=5, overlap=invalid_overlap
                    )

    def test_zero_overlap_matches_fixed_size_chunking(self) -> None:
        """A zero overlap should behave exactly like fixed-size chunking."""
        text = "one two three four five six seven"
        self.assertEqual(
            chunk_text_with_overlap(text, chunk_size=3, overlap=0),
            chunk_text(text, chunk_size=3),
        )

    def test_valid_overlap_shares_expected_adjacent_words(self) -> None:
        """The documented five-word example should share two boundary words."""
        text = "A B C D E F G H I J K L"
        chunks = chunk_text_with_overlap(text, chunk_size=5, overlap=2)
        self.assertEqual(
            chunks,
            ["A B C D E", "D E F G H", "G H I J K", "J K L"],
        )
        for previous, current in zip(chunks, chunks[1:]):
            self.assertEqual(previous.split()[-2:], current.split()[:2])

    def test_short_text_does_not_create_trailing_duplicate_chunk(self) -> None:
        """Reaching the final word should stop before an overlap-only chunk."""
        self.assertEqual(
            chunk_text_with_overlap("one two three", chunk_size=5, overlap=2),
            ["one two three"],
        )

    def test_overlap_preserves_source_order_and_covers_every_word(self) -> None:
        """Removing repeated boundaries should reconstruct the original words."""
        source_words = "A B C D E F G H I J K L".split()
        overlap = 2
        chunks = chunk_text_with_overlap(
            " ".join(source_words), chunk_size=5, overlap=overlap
        )
        reconstructed = chunks[0].split()
        for chunk in chunks[1:]:
            reconstructed.extend(chunk.split()[overlap:])
        self.assertEqual(reconstructed, source_words)

    def test_duplicate_word_count_uses_actual_chunk_occurrences(self) -> None:
        """Reported duplication should equal output occurrences minus source words."""
        chunks = chunk_text_with_overlap(document, chunk_size=20, overlap=5)
        expected = sum(len(chunk.split()) for chunk in chunks) - len(document.split())
        self.assertEqual(calculate_duplicate_words(chunks, document), expected)

    def test_requested_experiments_have_calculated_counts(self) -> None:
        """All four configurations should produce their expected measured totals."""
        calculated_results: list[tuple[str, int, int]] = []
        for name, chunk_size, overlap in EXPERIMENTS:
            chunks = chunk_text_with_overlap(document, chunk_size, overlap)
            calculated_results.append(
                (
                    name,
                    len(chunks),
                    calculate_duplicate_words(chunks, document),
                )
            )

        self.assertEqual(
            calculated_results,
            [("A", 8, 0), ("B", 4, 0), ("C", 5, 20), ("D", 3, 20)],
        )


if __name__ == "__main__":
    unittest.main()
