"""Verify genuine BM25 preprocessing, scoring, and index safety."""

from __future__ import annotations

import unittest

from app.rag.retrieval.keyword import BM25Index, LexicalChunk, tokenize_lexical


def chunk(identifier: str, text: str) -> LexicalChunk:
    """Build one complete lexical fixture with stable source metadata."""
    return LexicalChunk(
        chunk_id="chk_" + identifier * 64, text=text, tenant_id="tenant-a",
        document_id=f"DOC-{identifier}", document_version="1.0", title="Policy",
        heading_path=("Decision",), start_page=1, end_page=1, source_block_ids=(1,),
        source_path=f"sources/{identifier}.md", source_format="md", visibility="public",
        status="active", effective_date="2026-01-01",
    )


class BM25IndexTests(unittest.TestCase):
    """Exercise lexical matching without a database or model runtime."""

    def test_tokenizer_preserves_identifiers_and_components(self) -> None:
        """Match exact hyphenated identifiers plus case-insensitive components."""
        self.assertEqual(
            tokenize_lexical("Policy AST-REF-001, $25.00"),
            ("policy", "ast-ref-001", "ast", "ref", "001", "25", "00"),
        )

    def test_exact_terms_rank_above_partial_matches(self) -> None:
        """Use BM25 term frequency and rarity rather than substring scoring."""
        index = BM25Index((
            chunk("a", "duplicate payment duplicate payment AST-REF-001"),
            chunk("b", "payment policy"),
            chunk("c", "unrelated account access"),
        ))
        results = index.search("DUPLICATE payment", 3)
        self.assertEqual(
            [item.chunk.chunk_id for item in results],
            ["chk_" + "a" * 64, "chk_" + "b" * 64],
        )
        self.assertGreater(results[0].score, results[1].score)

    def test_empty_no_match_and_deterministic_ties(self) -> None:
        """Return no false matches and use immutable IDs for equal scores."""
        self.assertEqual(BM25Index(()).search("anything", 5), ())
        index = BM25Index((chunk("b", "same"), chunk("a", "same")))
        self.assertEqual(index.search("missing", 5), ())
        self.assertEqual(
            [item.chunk.chunk_id for item in index.search("same", 5)],
            ["chk_" + "a" * 64, "chk_" + "b" * 64],
        )

    def test_duplicate_and_empty_chunks_are_rejected(self) -> None:
        """Prevent ambiguous index mappings and invalid retrieval content."""
        duplicate = chunk("a", "valid")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            BM25Index((duplicate, duplicate))
        with self.assertRaisesRegex(ValueError, "empty"):
            BM25Index((chunk("a", "  "),))


if __name__ == "__main__":
    unittest.main()
