"""Verify RRF arithmetic, deduplication, and deterministic ranking."""

from __future__ import annotations

import unittest

from app.rag.retrieval.hybrid import reciprocal_rank_fusion
from app.rag.retrieval.keyword import KeywordMatch, LexicalChunk
from app.rag.retrieval.models import DenseSearchResult


def chunk(identifier: str, text: str) -> LexicalChunk:
    """Build one complete lexical fixture for fusion tests."""
    return LexicalChunk(
        chunk_id="chk_" + identifier * 64, text=text, tenant_id="tenant-a",
        document_id=f"DOC-{identifier}", document_version="1.0", title="Policy",
        heading_path=("Decision",), start_page=1, end_page=1, source_block_ids=(1,),
        source_path=f"sources/{identifier}.md", source_format="md", visibility="public",
        status="active", effective_date="2026-01-01",
    )


def dense(identifier: str, score: float = 0.8) -> DenseSearchResult:
    """Build one complete semantic fixture for fusion tests."""
    return DenseSearchResult(
        chunk_id="chk_" + identifier * 64, chunk_text="text", similarity=score,
        cosine_distance=1 - score, tenant_id="tenant-a", document_id=f"DOC-{identifier}",
        document_version="1.0", document_title="Policy", heading_path=("Decision",),
        start_page=1, end_page=1, source_block_ids=(1,), source_path=f"sources/{identifier}.md",
        source_format="md", visibility="public", status="active", effective_date="2026-01-01",
        embedding_model_id="model", embedding_model_revision="a" * 40,
        embedding_config_fingerprint="b" * 64,
    )


class ReciprocalRankFusionTests(unittest.TestCase):
    """Exercise the rank-only hybrid algorithm on controlled candidates."""

    def test_shared_candidate_receives_both_rank_contributions(self) -> None:
        """Deduplicate a shared chunk and calculate RRF from one-based ranks."""
        keyword = (KeywordMatch(chunk("a", "alpha"), 4.2, 1),)
        fused = reciprocal_rank_fusion(keyword, (dense("a"), dense("b")), limit=3)
        self.assertEqual([item.chunk_id for item in fused], ["chk_" + "a" * 64, "chk_" + "b" * 64])
        self.assertAlmostEqual(fused[0].fusion_score, 2 / 61)
        self.assertEqual(fused[0].semantic_rank, 1)

    def test_one_branch_and_empty_branches_are_supported(self) -> None:
        """Treat no matches as empty input without disguising branch exceptions."""
        semantic_only = reciprocal_rank_fusion((), (dense("a"),), limit=1)
        self.assertAlmostEqual(semantic_only[0].fusion_score, 1 / 61)
        self.assertEqual(reciprocal_rank_fusion((), (), limit=5), ())

    def test_equal_scores_use_branch_ranks_then_chunk_id(self) -> None:
        """Produce repeatable output when candidates have equal RRF scores."""
        keyword = (
            KeywordMatch(chunk("b", "same"), 1.0, 1),
            KeywordMatch(chunk("a", "same"), 1.0, 1),
        )
        fused = reciprocal_rank_fusion(keyword, (), limit=2)
        self.assertEqual([item.chunk_id for item in fused], ["chk_" + "a" * 64, "chk_" + "b" * 64])

    def test_conflicting_provenance_is_rejected(self) -> None:
        """Do not merge a corrupt stable ID that points to different documents."""
        lexical = KeywordMatch(chunk("a", "alpha"), 1.0, 1)
        semantic = dense("a")
        object.__setattr__(semantic, "document_id", "OTHER")
        with self.assertRaisesRegex(ValueError, "Conflicting provenance"):
            reciprocal_rank_fusion((lexical,), (semantic,), limit=1)


if __name__ == "__main__":
    unittest.main()
