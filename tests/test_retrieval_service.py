"""Verify mode orchestration, limits, and passage mapping in RetrievalService."""

from __future__ import annotations

import unittest

from app.rag.persistence.lifecycle import EmbeddingProfile
from app.rag.retrieval import (
    RetrievalError,
    RetrievalService,
    SearchContextOrigin,
    SearchFilters,
    SearchMode,
    SearchRequest,
    TrustedSearchContext,
)
from app.rag.retrieval.keyword import KeywordMatch, KeywordSearchResponse, LexicalChunk
from app.rag.retrieval.models import DenseSearchResponse, DenseSearchResult


PROFILE = EmbeddingProfile(
    model_id="BAAI/bge-small-en-v1.5",
    model_revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    config_fingerprint="c" * 64,
    normalize_embeddings=True,
    encoding_config_version="phase5-v1",
)
CONTEXT = TrustedSearchContext("tenant-a", ("public",), SearchContextOrigin.INTERNAL)


def chunk(identifier: str) -> LexicalChunk:
    """Build one complete lexical fixture."""
    return LexicalChunk(
        chunk_id="chk_" + identifier * 64, text=f"text {identifier}", tenant_id="tenant-a",
        document_id=f"DOC-{identifier}", document_version="1.0", title="Policy",
        heading_path=("Decision",), start_page=1, end_page=2, source_block_ids=(1,),
        source_path=f"sources/{identifier}.md", source_format="md", visibility="public",
        status="active", effective_date="2026-01-01",
    )


def dense(identifier: str, score: float) -> DenseSearchResult:
    """Build one complete semantic fixture whose provenance matches chunk()."""
    return DenseSearchResult(
        chunk_id="chk_" + identifier * 64, chunk_text=f"text {identifier}", similarity=score,
        cosine_distance=1 - score, tenant_id="tenant-a", document_id=f"DOC-{identifier}",
        document_version="1.0", document_title="Policy", heading_path=("Decision",),
        start_page=1, end_page=2, source_block_ids=(1,), source_path=f"sources/{identifier}.md",
        source_format="md", visibility="public", status="active", effective_date="2026-01-01",
        embedding_model_id=PROFILE.model_id, embedding_model_revision=PROFILE.model_revision,
        embedding_config_fingerprint=PROFILE.config_fingerprint,
    )


class StubKeyword:
    """Record keyword calls and return fixed BM25 matches or raise."""

    def __init__(self, identifiers: str = "", error: Exception | None = None) -> None:
        """Store the ranked identifiers or failure to produce."""
        self.identifiers = identifiers
        self.error = error
        self.limits: list[int] = []

    async def search(self, query, context, filters, limit) -> KeywordSearchResponse:
        """Return at most limit matches with descending synthetic scores."""
        self.limits.append(limit)
        if self.error:
            raise self.error
        matches = tuple(
            KeywordMatch(chunk(identifier), 10.0 - rank, rank)
            for rank, identifier in enumerate(self.identifiers[:limit], start=1)
        )
        return KeywordSearchResponse(matches, len(self.identifiers), 0.01)


class StubDense:
    """Record dense requests and return fixed cosine matches or raise."""

    def __init__(self, identifiers: str = "", error: Exception | None = None) -> None:
        """Store the ranked identifiers or failure to produce."""
        self.identifiers = identifiers
        self.error = error
        self.top_ks: list[int] = []

    async def search(self, request) -> DenseSearchResponse:
        """Return at most top_k matches with descending similarity."""
        self.top_ks.append(request.top_k)
        if self.error:
            raise self.error
        results = tuple(
            dense(identifier, 0.9 - rank / 100)
            for rank, identifier in enumerate(self.identifiers[:request.top_k], start=1)
        )
        return DenseSearchResponse(results, len(self.identifiers), 0.02, 0.03, 0.05)


def request(mode: SearchMode, k: int = 3, candidates: int = 4) -> SearchRequest:
    """Build one valid request for the selected mode."""
    return SearchRequest(
        "refund", mode, SearchFilters(), k, CONTEXT,
        PROFILE if mode is not SearchMode.KEYWORD else None, candidates, candidates,
    )


class RetrievalServiceTests(unittest.IsolatedAsyncioTestCase):
    """Exercise the service with deterministic branch stubs."""

    async def test_keyword_mode_runs_only_keyword_branch_with_k(self) -> None:
        """Use k as the keyword limit and skip the dense branch entirely."""
        keyword, semantic = StubKeyword("abcde"), StubDense("abcde")
        response = await RetrievalService(keyword, semantic).search(request(SearchMode.KEYWORD))
        self.assertEqual(keyword.limits, [3])
        self.assertEqual(semantic.top_ks, [])
        self.assertEqual([p.document_id for p in response.results], ["DOC-a", "DOC-b", "DOC-c"])
        self.assertEqual([p.final_rank for p in response.results], [1, 2, 3])
        first = response.results[0]
        self.assertEqual((first.bm25_rank, first.bm25_score), (1, 9.0))
        self.assertIsNone(first.semantic_score)
        self.assertIsNone(first.fusion_score)
        self.assertIs(first.retrieval_mode, SearchMode.KEYWORD)
        self.assertEqual(response.diagnostics.semantic_candidates, 0)
        self.assertEqual(response.diagnostics.returned_count, 3)

    async def test_semantic_mode_runs_only_dense_branch_with_k(self) -> None:
        """Use k as the dense limit and preserve similarity as semantic_score."""
        keyword, semantic = StubKeyword("abcde"), StubDense("abcde")
        response = await RetrievalService(keyword, semantic).search(request(SearchMode.SEMANTIC))
        self.assertEqual(keyword.limits, [])
        self.assertEqual(semantic.top_ks, [3])
        first = response.results[0]
        self.assertEqual((first.semantic_rank, first.semantic_score), (1, 0.89))
        self.assertIsNone(first.bm25_score)
        self.assertEqual(response.diagnostics.query_embedding_seconds, 0.02)
        self.assertEqual(response.diagnostics.vector_search_seconds, 0.03)

    async def test_hybrid_mode_uses_candidate_limits_and_fills_all_scores(self) -> None:
        """Fuse candidate lists, truncate to k, and keep both branch ranks."""
        keyword, semantic = StubKeyword("abc"), StubDense("bad")
        response = await RetrievalService(keyword, semantic).search(
            request(SearchMode.HYBRID, k=2, candidates=4)
        )
        self.assertEqual(keyword.limits, [4])
        self.assertEqual(semantic.top_ks, [4])
        # a: 1/61 + 1/62, b: 1/62 + 1/61; tie broken by BM25 rank.
        self.assertEqual([p.document_id for p in response.results], ["DOC-a", "DOC-b"])
        first = response.results[0]
        self.assertIs(first.retrieval_mode, SearchMode.HYBRID)
        self.assertEqual((first.bm25_rank, first.semantic_rank, first.final_rank), (1, 2, 1))
        self.assertEqual(first.bm25_score, 9.0)
        self.assertEqual(first.semantic_score, 0.88)
        self.assertAlmostEqual(first.fusion_score, 1 / 61 + 1 / 62)
        self.assertEqual(response.diagnostics.fused_candidates, 4)
        self.assertEqual(response.diagnostics.returned_count, 2)

    async def test_hybrid_keyword_only_candidate_has_no_semantic_fields(self) -> None:
        """Map candidates found only by BM25 without inventing semantic scores."""
        response = await RetrievalService(StubKeyword("a"), StubDense("")).search(
            request(SearchMode.HYBRID)
        )
        passage = response.results[0]
        self.assertEqual(passage.bm25_rank, 1)
        self.assertIsNone(passage.semantic_rank)
        self.assertIsNone(passage.semantic_score)
        self.assertAlmostEqual(passage.fusion_score, 1 / 61)

    async def test_no_matches_return_empty_results_not_errors(self) -> None:
        """Distinguish an ordinary empty search from a retrieval failure."""
        response = await RetrievalService(StubKeyword(""), StubDense("")).search(
            request(SearchMode.HYBRID)
        )
        self.assertEqual(response.results, ())
        self.assertEqual(response.diagnostics.fused_candidates, 0)

    async def test_either_hybrid_branch_failure_is_raised(self) -> None:
        """Never disguise one failed branch as an empty contribution."""
        for keyword, semantic in (
            (StubKeyword(error=RetrievalError("keyword down")), StubDense("a")),
            (StubKeyword("a"), StubDense(error=RetrievalError("dense down"))),
        ):
            with self.assertRaises(RetrievalError):
                await RetrievalService(keyword, semantic).search(request(SearchMode.HYBRID))

    def test_rrf_constant_must_be_positive(self) -> None:
        """Reject a fusion constant that would invert or divide by zero."""
        with self.assertRaisesRegex(ValueError, "RRF constant"):
            RetrievalService(StubKeyword(), StubDense(), rrf_constant=0)


if __name__ == "__main__":
    unittest.main()
