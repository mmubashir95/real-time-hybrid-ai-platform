"""Verify Phase 8 contracts, deterministic ranking, and provenance preservation."""

from __future__ import annotations

import unittest

from app.rag.persistence.lifecycle import EmbeddingProfile
from app.rag.reranking import PairScore, RerankerConfig, RerankingService, RetrievalRerankingService
from app.rag.reranking.qwen3 import Qwen3Reranker
from app.rag.retrieval import (
    SearchContextOrigin,
    SearchFilters,
    SearchMode,
    SearchRequest,
    TrustedSearchContext,
)
from app.rag.retrieval.models import RetrievedPassage, SearchDiagnostics, SearchResponse


PROFILE = EmbeddingProfile("model", "a" * 40, "b" * 64, True, "phase5-v1")
CONTEXT = TrustedSearchContext("asterdesk", ("public",), SearchContextOrigin.INTERNAL)


def passage(identifier: str, rank: int) -> RetrievedPassage:
    """Build a complete, source-traceable Phase 7 candidate fixture."""
    return RetrievedPassage(
        chunk_id="chk_" + identifier * 64,
        document_id=f"AST-{identifier}",
        document_version="2.0",
        tenant_id="asterdesk",
        title="Refund policy",
        section=("[DECISION]",),
        start_page=1,
        end_page=2,
        text=f"policy text {identifier}",
        source_reference=f"data/rag/{identifier}.md",
        source_format="md",
        source_block_ids=(1, 2),
        status="active",
        visibility="public",
        effective_date="2026-01-01",
        retrieval_mode=SearchMode.HYBRID,
        final_rank=rank,
        bm25_score=3.0 / rank,
        bm25_rank=rank,
        semantic_score=0.9 / rank,
        semantic_rank=rank,
        fusion_score=0.03 / rank,
    )


class FakeScorer:
    """Return controlled scores while recording the exact passage order."""

    def __init__(self, values: tuple[PairScore, ...]) -> None:
        """Store deterministic score records and reusable model diagnostics."""
        self.config = RerankerConfig(batch_size=2, local_files_only=True)
        self.model_load_seconds = 0.25
        self.values = values
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def score(self, query: str, passages) -> tuple[PairScore, ...]:
        """Record ordered text inputs and return the configured output."""
        self.calls.append((query, tuple(passages)))
        return self.values


class RerankingServiceTests(unittest.TestCase):
    """Exercise reranking without downloading or initializing a model."""

    def test_scores_map_to_candidates_and_metadata_object_is_preserved(self) -> None:
        """Reorder by score while retaining the exact original passage objects."""
        candidates = (passage("a", 1), passage("b", 2), passage("c", 3))
        scorer = FakeScorer((PairScore(0.1, 20, False), PairScore(0.9, 22, True), PairScore(0.4, 21, False)))
        response = RerankingService(scorer).rerank("refund", candidates, top_k=2)

        self.assertEqual([item.passage.chunk_id for item in response.results], [candidates[1].chunk_id, candidates[2].chunk_id])
        self.assertIs(response.results[0].passage, candidates[1])
        self.assertEqual([item.retrieval_rank for item in response.results], [2, 3])
        self.assertEqual([item.rerank_rank for item in response.results], [1, 2])
        self.assertEqual(response.diagnostics.truncated_inputs, 1)
        self.assertEqual(scorer.calls, [("refund", tuple(item.text for item in candidates))])

    def test_equal_scores_use_retrieval_rank_then_chunk_id(self) -> None:
        """Make ties repeatable without changing original retrieval diagnostics."""
        candidates = (passage("b", 2), passage("a", 1))
        scores = (PairScore(0.5, 10, False), PairScore(0.5, 10, False))
        response = RerankingService(FakeScorer(scores)).rerank("refund", candidates)
        self.assertEqual([item.retrieval_rank for item in response.results], [1, 2])

    def test_empty_candidates_skip_model(self) -> None:
        """Return an ordinary empty result without invoking inference."""
        scorer = FakeScorer(())
        response = RerankingService(scorer).rerank("refund", (), top_k=5)
        self.assertEqual(response.results, ())
        self.assertEqual(scorer.calls, [])
        self.assertEqual(response.diagnostics.candidate_count, 0)

    def test_invalid_requests_and_model_output_are_rejected(self) -> None:
        """Reject invalid boundaries and score-count mismatches explicitly."""
        service = RerankingService(FakeScorer(()))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            service.rerank(" ", (), 1)
        with self.assertRaisesRegex(ValueError, "top_k"):
            service.rerank("query", (), 0)
        with self.assertRaisesRegex(ValueError, "one score"):
            service.rerank("query", (passage("a", 1),), 1)
        with self.assertRaisesRegex(ValueError, "unique"):
            duplicate = passage("a", 1)
            service.rerank("query", (duplicate, duplicate), 1)

        other_tenant = passage("b", 2)
        object.__setattr__(other_tenant, "tenant_id", "brookhaven")
        with self.assertRaisesRegex(ValueError, "one authorized tenant"):
            service.rerank("query", (passage("a", 1), other_tenant), 1)

    def test_nonfinite_score_is_rejected_at_provider_boundary(self) -> None:
        """Prevent NaN or infinity from producing unstable ordering."""
        with self.assertRaisesRegex(ValueError, "finite"):
            PairScore(float("nan"), 10, False)


class CharacterTokenizer:
    """Provide deterministic character-token IDs for input-budget tests."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Represent every character by one stable synthetic token."""
        return [ord(character) for character in text]


class QwenInputPreparationTests(unittest.TestCase):
    """Verify explicit truncation without loading a model artifact."""

    def test_long_passage_truncates_but_preserves_fixed_query_tokens(self) -> None:
        """Allocate the fixed task/query first and bound only passage tokens."""
        scorer = Qwen3Reranker(RerankerConfig(max_length=500, local_files_only=True))
        scorer._tokenizer = CharacterTokenizer()
        short_tokens, short_truncated = scorer._prepare_input("refund?", "short")
        long_tokens, long_truncated = scorer._prepare_input("refund?", "x" * 1_000)
        self.assertFalse(short_truncated)
        self.assertTrue(long_truncated)
        self.assertEqual(len(long_tokens), 500)
        # Both inputs begin with exactly the same system, instruction, and query.
        shared_prefix = next(index for index, (left, right) in enumerate(zip(short_tokens, long_tokens)) if left != right)
        self.assertGreater(shared_prefix, len("refund?"))


class FakeRetrieval:
    """Return a fixed candidate snapshot for orchestration tests."""

    def __init__(self, results: tuple[RetrievedPassage, ...]) -> None:
        """Store one authorized Phase 7 response."""
        diagnostics = SearchDiagnostics(2, 2, len(results), len(results), 0, 0, 0, 0, 0.1)
        self.response = SearchResponse(results, diagnostics)
        self.requests: list[SearchRequest] = []

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Record and return the exact candidate snapshot once."""
        self.requests.append(request)
        return self.response


class RetrievalRerankingServiceTests(unittest.IsolatedAsyncioTestCase):
    """Verify Phase 7 candidates flow unchanged into Phase 8."""

    async def test_hybrid_snapshot_is_reused_for_reranking(self) -> None:
        """Retrieve once and rerank only the returned authorized candidates."""
        candidates = (passage("a", 1), passage("b", 2))
        retrieval = FakeRetrieval(candidates)
        scorer = FakeScorer((PairScore(0.2, 10, False), PairScore(0.8, 10, False)))
        service = RetrievalRerankingService(retrieval, RerankingService(scorer))
        request = SearchRequest("refund", SearchMode.HYBRID, SearchFilters(), 2, CONTEXT, PROFILE)

        original, reranked = await service.search(request, top_k=1)

        self.assertIs(original, retrieval.response)
        self.assertEqual(len(retrieval.requests), 1)
        self.assertIs(reranked.results[0].passage, candidates[1])

    async def test_nonhybrid_mode_is_rejected_before_retrieval(self) -> None:
        """Keep the integrated workflow scoped to the specified hybrid baseline."""
        retrieval = FakeRetrieval(())
        service = RetrievalRerankingService(retrieval, RerankingService(FakeScorer(())))
        request = SearchRequest("refund", SearchMode.KEYWORD, SearchFilters(), 5, CONTEXT)
        with self.assertRaisesRegex(ValueError, "hybrid"):
            await service.search(request)
        self.assertEqual(retrieval.requests, [])


if __name__ == "__main__":
    unittest.main()
