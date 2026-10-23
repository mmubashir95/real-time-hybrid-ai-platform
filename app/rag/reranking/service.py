"""Rerank authorized Phase 7 candidates and orchestrate optional integration."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from time import perf_counter
from typing import Protocol

from app.rag.reranking.models import (
    DEFAULT_MAX_RERANK_CANDIDATES,
    DEFAULT_RERANK_TOP_K,
    PairScore,
    RerankedPassage,
    RerankerConfig,
    RerankDiagnostics,
    RerankResponse,
)
from app.rag.retrieval.models import (
    DEFAULT_MAX_QUERY_LENGTH,
    RetrievedPassage,
    SearchMode,
    SearchRequest,
    SearchResponse,
)
from app.rag.retrieval.service import RetrievalService


# One model instance runs one inference at a time, so more slots only add
# threads blocked on the model lock.
DEFAULT_MAX_CONCURRENT_RERANKS = 1


class PassageScorer(Protocol):
    """Describe the small inference boundary used by deterministic tests."""

    config: RerankerConfig
    model_load_seconds: float

    def score(self, query: str, passages: Sequence[str]) -> tuple[PairScore, ...]:
        """Return one ordered score record per passage."""


class RerankingService:
    """Validate, score, and deterministically sort authorized passages."""

    def __init__(self, scorer: PassageScorer) -> None:
        """Bind one long-lived scorer so model artifacts are reused."""
        self._scorer = scorer

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedPassage],
        top_k: int = DEFAULT_RERANK_TOP_K,
    ) -> RerankResponse:
        """Return the highest-scoring original candidates with unchanged metadata."""
        self._validate_request(query, candidates, top_k)
        if not candidates:
            return RerankResponse((), self._diagnostics(0, 0, 0, 0.0))
        load_seconds_before = self._scorer.model_load_seconds
        started = perf_counter()
        pair_scores = self._scorer.score(query, [candidate.text for candidate in candidates])
        total_scoring_seconds = perf_counter() - started
        load_seconds_after = self._scorer.model_load_seconds
        # The first score call may initialize lazily. Keep cold-start loading
        # separate from steady-state inference instead of inflating query latency.
        inference_seconds = max(
            0.0, total_scoring_seconds - max(0.0, load_seconds_after - load_seconds_before)
        )
        if len(pair_scores) != len(candidates):
            raise ValueError("Reranker must return exactly one score per candidate")
        ranked = sorted(
            zip(candidates, pair_scores, strict=True),
            key=lambda item: (-item[1].score, item[0].final_rank, item[0].chunk_id),
        )
        results = tuple(
            RerankedPassage(
                passage=passage,
                retrieval_rank=passage.final_rank,
                rerank_score=pair_score.score,
                rerank_rank=rank,
                reranker_model_id=self._scorer.config.model_id,
                reranker_model_revision=self._scorer.config.model_revision,
                input_tokens=pair_score.input_tokens,
                passage_truncated=pair_score.passage_truncated,
            )
            for rank, (passage, pair_score) in enumerate(ranked[:top_k], start=1)
        )
        return RerankResponse(
            results,
            self._diagnostics(
                len(candidates), len(results),
                sum(score.passage_truncated for score in pair_scores), inference_seconds,
            ),
        )

    def _validate_request(
        self, query: str, candidates: Sequence[RetrievedPassage], top_k: int
    ) -> None:
        """Reject invalid work and conflicting candidate provenance early."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        if len(query) > DEFAULT_MAX_QUERY_LENGTH:
            raise ValueError(f"query must not exceed {DEFAULT_MAX_QUERY_LENGTH} characters")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        if top_k <= 0 or top_k > DEFAULT_MAX_RERANK_CANDIDATES:
            raise ValueError(f"top_k must be between 1 and {DEFAULT_MAX_RERANK_CANDIDATES}")
        if len(candidates) > DEFAULT_MAX_RERANK_CANDIDATES:
            raise ValueError("Reranking candidate count exceeds the configured maximum")
        if any(not isinstance(candidate, RetrievedPassage) for candidate in candidates):
            raise TypeError("candidates must contain RetrievedPassage values")
        chunk_ids = [candidate.chunk_id for candidate in candidates]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Reranking candidates must have unique chunk IDs")
        if any(not candidate.text.strip() for candidate in candidates):
            raise ValueError("Reranking candidates must contain passage text")
        tenant_ids = {candidate.tenant_id for candidate in candidates}
        if len(tenant_ids) > 1:
            raise ValueError("Reranking candidates must belong to one authorized tenant")

    def _diagnostics(
        self, candidate_count: int, returned_count: int,
        truncated_inputs: int, inference_seconds: float,
    ) -> RerankDiagnostics:
        """Build diagnostics from actual scorer configuration and measurements."""
        config = self._scorer.config
        return RerankDiagnostics(
            candidate_count, returned_count, truncated_inputs, config.batch_size,
            self._scorer.model_load_seconds, inference_seconds, config.device, config.precision,
        )


class RetrievalRerankingService:
    """Compose existing hybrid retrieval with an optional Phase 8 rerank step."""

    def __init__(
        self,
        retrieval: RetrievalService,
        reranker: RerankingService,
        max_concurrent_reranks: int = DEFAULT_MAX_CONCURRENT_RERANKS,
    ) -> None:
        """Bind existing Phase 7 and Phase 8 services without duplicating either.

        max_concurrent_reranks bounds how many worker threads may run or wait
        on reranker inference at once; extra requests wait on the event loop.
        """
        if isinstance(max_concurrent_reranks, bool) or not isinstance(max_concurrent_reranks, int):
            raise TypeError("max_concurrent_reranks must be an integer")
        if max_concurrent_reranks <= 0:
            raise ValueError("max_concurrent_reranks must be positive")
        self._retrieval = retrieval
        self._reranker = reranker
        self._rerank_slots = asyncio.Semaphore(max_concurrent_reranks)

    async def search(
        self, request: SearchRequest, *, top_k: int = DEFAULT_RERANK_TOP_K,
    ) -> tuple[SearchResponse, RerankResponse]:
        """Retrieve one authorized hybrid candidate set, then rerank that exact set."""
        if request.mode is not SearchMode.HYBRID:
            raise ValueError("Integrated reranking requires hybrid retrieval mode")
        if request.k < top_k:
            raise ValueError("Hybrid candidate k must be greater than or equal to rerank top_k")
        retrieval_response = await self._retrieval.search(request)
        # Transformer inference is blocking; keep it off the async database event loop.
        # The model lock already serializes inference, so without this bound every
        # extra request would park an executor thread on that lock. Waiting here
        # instead is cheap and cancellable: a request cancelled before its slot
        # never starts inference. A cancelled request that already holds a slot
        # cannot interrupt its worker thread; inference finishes and the slot is
        # released when the thread returns.
        async with self._rerank_slots:
            rerank_response = await asyncio.to_thread(
                self._reranker.rerank, request.query, retrieval_response.results, top_k
            )
        return retrieval_response, rerank_response

