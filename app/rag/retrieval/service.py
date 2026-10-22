"""Orchestrate keyword, semantic, and hybrid retrieval behind one contract."""

from __future__ import annotations

from time import perf_counter

from app.rag.retrieval.dense import DenseRetriever
from app.rag.retrieval.hybrid import DEFAULT_RRF_CONSTANT, FusedCandidate, reciprocal_rank_fusion
from app.rag.retrieval.keyword import KeywordMatch, KeywordRetriever
from app.rag.retrieval.models import (
    DenseSearchRequest,
    DenseSearchResult,
    RetrievedPassage,
    SearchDiagnostics,
    SearchMode,
    SearchRequest,
    SearchResponse,
)


class RetrievalService:
    """Execute only the branches required by a validated search mode."""

    def __init__(
        self,
        keyword_retriever: KeywordRetriever,
        dense_retriever: DenseRetriever,
        *,
        rrf_constant: int = DEFAULT_RRF_CONSTANT,
    ) -> None:
        """Bind reusable branch services and the evaluated RRF constant."""
        if rrf_constant <= 0:
            raise ValueError("RRF constant must be positive")
        self._keyword = keyword_retriever
        self._dense = dense_retriever
        self._rrf_constant = rrf_constant

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Return consistently shaped passages for the selected retrieval mode."""
        started = perf_counter()
        keyword_response = None
        dense_response = None
        fusion_seconds = 0.0

        if request.mode in (SearchMode.KEYWORD, SearchMode.HYBRID):
            keyword_response = await self._keyword.search(
                request.query, request.context, request.filters,
                request.keyword_candidate_limit if request.mode is SearchMode.HYBRID else request.k,
            )
        if request.mode in (SearchMode.SEMANTIC, SearchMode.HYBRID):
            # SearchRequest guarantees this profile before either dense mode can run.
            assert request.profile is not None
            dense_response = await self._dense.search(DenseSearchRequest(
                context=request.context,
                profile=request.profile,
                question=request.query,
                top_k=request.semantic_candidate_limit if request.mode is SearchMode.HYBRID else request.k,
                max_top_k=request.max_candidate_limit,
                filters=request.filters,
            ))

        if request.mode is SearchMode.KEYWORD:
            assert keyword_response is not None
            passages = tuple(
                self._from_keyword(match, rank, request.mode)
                for rank, match in enumerate(keyword_response.results[:request.k], start=1)
            )
            fused_count = len(passages)
        elif request.mode is SearchMode.SEMANTIC:
            assert dense_response is not None
            passages = tuple(
                self._from_semantic(result, rank, request.mode)
                for rank, result in enumerate(dense_response.results[:request.k], start=1)
            )
            fused_count = len(passages)
        else:
            assert keyword_response is not None and dense_response is not None
            fusion_started = perf_counter()
            fused = reciprocal_rank_fusion(
                keyword_response.results, dense_response.results,
                limit=request.k, constant=self._rrf_constant,
            )
            passages = tuple(
                self._from_fused(candidate, rank)
                for rank, candidate in enumerate(fused, start=1)
            )
            fusion_seconds = perf_counter() - fusion_started
            fused_count = len({item.chunk_id for item in keyword_response.results} | {
                item.chunk_id for item in dense_response.results
            })

        diagnostics = SearchDiagnostics(
            keyword_candidates=len(keyword_response.results) if keyword_response else 0,
            semantic_candidates=len(dense_response.results) if dense_response else 0,
            fused_candidates=fused_count,
            returned_count=len(passages),
            keyword_seconds=keyword_response.total_seconds if keyword_response else 0.0,
            query_embedding_seconds=dense_response.query_encoding_seconds if dense_response else 0.0,
            vector_search_seconds=dense_response.database_seconds if dense_response else 0.0,
            fusion_seconds=fusion_seconds,
            total_seconds=perf_counter() - started,
        )
        return SearchResponse(passages, diagnostics)

    @staticmethod
    def _from_keyword(match: KeywordMatch, rank: int, mode: SearchMode) -> RetrievedPassage:
        """Convert a lexical match to the shared passage contract."""
        chunk = match.chunk
        return RetrievedPassage(
            chunk_id=chunk.chunk_id, document_id=chunk.document_id,
            document_version=chunk.document_version, tenant_id=chunk.tenant_id,
            title=chunk.title, section=chunk.heading_path, start_page=chunk.start_page,
            end_page=chunk.end_page, text=chunk.text, source_reference=chunk.source_path,
            source_format=chunk.source_format, source_block_ids=chunk.source_block_ids,
            status=chunk.status, visibility=chunk.visibility, effective_date=chunk.effective_date,
            retrieval_mode=mode, final_rank=rank, bm25_score=match.score,
            bm25_rank=match.rank,
        )

    @staticmethod
    def _from_semantic(result: DenseSearchResult, rank: int, mode: SearchMode) -> RetrievedPassage:
        """Convert a dense match to the shared passage contract."""
        return RetrievedPassage(
            chunk_id=result.chunk_id, document_id=result.document_id,
            document_version=result.document_version, tenant_id=result.tenant_id,
            title=result.document_title, section=result.heading_path,
            start_page=result.start_page, end_page=result.end_page, text=result.chunk_text,
            source_reference=result.source_path, source_format=result.source_format,
            source_block_ids=result.source_block_ids, status=result.status,
            visibility=result.visibility, effective_date=result.effective_date,
            retrieval_mode=mode, final_rank=rank, semantic_score=result.similarity,
            semantic_rank=rank,
        )

    def _from_fused(self, candidate: FusedCandidate, rank: int) -> RetrievedPassage:
        """Build one hybrid passage while preserving both branch diagnostics."""
        if candidate.semantic is not None:
            passage = self._from_semantic(candidate.semantic, rank, SearchMode.HYBRID)
        else:
            assert candidate.keyword is not None
            passage = self._from_keyword(candidate.keyword, rank, SearchMode.HYBRID)
        return RetrievedPassage(
            **{**passage.__dict__,
               "bm25_score": candidate.keyword.score if candidate.keyword else None,
               "bm25_rank": candidate.keyword.rank if candidate.keyword else None,
               "semantic_rank": candidate.semantic_rank,
               "fusion_score": candidate.fusion_score}
        )
