"""Public contracts and service entry points for Phase 7 retrieval."""

from app.rag.retrieval.dense import DenseRetriever, RetrievalError
from app.rag.retrieval.hybrid import DEFAULT_RRF_CONSTANT, reciprocal_rank_fusion
from app.rag.retrieval.keyword import BM25Index, KeywordRetriever, tokenize_lexical
from app.rag.retrieval.models import (
    DenseSearchRequest,
    DenseSearchResponse,
    DenseSearchResult,
    HistoricalVersionScope,
    RetrievedPassage,
    SearchFilters,
    SearchContextOrigin,
    SearchMode,
    SearchRequest,
    SearchResponse,
    TrustedSearchContext,
)
from app.rag.retrieval.service import RetrievalService
from app.rag.retrieval.verification import (
    IntegritySummary,
    RankingComparison,
    compare_exact_ranking,
    reconcile_import_plan,
)

__all__ = [
    "BM25Index",
    "DEFAULT_RRF_CONSTANT",
    "DenseRetriever",
    "DenseSearchRequest",
    "DenseSearchResponse",
    "DenseSearchResult",
    "HistoricalVersionScope",
    "KeywordRetriever",
    "IntegritySummary",
    "RankingComparison",
    "RetrievalError",
    "RetrievalService",
    "RetrievedPassage",
    "SearchContextOrigin",
    "SearchFilters",
    "SearchMode",
    "SearchRequest",
    "SearchResponse",
    "TrustedSearchContext",
    "compare_exact_ranking",
    "reconcile_import_plan",
    "reciprocal_rank_fusion",
    "tokenize_lexical",
]
