"""Public contracts and service entry points for exact dense retrieval."""

from app.rag.retrieval.dense import DenseRetriever, RetrievalError
from app.rag.retrieval.evaluation import DenseEvaluationReport, evaluate_against_phase5
from app.rag.retrieval.models import (
    DenseSearchRequest,
    DenseSearchResponse,
    DenseSearchResult,
    HistoricalVersionScope,
    SearchContextOrigin,
    TrustedSearchContext,
)
from app.rag.retrieval.verification import (
    IntegritySummary,
    RankingComparison,
    compare_exact_ranking,
    reconcile_import_plan,
)

__all__ = [
    "DenseRetriever",
    "DenseEvaluationReport",
    "DenseSearchRequest",
    "DenseSearchResponse",
    "DenseSearchResult",
    "HistoricalVersionScope",
    "IntegritySummary",
    "RankingComparison",
    "RetrievalError",
    "SearchContextOrigin",
    "TrustedSearchContext",
    "compare_exact_ranking",
    "evaluate_against_phase5",
    "reconcile_import_plan",
]
