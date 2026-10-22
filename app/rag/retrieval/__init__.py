"""Public contracts and service entry points for exact dense retrieval."""

from app.rag.retrieval.dense import DenseRetriever, RetrievalError
from app.rag.retrieval.evaluation import DenseEvaluationReport, evaluate_against_phase5
from app.rag.retrieval.models import (
    DenseSearchRequest,
    DenseSearchResponse,
    DenseSearchResult,
    HistoricalVersionScope,
    TrustedSearchContext,
)

__all__ = [
    "DenseRetriever",
    "DenseEvaluationReport",
    "DenseSearchRequest",
    "DenseSearchResponse",
    "DenseSearchResult",
    "HistoricalVersionScope",
    "RetrievalError",
    "TrustedSearchContext",
    "evaluate_against_phase5",
]
