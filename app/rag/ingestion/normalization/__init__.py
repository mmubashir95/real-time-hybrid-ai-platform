"""Deterministic Phase 3 cleanup for already parsed RAG documents."""

from app.rag.ingestion.normalization.normalizer import (
    normalize_document,
    normalize_documents,
)

__all__ = ["normalize_document", "normalize_documents"]
