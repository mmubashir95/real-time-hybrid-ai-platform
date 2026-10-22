"""Validated PostgreSQL persistence for RAG documents and embeddings."""

from app.rag.persistence.artifacts import ImportPlan, load_import_plan
from app.rag.persistence.importer import (
    ImportCounts,
    ImportResult,
    RagImportConflict,
    import_rag_embeddings,
)
from app.rag.persistence.lifecycle import (
    DocumentLifecycleError,
    DocumentNotFound,
    EmbeddingProfile,
    IncompleteVersion,
    LifecycleConflict,
    StageResult,
    TransitionResult,
    VersionNotFound,
    chunk_ids_fingerprint,
    fetch_current_chunk_ids,
    publish_document_version,
    stage_document_version,
    withdraw_document_version,
)

__all__ = [
    "ImportCounts",
    "ImportPlan",
    "ImportResult",
    "DocumentLifecycleError",
    "DocumentNotFound",
    "EmbeddingProfile",
    "IncompleteVersion",
    "LifecycleConflict",
    "RagImportConflict",
    "StageResult",
    "TransitionResult",
    "VersionNotFound",
    "chunk_ids_fingerprint",
    "fetch_current_chunk_ids",
    "import_rag_embeddings",
    "load_import_plan",
    "publish_document_version",
    "stage_document_version",
    "withdraw_document_version",
]
