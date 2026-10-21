"""Production Phase 5 embedding generation entry points."""

from app.rag.embeddings.encoder import SentenceTransformerEncoder
from app.rag.embeddings.models import EmbeddingConfig, EmbeddingRecord
from app.rag.embeddings.service import (
    generate_embeddings,
    load_chunks,
    load_embedding_artifacts,
    run_semantic_experiment,
    write_embedding_artifacts,
)

__all__ = [
    "EmbeddingConfig",
    "EmbeddingRecord",
    "SentenceTransformerEncoder",
    "generate_embeddings",
    "load_chunks",
    "load_embedding_artifacts",
    "run_semantic_experiment",
    "write_embedding_artifacts",
]
