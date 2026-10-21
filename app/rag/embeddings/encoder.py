"""Pinned Sentence Transformers adapter for BGE document and query encoding.

Model loading is intentionally lazy at the package boundary but occurs exactly
once per encoder instance. No hosted provider or alternate-model fallback is
used when the configured local model cannot be loaded.
"""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter
from typing import Any

from app.rag.embeddings.models import EmbeddingConfig


class SentenceTransformerEncoder:
    """Load one pinned model and expose explicit document/query operations."""

    def __init__(self, config: EmbeddingConfig) -> None:
        """Load the configured revision and verify its runtime contract."""
        self.config = config
        started = perf_counter()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "sentence-transformers is required; install requirements.txt"
            ) from error

        try:
            self._model = SentenceTransformer(
                config.model_id,
                revision=config.model_revision,
                device=config.device,
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to load {config.model_id}@{config.model_revision} "
                f"on device {config.device}: {error}"
            ) from error

        self.model_load_seconds = perf_counter() - started
        self.tokenizer = self._model.tokenizer
        self.tokenizer_name = str(
            getattr(self.tokenizer, "name_or_path", config.model_id)
        )
        self.model_max_length = int(self._model.max_seq_length)
        actual_dimensions = int(self._model.get_sentence_embedding_dimension())
        if actual_dimensions != config.expected_dimensions:
            raise RuntimeError(
                "Embedding dimension mismatch: expected "
                f"{config.expected_dimensions}, model reports {actual_dimensions}"
            )
        if self.model_max_length <= 0:
            raise RuntimeError("Model reports an invalid maximum input length")

    def prepare_document(self, text: str) -> str:
        """Build exact deterministic document input without rewriting text."""
        return self.config.document_instruction + text

    def prepare_query(self, query: str) -> str:
        """Apply BGE's published retrieval instruction to a query."""
        return self.config.query_instruction + query

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        """Count model tokens including special tokens, without truncation."""
        if not texts:
            return []
        encoded = self.tokenizer(
            list(texts),
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return [len(token_ids) for token_ids in encoded["input_ids"]]

    def encode_documents(self, texts: Sequence[str]) -> Any:
        """Encode a prevalidated document batch without silent truncation."""
        prepared = [self.prepare_document(text) for text in texts]
        self._reject_overlength(prepared)
        return self._encode(prepared)

    def encode_query(self, query: str) -> Any:
        """Encode one query using the model-specific retrieval convention."""
        if not query.strip():
            raise ValueError("query must not be empty")
        prepared = self.prepare_query(query)
        self._reject_overlength([prepared])
        return self._encode([prepared])[0]

    def _reject_overlength(self, prepared_texts: Sequence[str]) -> None:
        """Fail before inference because Sentence Transformers can truncate."""
        lengths = self.count_tokens(prepared_texts)
        if any(length > self.model_max_length for length in lengths):
            raise ValueError(
                f"Input exceeds model limit of {self.model_max_length} tokens"
            )

    def _encode(self, prepared_texts: Sequence[str]) -> Any:
        """Return float32 NumPy vectors using inference-only model execution."""
        return self._model.encode(
            list(prepared_texts),
            batch_size=len(prepared_texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
        )
