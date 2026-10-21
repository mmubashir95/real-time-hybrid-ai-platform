"""Typed configuration and result contracts for local chunk embeddings.

The records reference the immutable Phase 4 chunk identity and copy only the
metadata required to validate and later index each vector row.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_MODEL_ID = "BAAI/bge-small-en-v1.5"
DEFAULT_MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
DEFAULT_DIMENSIONS = 384
DEFAULT_BATCH_SIZE = 32
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
ENCODING_CONFIG_VERSION = "phase5-v1"


@dataclass(frozen=True)
class EmbeddingConfig:
    """Pinned model and inference settings for one homogeneous artifact."""

    model_id: str = DEFAULT_MODEL_ID
    model_revision: str = DEFAULT_MODEL_REVISION
    expected_dimensions: int = DEFAULT_DIMENSIONS
    batch_size: int = DEFAULT_BATCH_SIZE
    device: str = "cpu"
    normalize_embeddings: bool = True
    query_instruction: str = QUERY_INSTRUCTION
    document_instruction: str = ""
    encoding_config_version: str = ENCODING_CONFIG_VERSION

    def __post_init__(self) -> None:
        """Reject invalid or non-reproducible settings before model loading."""
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if len(self.model_revision) != 40 or any(
            character not in "0123456789abcdef" for character in self.model_revision
        ):
            raise ValueError("model_revision must be a 40-character commit SHA")
        if self.expected_dimensions <= 0:
            raise ValueError("expected_dimensions must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self.device.strip():
            raise ValueError("device must not be empty")

    def fingerprint_fields(self) -> dict[str, Any]:
        """Return fields that determine the exact document-vector contract."""
        return {
            "document_instruction": self.document_instruction,
            "encoding_config_version": self.encoding_config_version,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "normalize_embeddings": self.normalize_embeddings,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


@dataclass(frozen=True)
class EmbeddingRecord:
    """Metadata mapping one matrix row to one immutable source chunk."""

    row_index: int
    chunk_id: str
    tenant_id: str
    document_id: str
    document_version: str
    document_status: str
    access_scope: str
    heading_path: tuple[str, ...]
    start_page: int | None
    end_page: int | None
    source_file: str
    source_type: str
    source_block_ids: tuple[int, ...]
    chunk_token_count: int
    chunk_tokenizer_name: str
    chunk_tokenizer_version: str
    chunking_config_version: str
    input_fingerprint: str
    embedding_dimensions: int
    embedding_model_id: str
    embedding_model_revision: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation for JSONL persistence."""
        return asdict(self)


@dataclass(frozen=True)
class TokenLength:
    """Token compatibility measurement for one prepared chunk input."""

    chunk_id: str
    original_chunk_tokens: int
    embedding_tokens: int


@dataclass(frozen=True)
class GenerationResult:
    """Validated vectors, row metadata, and measured generation statistics."""

    vectors: Any
    records: tuple[EmbeddingRecord, ...]
    token_lengths: tuple[TokenLength, ...]
    model_load_seconds: float
    encoding_seconds: float
    batch_durations_seconds: tuple[float, ...]
    tokenizer_name: str
    model_max_length: int
    device: str
    corpus_sha256: str
