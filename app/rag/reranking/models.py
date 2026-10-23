"""Typed contracts for pretrained reranking of authorized retrieval results."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from app.rag.retrieval.models import RetrievedPassage


DEFAULT_RERANKER_MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_RERANKER_REVISION = "fd9fb1d26c07223ced488065909faf522e29cc7d"
DEFAULT_RERANKER_BATCH_SIZE = 4
DEFAULT_RERANKER_MAX_LENGTH = 8_192
DEFAULT_MAX_RERANK_CANDIDATES = 200
DEFAULT_RERANK_TOP_K = 5
DEFAULT_RERANK_INSTRUCTION = (
    "Given a customer-support question, retrieve policy passages that answer the question"
)


@dataclass(frozen=True)
class RerankerConfig:
    """Define one reproducible Qwen3 inference configuration."""

    model_id: str = DEFAULT_RERANKER_MODEL_ID
    model_revision: str = DEFAULT_RERANKER_REVISION
    tokenizer_revision: str = DEFAULT_RERANKER_REVISION
    instruction: str = DEFAULT_RERANK_INSTRUCTION
    max_length: int = DEFAULT_RERANKER_MAX_LENGTH
    batch_size: int = DEFAULT_RERANKER_BATCH_SIZE
    device: str = "cpu"
    precision: str = "auto"
    local_files_only: bool = False

    def __post_init__(self) -> None:
        """Reject moving revisions and unsafe inference bounds before loading."""
        if not self.model_id.strip() or not self.instruction.strip():
            raise ValueError("Reranker model_id and instruction must not be empty")
        for name in ("model_revision", "tokenizer_revision"):
            revision = getattr(self, name)
            if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
                raise ValueError(f"{name} must be a 40-character commit SHA")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (self.max_length, self.batch_size)):
            raise TypeError("Reranker max_length and batch_size must be integers")
        if self.max_length <= 0 or self.batch_size <= 0:
            raise ValueError("Reranker max_length and batch_size must be positive")
        if not self.device.strip():
            raise ValueError("Reranker device must not be empty")
        if self.precision not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("Unsupported reranker precision")


@dataclass(frozen=True)
class PairScore:
    """Associate one finite model score with its measured input handling."""

    score: float
    input_tokens: int
    passage_truncated: bool

    def __post_init__(self) -> None:
        """Reject malformed provider output before it can affect ranking."""
        if not isfinite(self.score):
            raise ValueError("Reranker scores must be finite")
        if isinstance(self.input_tokens, bool) or not isinstance(self.input_tokens, int):
            raise TypeError("Reranker input token count must be an integer")
        if self.input_tokens <= 0:
            raise ValueError("Reranker input token count must be positive")


@dataclass(frozen=True)
class RerankedPassage:
    """Add reranking evidence without copying or modifying passage provenance."""

    passage: RetrievedPassage
    retrieval_rank: int
    rerank_score: float
    rerank_rank: int
    reranker_model_id: str
    reranker_model_revision: str
    input_tokens: int
    passage_truncated: bool


@dataclass(frozen=True)
class RerankDiagnostics:
    """Report measured reranker work separately from Phase 7 retrieval."""

    candidate_count: int
    returned_count: int
    truncated_inputs: int
    batch_size: int
    model_load_seconds: float
    inference_seconds: float
    device: str
    precision: str


@dataclass(frozen=True)
class RerankResponse:
    """Return ranked authorized passages and operational measurements."""

    results: tuple[RerankedPassage, ...]
    diagnostics: RerankDiagnostics

