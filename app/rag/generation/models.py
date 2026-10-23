"""Typed Phase 9 contracts for context, generation, cost, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from pydantic import BaseModel, ConfigDict, Field

from app.rag.reranking.models import RerankedPassage


GPT5_NANO_INPUT_USD_PER_MILLION = 0.05
GPT5_NANO_CACHED_INPUT_USD_PER_MILLION = 0.005
GPT5_NANO_OUTPUT_USD_PER_MILLION = 0.40
PRICING_EFFECTIVE_DATE = "2026-10-23"


class GenerationStatus(str, Enum):
    """Distinguish successful generation from a valid evidence absence."""

    GENERATED = "generated"
    NO_EVIDENCE = "no_evidence"


class GenerationErrorCode(str, Enum):
    """Classify failures without exposing provider or infrastructure details."""

    RETRIEVAL_ERROR = "retrieval_error"
    RERANKING_ERROR = "reranking_error"
    CONTEXT_ERROR = "context_error"
    LLM_CONFIGURATION_ERROR = "llm_configuration_error"
    LLM_PROVIDER_ERROR = "llm_provider_error"
    GENERATION_ERROR = "generation_error"
    EXTERNAL_DATA_POLICY_ERROR = "external_data_policy_error"


class GenerationPipelineError(RuntimeError):
    """Carry a safe Phase 9 failure category for API and CLI boundaries."""

    def __init__(self, code: GenerationErrorCode, message: str) -> None:
        """Store a stable category and non-sensitive diagnostic message."""
        super().__init__(message)
        self.code = code


class TokenUsage(BaseModel):
    """Preserve actual Responses API token accounting."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class CostEstimate(BaseModel):
    """Record the versioned price inputs and calculated request cost."""

    model_config = ConfigDict(frozen=True)

    estimated_cost_usd: float = Field(ge=0)
    input_usd_per_million: float = GPT5_NANO_INPUT_USD_PER_MILLION
    cached_input_usd_per_million: float = GPT5_NANO_CACHED_INPUT_USD_PER_MILLION
    output_usd_per_million: float = GPT5_NANO_OUTPUT_USD_PER_MILLION
    pricing_effective_date: str = PRICING_EFFECTIVE_DATE


class ContextDiagnostics(BaseModel):
    """Describe exactly how authorized evidence fit the context budget."""

    model_config = ConfigDict(frozen=True)

    available_context_tokens: int = Field(ge=0)
    estimated_context_tokens: int = Field(ge=0)
    estimated_total_input_tokens: int = Field(ge=0)
    included_passage_count: int = Field(ge=0)
    excluded_passage_count: int = Field(ge=0)
    truncated_chunk_ids: tuple[str, ...]


class GenerationLatency(BaseModel):
    """Separate measured pipeline stages in milliseconds."""

    model_config = ConfigDict(frozen=True)

    retrieval_ms: float = Field(ge=0)
    reranking_ms: float = Field(ge=0)
    generation_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)


class EvidenceIdentity(BaseModel):
    """Expose provenance for each passage actually supplied to generation."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    document_version: str
    title: str
    section: tuple[str, ...]
    start_page: int | None = Field(default=None, ge=1)
    end_page: int | None = Field(default=None, ge=1)
    source_reference: str
    rerank_rank: int = Field(ge=1)


class RagAnswerResponse(BaseModel):
    """Return a validated answer with evidence-supply and operational facts."""

    model_config = ConfigDict(frozen=True)

    answer: str
    model: str
    status: GenerationStatus
    evidence_chunk_ids: tuple[str, ...]
    evidence: tuple[EvidenceIdentity, ...]
    request_count: int = Field(ge=0)
    provider_response_id: str | None = None
    usage: TokenUsage | None = None
    cost: CostEstimate | None = None
    context: ContextDiagnostics
    latency: GenerationLatency


@dataclass(frozen=True)
class ContextBuildResult:
    """Carry formatted evidence and the original passages supplied to the LLM."""

    text: str
    included: tuple[RerankedPassage, ...]
    excluded_chunk_ids: tuple[str, ...]
    truncated_chunk_ids: tuple[str, ...]
    estimated_tokens: int
    available_tokens: int


@dataclass(frozen=True)
class LlmGenerationResult:
    """Represent one validated provider response before pipeline assembly."""

    answer: str
    model: str
    provider_response_id: str | None
    usage: TokenUsage | None
    cost: CostEstimate | None
    latency_ms: float

    def __post_init__(self) -> None:
        """Reject impossible timing before it reaches the public contract."""
        if not isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("Generation latency must be finite and nonnegative")
