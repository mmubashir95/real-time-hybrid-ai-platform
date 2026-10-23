"""Typed contracts for Phase 10 claim grounding, citations, and diagnostics."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.rag.generation.models import CostEstimate, TokenUsage


class GroundingStatus(str, Enum):
    """Describe whether a safe, verified answer could be produced."""

    GROUNDED = "grounded"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    GROUNDING_FAILED = "grounding_failed"


class ClaimSupport(str, Enum):
    """Use conservative semantic support outcomes for one generated claim."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class GeneratedClaim(BaseModel):
    """Represent one model-produced factual claim and its requested evidence IDs."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(min_length=1, max_length=2_000)
    evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def reject_duplicate_evidence(self) -> GeneratedClaim:
        """Reject repeated IDs so downstream citation accounting stays unambiguous."""
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("Claim evidence_ids must be unique")
        return self


class GroundedDraft(BaseModel):
    """Constrain the first provider pass to claims or explicit insufficiency."""

    model_config = ConfigDict(extra="forbid")

    status: GroundingStatus
    claims: tuple[GeneratedClaim, ...] = Field(default=(), max_length=12)

    @model_validator(mode="after")
    def validate_status_claims(self) -> GroundedDraft:
        """Require claims only for a proposed grounded draft."""
        if self.status == GroundingStatus.GROUNDED and not self.claims:
            raise ValueError("A grounded draft requires at least one claim")
        if self.status != GroundingStatus.GROUNDED and self.claims:
            raise ValueError("An insufficient draft must not include claims")
        return self


class ClaimVerification(BaseModel):
    """Carry the verifier decision for exactly one indexed generated claim."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    claim_index: int = Field(ge=0)
    outcome: ClaimSupport
    reason: str = Field(min_length=1, max_length=500)


class VerificationBatch(BaseModel):
    """Constrain one bounded verifier response for all structurally valid claims."""

    model_config = ConfigDict(extra="forbid")

    decisions: tuple[ClaimVerification, ...] = Field(max_length=12)


class EvidenceRecord(BaseModel):
    """Preserve server-authoritative identity for one supplied passage."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str
    chunk_id: str
    tenant_id: str
    document_id: str
    document_version: str
    document_title: str
    source_reference: str
    source_format: str
    section: tuple[str, ...]
    start_page: int | None = Field(default=None, ge=1)
    end_page: int | None = Field(default=None, ge=1)
    passage_text: str = Field(repr=False)


class ValidatedCitation(BaseModel):
    """Expose a rendered citation backed solely by the immutable registry."""

    model_config = ConfigDict(frozen=True)

    citation_number: int = Field(ge=1)
    evidence_id: str
    chunk_id: str
    document_id: str
    document_version: str
    document_title: str
    section: tuple[str, ...]
    start_page: int | None = Field(default=None, ge=1)
    end_page: int | None = Field(default=None, ge=1)


class GroundingDiagnostics(BaseModel):
    """Report measured and counted grounding outcomes without false confidence."""

    model_config = ConfigDict(frozen=True)

    generated_claim_count: int = Field(ge=0)
    verified_claim_count: int = Field(ge=0)
    supported_claim_count: int = Field(ge=0)
    unsupported_claim_count: int = Field(ge=0)
    contradicted_claim_count: int = Field(ge=0)
    insufficient_evidence_count: int = Field(ge=0)
    invalid_citation_count: int = Field(ge=0)
    included_evidence_count: int = Field(ge=0)
    verification_limitation: str


class GroundingLatency(BaseModel):
    """Separate measured retrieval, reranking, generation, and verification time."""

    model_config = ConfigDict(frozen=True)

    retrieval_ms: float = Field(ge=0)
    reranking_ms: float = Field(ge=0)
    generation_ms: float = Field(ge=0)
    verification_ms: float | None = Field(default=None, ge=0)
    total_ms: float = Field(ge=0)


class GroundedAnswerResponse(BaseModel):
    """Return only semantically accepted claims with traceable citations."""

    model_config = ConfigDict(frozen=True)

    status: GroundingStatus
    answer: str
    citations: tuple[ValidatedCitation, ...]
    evidence: tuple[EvidenceRecord, ...]
    grounding: GroundingDiagnostics
    generation_model: str
    verification_model: str | None = None
    generation_usage: TokenUsage | None = None
    verification_usage: TokenUsage | None = None
    generation_cost: CostEstimate | None = None
    verification_cost: CostEstimate | None = None
    request_count: int = Field(ge=0)
    latency: GroundingLatency
