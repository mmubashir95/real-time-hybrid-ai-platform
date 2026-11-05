"""Validate independent human judgments and aggregate answer-quality metrics."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.rag.evaluation.metrics import rate


class HumanSupport(str, Enum):
    """Record a human judgment without inheriting the system verifier label."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class AbstentionOutcome(str, Enum):
    """Classify system answer/abstention behavior against reviewed expectation."""

    CORRECT_ANSWER = "correct_answer"
    CORRECT_ABSTENTION = "correct_abstention"
    INCORRECT_ANSWER = "incorrect_answer"
    INCORRECT_ABSTENTION = "incorrect_abstention"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


class ClaimReview(BaseModel):
    """Capture one independent human judgment for a generated factual claim."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question_id: str = Field(min_length=1)
    claim_index: int = Field(ge=0)
    claim_text: str = Field(min_length=1)
    cited_evidence_ids: tuple[str, ...]
    system_verifier_status: str = Field(min_length=1)
    human_support: HumanSupport
    answer_correct: bool
    answer_complete: bool
    answer_relevant: bool
    citation_relationships: tuple[bool, ...]
    reviewer_explanation: str = Field(min_length=1)
    reviewed_by: str = Field(min_length=1)
    reviewed_at: datetime

    @model_validator(mode="after")
    def align_citation_judgments(self) -> ClaimReview:
        """Require one human support judgment for every cited relationship."""
        if len(self.citation_relationships) != len(self.cited_evidence_ids):
            raise ValueError("citation_relationships must align with cited_evidence_ids")
        if len(set(self.cited_evidence_ids)) != len(self.cited_evidence_ids):
            raise ValueError("cited_evidence_ids must be unique")
        return self


class QuestionReview(BaseModel):
    """Record reviewed answerability and system outcome for one question."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question_id: str = Field(min_length=1)
    abstention_outcome: AbstentionOutcome
    grounding_failed: bool = False
    reviewer_explanation: str = Field(min_length=1)
    reviewed_by: str = Field(min_length=1)
    reviewed_at: datetime


class AnswerQualitySummary(BaseModel):
    """Expose answer metrics together with every human-review denominator."""

    model_config = ConfigDict(frozen=True)

    reviewed_questions: int
    reviewed_claims: int
    reviewed_citation_relationships: int
    supported_claims: int
    unsupported_claims: int
    correct_citations: int
    correct_answer_claims: int
    complete_answer_claims: int
    relevant_answer_claims: int
    correct_abstention_decisions: int
    grounding_failures: int
    claim_support_rate: float | None
    unsupported_claim_rate: float | None
    citation_correctness: float | None
    answer_correctness: float | None
    answer_completeness: float | None
    answer_relevance: float | None
    abstention_accuracy: float | None
    grounding_failure_rate: float | None


def summarize_answer_reviews(
    claims: tuple[ClaimReview, ...],
    questions: tuple[QuestionReview, ...],
) -> AnswerQualitySummary:
    """Aggregate independent review labels without treating model output as truth."""
    claim_keys = [(item.question_id, item.claim_index) for item in claims]
    if len(claim_keys) != len(set(claim_keys)):
        raise ValueError("Claim reviews must have unique question/index identities")
    question_ids = [item.question_id for item in questions]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Question reviews must have unique question IDs")

    supported = sum(item.human_support is HumanSupport.SUPPORTED for item in claims)
    unsupported = len(claims) - supported
    citation_count = sum(len(item.citation_relationships) for item in claims)
    correct_citations = sum(sum(item.citation_relationships) for item in claims)
    correct_outcomes = sum(
        item.abstention_outcome
        in {AbstentionOutcome.CORRECT_ANSWER, AbstentionOutcome.CORRECT_ABSTENTION}
        for item in questions
    )
    grounding_failures = sum(item.grounding_failed for item in questions)
    return AnswerQualitySummary(
        reviewed_questions=len(questions),
        reviewed_claims=len(claims),
        reviewed_citation_relationships=citation_count,
        supported_claims=supported,
        unsupported_claims=unsupported,
        correct_citations=correct_citations,
        correct_answer_claims=sum(item.answer_correct for item in claims),
        complete_answer_claims=sum(item.answer_complete for item in claims),
        relevant_answer_claims=sum(item.answer_relevant for item in claims),
        correct_abstention_decisions=correct_outcomes,
        grounding_failures=grounding_failures,
        claim_support_rate=rate(supported, len(claims)),
        unsupported_claim_rate=rate(unsupported, len(claims)),
        citation_correctness=rate(correct_citations, citation_count),
        answer_correctness=rate(sum(item.answer_correct for item in claims), len(claims)),
        answer_completeness=rate(sum(item.answer_complete for item in claims), len(claims)),
        answer_relevance=rate(sum(item.answer_relevant for item in claims), len(claims)),
        abstention_accuracy=rate(correct_outcomes, len(questions)),
        grounding_failure_rate=rate(grounding_failures, len(questions)),
    )
