"""Load and validate versioned Phase 11 questions against corpus source catalogs.

This module validates annotation integrity only. It deliberately performs no
retrieval, model inference, database access, or automatic human approval.
"""

from __future__ import annotations

import csv
from datetime import datetime
from enum import Enum
import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QuestionType(str, Enum):
    """Classify the reviewed behavior exercised by one question."""

    DIRECT = "direct"
    PARAPHRASE = "paraphrase"
    MULTI_EVIDENCE = "multi_evidence"
    EXACT_TERM = "exact_term"
    UNANSWERABLE = "unanswerable"
    HISTORICAL_VERSION = "historical_version"
    AUTHORIZATION_PROBE = "authorization_probe"


class ExpectedBehavior(str, Enum):
    """Describe whether the system should answer, abstain, or deny access."""

    ANSWER = "answer"
    ABSTAIN = "abstain"
    DENY = "deny"


class ReviewStatus(str, Enum):
    """Track annotation state without treating generated seeds as ground truth."""

    DRAFT = "draft"
    NEEDS_REVIEW = "needs_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class DatasetSplit(str, Enum):
    """Separate development work from the frozen unbiased test split."""

    DEVELOPMENT = "development"
    HELD_OUT_TEST = "held_out_test"


class GoldEvidence(BaseModel):
    """Identify one stable source section required to answer a question."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    tenant_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    document_version: str = Field(min_length=1)
    section_key: str = Field(min_length=1)
    source_reference: str = Field(min_length=1)
    source_quote: str = Field(min_length=1)
    relevance_grade: int = Field(default=1, ge=1, le=3)


class EvaluationQuestion(BaseModel):
    """Represent one candidate or human-approved golden evaluation question."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question_id: str = Field(pattern=r"^[A-Z0-9][A-Z0-9_-]{2,63}$")
    query: str = Field(min_length=5, max_length=2_000)
    tenant_id: str = Field(min_length=1)
    question_type: QuestionType
    expected_behavior: ExpectedBehavior
    expected_answer: str | None = Field(default=None, min_length=1)
    required_facts: tuple[str, ...] = Field(default=())
    gold_evidence: tuple[GoldEvidence, ...] = Field(default=())
    annotation_notes: str = Field(min_length=1)
    review_status: ReviewStatus
    reviewed_by: str | None = Field(default=None, min_length=1)
    reviewed_at: datetime | None = None
    dataset_split: DatasetSplit
    leakage_group: str = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_label_consistency(self) -> EvaluationQuestion:
        """Reject answer/evidence and review-state contradictions."""
        if self.expected_behavior is ExpectedBehavior.ANSWER:
            if not self.expected_answer or not self.required_facts or not self.gold_evidence:
                raise ValueError(
                    "Answerable questions require an expected answer, required facts, and evidence"
                )
            if any(item.tenant_id != self.tenant_id for item in self.gold_evidence):
                raise ValueError("Gold evidence tenant must match the question tenant")
        elif self.expected_answer is not None or self.required_facts or self.gold_evidence:
            raise ValueError("Abstention and denial questions must not contain answer evidence")

        has_review_identity = self.reviewed_by is not None or self.reviewed_at is not None
        if self.review_status is ReviewStatus.APPROVED:
            if self.reviewed_by is None or self.reviewed_at is None:
                raise ValueError("Approved questions require reviewer identity and timestamp")
        elif has_review_identity:
            raise ValueError("Only approved questions may contain reviewer metadata")
        return self


class DatasetManifest(BaseModel):
    """Pin dataset identity and the source catalogs used for validation."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    review_status: ReviewStatus
    records_path: str = Field(min_length=1)
    records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_manifest_path: str = Field(min_length=1)
    corpus_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sections_index_path: str = Field(min_length=1)
    sections_index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_root_path: str = Field(min_length=1)
    allowed_tenants: tuple[str, ...] = Field(min_length=1)
    allowed_splits: tuple[DatasetSplit, ...] = Field(min_length=1)
    notes: str = Field(min_length=1)

    @model_validator(mode="after")
    def reject_duplicate_scope_values(self) -> DatasetManifest:
        """Keep manifest scope deterministic and unambiguous."""
        if len(self.allowed_tenants) != len(set(self.allowed_tenants)):
            raise ValueError("allowed_tenants must be unique")
        if len(self.allowed_splits) != len(set(self.allowed_splits)):
            raise ValueError("allowed_splits must be unique")
        return self


class EvaluationDataset(BaseModel):
    """Carry a fully validated manifest and its ordered question records."""

    model_config = ConfigDict(frozen=True)

    manifest: DatasetManifest
    questions: tuple[EvaluationQuestion, ...]

    @property
    def approved_questions(self) -> tuple[EvaluationQuestion, ...]:
        """Return only records eligible for official quality reporting."""
        return tuple(
            question
            for question in self.questions
            if question.review_status is ReviewStatus.APPROVED
        )


def _sha256(path: Path) -> str:
    """Return the content identity used to freeze one source catalog."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(64 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_lines(path: Path) -> tuple[EvaluationQuestion, ...]:
    """Parse nonblank JSONL records with line-specific validation errors."""
    questions: list[EvaluationQuestion] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                questions.append(EvaluationQuestion.model_validate_json(line))
            except (ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid question at {path}:{line_number}: {error}") from error
    if not questions:
        raise ValueError("Evaluation dataset must contain at least one question")
    return tuple(questions)


def _load_catalog(path: Path) -> list[dict[str, str]]:
    """Read one repository CSV catalog without weakening its required headers."""
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"Catalog has no header: {path}")
        return list(reader)


def _validate_questions(
    manifest: DatasetManifest,
    questions: tuple[EvaluationQuestion, ...],
    corpus_rows: list[dict[str, str]],
    section_rows: list[dict[str, str]],
    source_root: Path,
    *,
    official: bool,
) -> None:
    """Validate uniqueness, split isolation, approval, and source resolution."""
    question_ids: set[str] = set()
    normalized_queries: set[tuple[str, str]] = set()
    split_by_group: dict[str, DatasetSplit] = {}
    corpus = {
        (row["tenant_id"], row["document_id"], row["version"]): row
        for row in corpus_rows
    }
    sections = {
        (
            row["tenant_id"], row["document_id"], row["version"],
            row["section_key"], row["source_path"],
        )
        for row in section_rows
    }

    for question in questions:
        if question.question_id in question_ids:
            raise ValueError(f"Duplicate question_id: {question.question_id}")
        question_ids.add(question.question_id)

        normalized_query = " ".join(question.query.casefold().split())
        query_key = (question.tenant_id, normalized_query)
        if query_key in normalized_queries:
            raise ValueError(f"Duplicate normalized query: {question.question_id}")
        normalized_queries.add(query_key)

        if question.tenant_id not in manifest.allowed_tenants:
            raise ValueError(f"Question tenant is outside manifest scope: {question.question_id}")
        if question.dataset_split not in manifest.allowed_splits:
            raise ValueError(f"Question split is outside manifest scope: {question.question_id}")

        previous_split = split_by_group.setdefault(
            question.leakage_group, question.dataset_split
        )
        if previous_split is not question.dataset_split:
            raise ValueError(
                f"Leakage group crosses dataset splits: {question.leakage_group}"
            )

        if official and question.review_status is not ReviewStatus.APPROVED:
            raise ValueError(f"Official evaluation requires approval: {question.question_id}")

        for evidence in question.gold_evidence:
            source_key = (evidence.tenant_id, evidence.document_id, evidence.document_version)
            source = corpus.get(source_key)
            if source is None:
                raise ValueError(f"Unresolved document version: {question.question_id}")
            section_key = (*source_key, evidence.section_key, evidence.source_reference)
            if section_key not in sections:
                raise ValueError(f"Unresolved source section: {question.question_id}")
            expected_status = (
                "archived"
                if question.question_type is QuestionType.HISTORICAL_VERSION
                else "active"
            )
            if source["status"] != expected_status:
                raise ValueError(
                    f"Evidence lifecycle does not match question type: {question.question_id}"
                )
            reference = Path(evidence.source_reference)
            if (
                reference.is_absolute()
                or not reference.parts
                or reference.parts[0] != "documents"
                or ".." in reference.parts
            ):
                raise ValueError(f"Unsafe source reference: {question.question_id}")
            source_path = (source_root / Path(*reference.parts[1:])).resolve()
            if not source_path.is_relative_to(source_root) or not source_path.is_file():
                raise ValueError(f"Source file does not resolve: {question.question_id}")
            if source_path.suffix.casefold() in {".md", ".txt", ".html"}:
                source_text = source_path.read_text(encoding="utf-8")
                if evidence.source_quote not in source_text:
                    raise ValueError(f"Source quote does not resolve: {question.question_id}")


def load_evaluation_dataset(
    manifest_path: Path,
    *,
    official: bool = False,
) -> EvaluationDataset:
    """Load one dataset and fail closed on drift or unapproved official labels."""
    manifest_path = manifest_path.resolve()
    manifest = DatasetManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    base_directory = manifest_path.parent
    records_path = (base_directory / manifest.records_path).resolve()
    corpus_path = (base_directory / manifest.corpus_manifest_path).resolve()
    sections_path = (base_directory / manifest.sections_index_path).resolve()
    source_root = (base_directory / manifest.source_root_path).resolve()

    for path, expected_hash in (
        (records_path, manifest.records_sha256),
        (corpus_path, manifest.corpus_manifest_sha256),
        (sections_path, manifest.sections_index_sha256),
    ):
        if _sha256(path) != expected_hash:
            raise ValueError(f"Catalog hash does not match dataset manifest: {path}")

    if official and manifest.review_status is not ReviewStatus.APPROVED:
        raise ValueError("Official evaluation requires an approved dataset manifest")

    questions = _load_json_lines(records_path)
    _validate_questions(
        manifest,
        questions,
        _load_catalog(corpus_path),
        _load_catalog(sections_path),
        source_root,
        official=official,
    )
    return EvaluationDataset(manifest=manifest, questions=questions)
