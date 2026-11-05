"""Verify Phase 11 dataset contracts, source resolution, and review gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from pydantic import ValidationError

from app.rag.evaluation.dataset import EvaluationQuestion, load_evaluation_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_MANIFEST = (
    PROJECT_ROOT / "data/rag/evaluation/dataset_manifest_v1.json"
)


def _hash(content: str) -> str:
    """Return the SHA-256 value placed in a temporary manifest."""
    return hashlib.sha256(content.encode()).hexdigest()


def _answer_record(**overrides: object) -> dict[str, object]:
    """Build one valid approved record for focused mutation tests."""
    values: dict[str, object] = {
        "question_id": "TEST-Q01",
        "query": "When does monthly cancellation end access?",
        "tenant_id": "asterdesk",
        "question_type": "direct",
        "expected_behavior": "answer",
        "expected_answer": "Access remains through the paid-through timestamp.",
        "required_facts": ["Access remains through the paid-through timestamp."],
        "gold_evidence": [{
            "tenant_id": "asterdesk",
            "document_id": "AST-CAN-003",
            "document_version": "2.0",
            "section_key": "DECISION",
            "source_reference": "documents/asterdesk/AST-CAN-003_v2-0.md",
            "source_quote": "paid features accessible through the paid-through timestamp",
            "relevance_grade": 3,
        }],
        "annotation_notes": "Human-reviewed test fixture.",
        "review_status": "approved",
        "reviewed_by": "reviewer@example.test",
        "reviewed_at": "2026-11-04T12:00:00Z",
        "dataset_split": "development",
        "leakage_group": "cancellation-current",
    }
    values.update(overrides)
    return values


class TemporaryDataset:
    """Create a minimal frozen source catalog and JSONL dataset for one test."""

    def __init__(self, directory: Path, records: list[dict[str, object]]) -> None:
        """Write deterministic fixtures and expose their manifest path."""
        corpus = (
            "document_id,tenant_id,version,status,visibility\n"
            "AST-CAN-003,asterdesk,2.0,active,public\n"
        )
        sections = (
            "document_id,tenant_id,version,section_key,source_path\n"
            "AST-CAN-003,asterdesk,2.0,DECISION,"
            "documents/asterdesk/AST-CAN-003_v2-0.md\n"
        )
        (directory / "corpus.csv").write_text(corpus, encoding="utf-8")
        (directory / "sections.csv").write_text(sections, encoding="utf-8")
        (directory / "questions.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        questions = (directory / "questions.jsonl").read_text(encoding="utf-8")
        source_directory = directory / "raw/asterdesk"
        source_directory.mkdir(parents=True)
        (source_directory / "AST-CAN-003_v2-0.md").write_text(
            "paid features accessible through the paid-through timestamp\n",
            encoding="utf-8",
        )
        manifest = {
            "dataset_id": "test-dataset",
            "dataset_version": "1.0.0",
            "review_status": "approved",
            "records_path": "questions.jsonl",
            "records_sha256": _hash(questions),
            "corpus_manifest_path": "corpus.csv",
            "corpus_manifest_sha256": _hash(corpus),
            "sections_index_path": "sections.csv",
            "sections_index_sha256": _hash(sections),
            "source_root_path": "raw",
            "allowed_tenants": ["asterdesk"],
            "allowed_splits": ["development", "held_out_test"],
            "notes": "Deterministic unit-test fixture.",
        }
        self.manifest_path = directory / "manifest.json"
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


class EvaluationDatasetTests(unittest.TestCase):
    """Exercise candidate loading and official evaluation safety checks."""

    def test_repository_candidates_resolve_but_are_not_official(self) -> None:
        """Load real candidate anchors while refusing unapproved official use."""
        dataset = load_evaluation_dataset(CANDIDATE_MANIFEST)

        self.assertEqual(len(dataset.questions), 7)
        self.assertEqual(dataset.approved_questions, ())
        with self.assertRaisesRegex(ValueError, "approved dataset manifest"):
            load_evaluation_dataset(CANDIDATE_MANIFEST, official=True)

    def test_approved_record_loads_for_official_evaluation(self) -> None:
        """Permit official use only when manifest, record, and source resolve."""
        with tempfile.TemporaryDirectory() as directory:
            fixture = TemporaryDataset(Path(directory), [_answer_record()])
            dataset = load_evaluation_dataset(fixture.manifest_path, official=True)

        self.assertEqual(dataset.questions[0].question_id, "TEST-Q01")
        self.assertEqual(len(dataset.approved_questions), 1)

    def test_duplicate_ids_and_split_leakage_are_rejected(self) -> None:
        """Prevent ambiguous identity and related examples crossing splits."""
        cases = (
            ([_answer_record(), _answer_record()], "Duplicate question_id"),
            ([
                _answer_record(),
                _answer_record(
                    question_id="TEST-Q02",
                    query="What is the paid-through cancellation rule?",
                    dataset_split="held_out_test",
                ),
            ], "Leakage group crosses"),
        )
        for records, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                fixture = TemporaryDataset(Path(directory), records)
                with self.assertRaisesRegex(ValueError, message):
                    load_evaluation_dataset(fixture.manifest_path)

    def test_unresolved_and_cross_tenant_evidence_are_rejected(self) -> None:
        """Fail before metrics when a gold source is absent or unauthorized."""
        unresolved = _answer_record()
        unresolved["gold_evidence"][0]["section_key"] = "MISSING"  # type: ignore[index]
        cross_tenant = _answer_record()
        cross_tenant["gold_evidence"][0]["tenant_id"] = "brookhaven"  # type: ignore[index]
        cases = (
            (unresolved, "Unresolved source section"),
            (cross_tenant, "Gold evidence tenant"),
        )
        for record, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                fixture = TemporaryDataset(Path(directory), [record])
                with self.assertRaisesRegex(ValueError, message):
                    load_evaluation_dataset(fixture.manifest_path)

    def test_equivalent_sections_must_resolve_and_be_distinct(self) -> None:
        """Reject equivalent sections that are absent from the catalog or repeated."""
        unresolved = _answer_record()
        unresolved["gold_evidence"][0]["equivalent_sections"] = ["FAQ"]  # type: ignore[index]
        duplicate = _answer_record()
        duplicate["gold_evidence"][0]["equivalent_sections"] = ["DECISION"]  # type: ignore[index]
        cases = (
            (unresolved, "Unresolved source section"),
            (duplicate, "equivalent_sections must be unique"),
        )
        for record, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                fixture = TemporaryDataset(Path(directory), [record])
                with self.assertRaisesRegex(ValueError, message):
                    load_evaluation_dataset(fixture.manifest_path)

    def test_unresolved_readable_source_quote_is_rejected(self) -> None:
        """Ensure a catalog-valid anchor cannot carry a fabricated text quote."""
        record = _answer_record()
        record["gold_evidence"][0]["source_quote"] = "quote absent from source"  # type: ignore[index]
        with tempfile.TemporaryDirectory() as directory:
            fixture = TemporaryDataset(Path(directory), [record])
            with self.assertRaisesRegex(ValueError, "Source quote does not resolve"):
                load_evaluation_dataset(fixture.manifest_path)

    def test_answer_and_review_state_contradictions_are_rejected(self) -> None:
        """Require evidence for answers and reviewer metadata only for approval."""
        invalid_answer = _answer_record(gold_evidence=[])
        invalid_review = _answer_record(
            review_status="needs_review", reviewed_by=None, reviewed_at="2026-11-04T12:00:00Z"
        )
        for record, message in (
            (invalid_answer, "Answerable questions require"),
            (invalid_review, "Only approved questions"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                fixture = TemporaryDataset(Path(directory), [record])
                with self.assertRaisesRegex(ValueError, message):
                    load_evaluation_dataset(fixture.manifest_path)

    def test_historical_scope_is_a_separate_question_input(self) -> None:
        """Require scope for historical questions only, independent of gold labels."""
        scope = {"document_id": "AST-CAN-003", "document_version": "1.0"}
        cases = (
            (_answer_record(question_type="historical_version"), "require historical_scope"),
            (_answer_record(historical_scope=scope), "Only historical-version"),
        )
        for record, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValidationError, message):
                    EvaluationQuestion.model_validate(record)

    def test_historical_scope_must_name_an_archived_version(self) -> None:
        """Reject a scope that points at an active or unknown document version."""
        record = _answer_record(
            question_type="historical_version",
            historical_scope={"document_id": "AST-CAN-003", "document_version": "2.0"},
        )
        with tempfile.TemporaryDirectory() as directory:
            fixture = TemporaryDataset(Path(directory), [record])
            with self.assertRaisesRegex(ValueError, "archived document version"):
                load_evaluation_dataset(fixture.manifest_path)

    def test_catalog_drift_is_rejected(self) -> None:
        """Detect corpus changes instead of silently evaluating a new snapshot."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = TemporaryDataset(root, [_answer_record()])
            (root / "corpus.csv").write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Catalog hash"):
                load_evaluation_dataset(fixture.manifest_path)


if __name__ == "__main__":
    unittest.main()
