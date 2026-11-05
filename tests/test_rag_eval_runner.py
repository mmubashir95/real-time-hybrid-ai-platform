"""Verify Phase 11 aggregation, fixed candidates, and artifact reporting."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from app.rag.evaluation.reporting import (
    EvaluationArtifact,
    RunMetadata,
    read_artifact,
    render_markdown_report,
    write_artifact,
)
from app.rag.evaluation.runner import (
    CandidateRecord,
    EvaluationConfiguration,
    aggregate_question_results,
    assert_fixed_candidate_set,
    build_question_result,
)


def candidate(identifier: str, rank: int, *, relevant: int = 0) -> CandidateRecord:
    """Build one source-traceable candidate fixture."""
    return CandidateRecord(
        rank=rank, chunk_id=f"chk_{identifier}", tenant_id="asterdesk",
        document_id=f"DOC-{identifier}", document_version="2.0",
        section_key="DECISION", relevance_grade=relevant,
    )


class EvaluationRunnerTests(unittest.TestCase):
    """Test deterministic per-question and aggregate calculations."""

    def test_question_and_aggregate_results_keep_sample_counts(self) -> None:
        """Aggregate two questions without hiding a failed observation."""
        a = candidate("A", 1, relevant=1)
        gold = {a.evidence_key: 1}
        complete = build_question_result(
            "Q1", EvaluationConfiguration.HYBRID, (a,), gold, 10.0
        )
        failed = build_question_result(
            "Q2", EvaluationConfiguration.HYBRID, (), {}, 5.0,
            error_code="retrieval_error",
        )
        aggregate = aggregate_question_results(
            EvaluationConfiguration.HYBRID, (complete, failed)
        )
        self.assertEqual(aggregate.question_count, 2)
        self.assertEqual(aggregate.failed_question_count, 1)
        self.assertEqual(aggregate.recall_at_5, 1.0)
        self.assertEqual(aggregate.latency_p50_ms, 10.0)

    def test_reranker_must_preserve_identical_candidate_set(self) -> None:
        """Allow reordering but reject introduced or missing passages."""
        baseline = (candidate("A", 1), candidate("B", 2))
        reranked = (candidate("B", 1), candidate("A", 2))
        assert_fixed_candidate_set(baseline, reranked)
        with self.assertRaisesRegex(ValueError, "preserve"):
            assert_fixed_candidate_set(baseline, (candidate("A", 1),))

    def test_artifact_round_trip_and_missing_measurement_labels(self) -> None:
        """Persist JSON atomically and render absent results honestly."""
        artifact = EvaluationArtifact(
            metadata=RunMetadata(
                run_id="run-1", timestamp="2026-11-04T12:00:00Z",
                dataset_id="dataset", dataset_version="1.0.0",
                dataset_split="development", dataset_sha256="a" * 64,
                corpus_snapshot="manifest-sha", index_version="index-1",
                embedding_model="embedding", evaluation_configuration="E",
                candidate_limit=20, test_environment="unit-test",
            ),
            limitations=("Human review pending.",),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            write_artifact(artifact, path)
            loaded = read_artifact(path)
        report = render_markdown_report(loaded)
        self.assertEqual(loaded.metadata.run_id, "run-1")
        self.assertIn("NOT MEASURED", report)
        self.assertIn("PENDING HUMAN REVIEW", report)


if __name__ == "__main__":
    unittest.main()
