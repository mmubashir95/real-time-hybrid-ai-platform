"""Verify the Phase 11 aggregation CLI without live RAG infrastructure."""

from __future__ import annotations

import importlib.util
import io
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def _load_cli():
    """Import the script without invoking its command-line main guard."""
    spec = importlib.util.spec_from_file_location(
        "evaluate_rag_phase11", ROOT / "scripts/evaluate_rag.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLI = _load_cli()


def _arguments(output: Path) -> list[str]:
    """Build the required reproducibility arguments for one candidate run."""
    return [
        "--split", "development",
        "--configuration", "C",
        "--output-directory", str(output),
        "--corpus-snapshot", "manifest-sha",
        "--index-version", "index-v1",
        "--embedding-model", "embedding-model@revision",
        "--test-environment", "unit-test",
    ]


class EvaluationCliTests(unittest.TestCase):
    """Exercise approval gates and durable pending-result reports."""

    def test_official_run_rejects_unapproved_dataset_before_writing(self) -> None:
        """Prevent candidate labels from becoming official metrics."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("sys.stderr", io.StringIO()) as errors:
                code = CLI.main(_arguments(output))
            self.assertEqual(code, 1)
            self.assertIn("approved dataset manifest", errors.getvalue())
            self.assertEqual(list(output.iterdir()), [])

    def test_retrieval_details_persist_ranks_rrf_components_and_text_hash(self) -> None:
        """Make hybrid diagnostics readable from the artifact without reconstruction."""
        passage = SimpleNamespace(
            final_rank=3, bm25_rank=2, bm25_score=7.5, semantic_rank=None,
            semantic_score=None, fusion_score=1 / 62, text="Paid features remain.",
        )
        details = CLI._retrieval_details(passage, 60)
        self.assertEqual(details["retrieval_rank"], 3)
        self.assertAlmostEqual(details["bm25_rrf_component"], 1 / 62)
        self.assertIsNone(details["semantic_rrf_component"])
        self.assertEqual(
            details["text_sha256"], hashlib.sha256(b"Paid features remain.").hexdigest()
        )
        # Without fusion there are no RRF components to report.
        self.assertIsNone(CLI._retrieval_details(passage, None)["bm25_rrf_component"])

    def test_live_reranked_run_requires_candidate_baseline(self) -> None:
        """Refuse a D run that could silently use different candidates than C."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            arguments = _arguments(output)
            arguments[arguments.index("C")] = "D"
            with patch("sys.stderr", io.StringIO()) as errors:
                code = CLI.main([*arguments, "--exploratory", "--execute-live"])
            self.assertEqual(code, 1)
            self.assertIn("--candidate-baseline", errors.getvalue())
            self.assertEqual(list(output.iterdir()), [])

    def test_exploratory_run_writes_pending_json_and_markdown(self) -> None:
        """Create honest artifacts without fabricating unsupplied measurements."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            code = CLI.main([*_arguments(output), "--exploratory"])
            run_directories = list(output.iterdir())
            self.assertEqual(code, 0)
            self.assertEqual(len(run_directories), 1)
            report = (run_directories[0] / "report.md").read_text(encoding="utf-8")
            self.assertTrue((run_directories[0] / "run.json").is_file())
            self.assertIn("NOT MEASURED", report)
            self.assertIn("PENDING HUMAN REVIEW", report)


if __name__ == "__main__":
    unittest.main()
