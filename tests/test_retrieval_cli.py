"""Verify retrieval CLI validation and exit codes without a database."""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
import unittest.mock


ROOT = Path(__file__).resolve().parent.parent


def load_script(name: str):
    """Import one script module by path without executing its main guard."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


search_cli = load_script("search_rag_documents")
evaluate_cli = load_script("evaluate_rag_retrieval")


class SearchCliTests(unittest.TestCase):
    """Reject invalid trusted scope or configuration before retrieval."""

    def run_cli(self, *arguments: str) -> tuple[int, str, str]:
        """Run the search CLI with an explicitly empty database URL."""
        output, errors = io.StringIO(), io.StringIO()
        code = search_cli.main(
            ["--query", "refund", "--tenant", "asterdesk", "--database-url", "", *arguments],
            output,
            errors,
        )
        return code, output.getvalue(), errors.getvalue()

    def test_semantic_and_hybrid_require_config_fingerprint(self) -> None:
        """Fail before loading the model when the vector profile is unknown."""
        for mode in ("semantic", "hybrid"):
            code, output, errors = self.run_cli("--mode", mode)
            self.assertEqual(code, 1, mode)
            self.assertIn("--config-fingerprint", errors)
            self.assertIn("FINAL STATUS: FAILED", output)

    def test_historical_scope_requires_explicit_authorization(self) -> None:
        """Refuse archived access without both identifiers and acknowledgement."""
        for arguments in (
            ("--historical-document-id", "AST-REF-001", "--historical-version", "1.0"),
            ("--historical-document-id", "AST-REF-001", "--authorize-historical"),
            ("--authorize-historical",),
        ):
            code, _, errors = self.run_cli("--mode", "keyword", *arguments)
            self.assertEqual(code, 1, arguments)
            self.assertIn("Historical", errors)

    def test_invalid_request_bounds_fail_with_exit_code_one(self) -> None:
        """Convert contract validation errors into a controlled failure."""
        code, _, errors = self.run_cli("--mode", "keyword", "--k", "0")
        self.assertEqual(code, 1)
        self.assertIn("k must be between", errors)

    def test_database_url_is_required_and_must_be_postgresql(self) -> None:
        """Report missing or non-PostgreSQL configuration without connecting."""
        code, _, errors = self.run_cli("--mode", "keyword")
        self.assertEqual(code, 1)
        self.assertIn("DATABASE_URL", errors)
        code, _, errors = self.run_cli(
            "--mode", "keyword", "--database-url", "mysql://user@localhost/db"
        )
        self.assertEqual(code, 1)
        self.assertIn("PostgreSQL", errors)

    def test_unknown_mode_is_rejected_by_argument_parser(self) -> None:
        """Expose only the three supported retrieval modes."""
        with self.assertRaises(SystemExit), unittest.mock.patch("sys.stderr", io.StringIO()):
            search_cli.parse_args(["--query", "q", "--tenant", "t", "--mode", "fuzzy"])


class EvaluateCliTests(unittest.TestCase):
    """Refuse unreviewed labels unless exploratory results are requested."""

    def test_unreviewed_labels_are_refused_and_no_report_is_written(self) -> None:
        """Fail before any database access and leave the output path untouched."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            with unittest.mock.patch("sys.stderr", io.StringIO()) as errors:
                code = evaluate_cli.main([
                    "--config-fingerprint", "c" * 64, "--database-url", "",
                    "--output", str(output),
                ])
            self.assertEqual(code, 1)
            self.assertIn("not reviewed", errors.getvalue())
            self.assertFalse(output.exists())

    def test_unknown_split_fails(self) -> None:
        """Report an empty split instead of producing empty metrics."""
        with unittest.mock.patch("sys.stderr", io.StringIO()) as errors:
            code = evaluate_cli.main([
                "--config-fingerprint", "c" * 64, "--database-url", "",
                "--split", "missing-split", "--allow-unreviewed-labels",
                "--output", str(Path(tempfile.gettempdir()) / "unused-report.json"),
            ])
        self.assertEqual(code, 1)
        self.assertIn("No questions found", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
