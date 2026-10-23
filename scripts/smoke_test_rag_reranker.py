#!/usr/bin/env python3
"""Run the required real Qwen3 relevance smoke test without a database."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.reranking import Qwen3Reranker, RerankerConfig, RerankerInferenceError  # noqa: E402


QUERY = "Can I get a refund after being charged twice?"
CANDIDATES = (
    "Verified duplicate subscription charges may qualify for a refund.",
    "Employees can change their profile photographs.",
)


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse an explicit device and reproducible local-cache option."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--precision", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    """Execute inference and fail unless the relevant passage ranks first."""
    args = parse_args(arguments)
    config = RerankerConfig(
        device=args.device,
        precision=args.precision,
        batch_size=args.batch_size,
        max_length=args.max_length,
        local_files_only=args.local_files_only,
    )
    scorer = Qwen3Reranker(config)
    prediction_started = perf_counter()
    try:
        scores = scorer.score(QUERY, CANDIDATES)
    except (ValueError, RerankerInferenceError) as error:
        print(f"Smoke test failed: {error}", file=sys.stderr)
        return 1
    prediction_seconds = perf_counter() - prediction_started - scorer.model_load_seconds
    ordered = sorted(range(len(scores)), key=lambda index: (-scores[index].score, index))

    print("PHASE 8 — QWEN3 SMOKE TEST")
    print(f"Model: {config.model_id}@{config.model_revision}")
    print(f"Device/precision: {config.device}/{config.precision}")
    print(f"Query: {QUERY}")
    for index, (candidate, score) in enumerate(zip(CANDIDATES, scores, strict=True), start=1):
        print(f"Candidate {index}: score={score.score} tokens={score.input_tokens} truncated={score.passage_truncated}")
        print(f"  {candidate}")
    print(f"Model load time: {scorer.model_load_seconds:.6f}s")
    print(f"Prediction time: {prediction_seconds:.6f}s")
    if ordered != [0, 1]:
        print("FINAL STATUS: FAILED (relevant candidate did not rank first)")
        return 1
    print("FINAL STATUS: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
