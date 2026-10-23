#!/usr/bin/env python3
"""Make one explicit paid OpenAI Responses API call for Phase 9 verification."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import ConfigurationError, load_rag_generation_settings  # noqa: E402
from app.rag.generation import (  # noqa: E402
    GenerationPipelineError,
    OpenAIResponsesClient,
)


async def _run() -> None:
    """Issue one small request and print provider accounting for manual review."""
    settings = load_rag_generation_settings()
    result = await OpenAIResponsesClient(settings).generate(
        "Answer in one short sentence.",
        "Reply with: Phase 9 OpenAI smoke test passed.",
        request_id="phase9-smoke",
    )
    print("PHASE 9 — OPENAI SMOKE TEST")
    print(f"Model: {result.model}")
    print(f"Response: {result.answer}")
    if result.usage:
        print(
            f"Tokens: input={result.usage.input_tokens} "
            f"output={result.usage.output_tokens} total={result.usage.total_tokens}"
        )
    if result.cost:
        print(f"Estimated cost (USD): {result.cost.estimated_cost_usd:.10f}")
    print(f"Generation latency: {result.latency_ms:.3f}ms")
    print("FINAL STATUS: PASSED")


def main() -> int:
    """Keep live failures explicit without exposing credentials or SDK payloads."""
    try:
        asyncio.run(_run())
    except (ConfigurationError, GenerationPipelineError, ValueError, OSError) as error:
        print(f"OpenAI smoke test failed: {error}", file=sys.stderr)
        print("FINAL STATUS: FAILED")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
