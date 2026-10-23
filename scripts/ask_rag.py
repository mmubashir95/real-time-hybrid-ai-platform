#!/usr/bin/env python3
"""Answer one authorized RAG question through retrieval, reranking, and OpenAI."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import sys
from typing import TextIO
from uuid import uuid4

import asyncpg
from sqlalchemy.engine import make_url


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import ConfigurationError, load_rag_generation_settings  # noqa: E402
from app.rag.embeddings.encoder import SentenceTransformerEncoder  # noqa: E402
from app.rag.embeddings.models import EmbeddingConfig  # noqa: E402
from app.rag.generation import (  # noqa: E402
    GenerationPipelineError,
    OpenAIResponsesClient,
    RagGenerationService,
)
from app.rag.persistence.lifecycle import EmbeddingProfile  # noqa: E402
from app.rag.reranking import (  # noqa: E402
    Qwen3Reranker,
    RerankerConfig,
    RerankingService,
    RetrievalRerankingService,
)
from app.rag.retrieval import (  # noqa: E402
    DenseRetriever,
    KeywordRetriever,
    RetrievalService,
    SearchContextOrigin,
    SearchFilters,
    SearchMode,
    SearchRequest,
    TrustedSearchContext,
)


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse trusted local scope and bounded model configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--tenant", required=True, help="Trusted local tenant scope.")
    parser.add_argument("--visibility", action="append", choices=("public", "internal"))
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--config-fingerprint", required=True)
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", ""))
    parser.add_argument("--schema", default="public")
    parser.add_argument("--document-id")
    parser.add_argument("--section")
    parser.add_argument("--effective-date-from")
    parser.add_argument("--effective-date-to")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--reranker-device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument(
        "--reranker-precision",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(arguments)


def _database_dsn(value: str) -> str:
    """Return an asyncpg DSN after validating the database scheme."""
    if not value:
        raise ValueError("DATABASE_URL or --database-url is required")
    url = make_url(value)
    if url.drivername not in {"postgresql", "postgresql+asyncpg"}:
        raise ValueError("DATABASE_URL must use PostgreSQL")
    return url.set(drivername="postgresql").render_as_string(hide_password=False)


def _build_request(arguments: argparse.Namespace) -> tuple[SearchRequest, EmbeddingConfig]:
    """Build one hybrid request with scope established by this developer CLI."""
    if arguments.top_k <= 0 or arguments.top_k > arguments.candidate_k:
        raise ValueError("--top-k must be positive and not exceed --candidate-k")
    embedding = EmbeddingConfig(device=arguments.embedding_device)
    profile = EmbeddingProfile(
        embedding.model_id,
        embedding.model_revision,
        arguments.config_fingerprint,
        embedding.normalize_embeddings,
        embedding.encoding_config_version,
        embedding.query_instruction,
    )
    context = TrustedSearchContext(
        arguments.tenant,
        tuple(arguments.visibility or ("public",)),
        SearchContextOrigin.INTERNAL,
    )
    request = SearchRequest(
        arguments.query,
        SearchMode.HYBRID,
        SearchFilters(
            document_id=arguments.document_id,
            section=arguments.section,
            effective_date_from=arguments.effective_date_from,
            effective_date_to=arguments.effective_date_to,
        ),
        arguments.candidate_k,
        context,
        profile,
        arguments.candidate_k,
        arguments.candidate_k,
    )
    return request, embedding


def _print_response(response, output: TextIO) -> None:
    """Print the answer, supplied evidence identities, and safe diagnostics."""
    print("=" * 64, file=output)
    print("PHASE 9 — OPENAI RAG ANSWER", file=output)
    print("=" * 64, file=output)
    print(f"Status: {response.status.value}", file=output)
    print(f"Model: {response.model}", file=output)
    print(f"Answer: {response.answer}", file=output)
    print("\nEVIDENCE SUPPLIED", file=output)
    if not response.evidence:
        print("  (none)", file=output)
    for item in response.evidence:
        section = " > ".join(item.section) or "(root)"
        print(
            f"  {item.rerank_rank}. {item.chunk_id} | {item.document_id} "
            f"v{item.document_version} | {section} | "
            f"pages {item.start_page or '-'}-{item.end_page or '-'}",
            file=output,
        )
        print(f"     Source: {item.source_reference}", file=output)
    print("\nDIAGNOSTICS", file=output)
    print(f"  Context tokens (estimated): {response.context.estimated_context_tokens}", file=output)
    print(f"  Total input tokens (estimated): {response.context.estimated_total_input_tokens}", file=output)
    print(f"  Truncated chunks: {len(response.context.truncated_chunk_ids)}", file=output)
    print(f"  OpenAI request count: {response.request_count}", file=output)
    if response.usage:
        print(
            f"  Actual tokens: input={response.usage.input_tokens} "
            f"cached={response.usage.cached_input_tokens} "
            f"output={response.usage.output_tokens} total={response.usage.total_tokens}",
            file=output,
        )
    if response.cost:
        print(f"  Estimated cost (USD): {response.cost.estimated_cost_usd:.10f}", file=output)
    print(
        f"  Latency ms: retrieval={response.latency.retrieval_ms:.3f} "
        f"reranking={response.latency.reranking_ms:.3f} "
        f"generation={response.latency.generation_ms:.3f} "
        f"total={response.latency.total_ms:.3f}",
        file=output,
    )
    print("FINAL STATUS: PASSED", file=output)


async def _run(arguments: argparse.Namespace, output: TextIO) -> None:
    """Create reusable clients and execute one end-to-end generation request."""
    settings = load_rag_generation_settings()
    request, embedding = _build_request(arguments)
    connection = await asyncpg.connect(_database_dsn(arguments.database_url), timeout=10)
    try:
        retrieval = RetrievalService(
            KeywordRetriever(connection, schema=arguments.schema),
            DenseRetriever(
                connection,
                SentenceTransformerEncoder(embedding),
                schema=arguments.schema,
            ),
        )
        reranker_config = RerankerConfig(
            batch_size=arguments.batch_size,
            max_length=arguments.max_length,
            device=arguments.reranker_device,
            precision=arguments.reranker_precision,
            local_files_only=arguments.local_files_only,
        )
        pipeline = RetrievalRerankingService(
            retrieval,
            RerankingService(Qwen3Reranker(reranker_config)),
        )
        service = RagGenerationService(
            pipeline,
            OpenAIResponsesClient(settings),
            settings,
        )
        response = await service.answer_question(
            request,
            top_k=arguments.top_k,
            request_id=str(uuid4()),
        )
    finally:
        await connection.close()
    _print_response(response, output)


def main(
    arguments: list[str] | None = None,
    output: TextIO = sys.stdout,
    error_output: TextIO = sys.stderr,
) -> int:
    """Run the CLI and map expected errors to safe terminal output."""
    try:
        asyncio.run(_run(parse_args(arguments), output))
    except GenerationPipelineError as error:
        print(f"RAG generation failed ({error.code.value}): {error}", file=error_output)
        print("FINAL STATUS: FAILED", file=output)
        return 1
    except (ConfigurationError, ValueError, TypeError, OSError, asyncpg.PostgresError) as error:
        print(f"RAG generation failed: {error}", file=error_output)
        print("FINAL STATUS: FAILED", file=output)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
