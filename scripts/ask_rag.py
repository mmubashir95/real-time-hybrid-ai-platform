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
from app.rag.grounding import GroundedRagService, GroundingStatus  # noqa: E402
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
    parser.add_argument(
        "--mode",
        choices=("grounded", "ungrounded"),
        default="grounded",
        help="Use Phase 10 verification or retain the Phase 9 comparison path.",
    )
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


def _format_optional_tokens(usage) -> str:
    """Render provider usage without inventing unavailable token values."""
    if usage is None:
        return "unavailable"
    return (
        f"input={usage.input_tokens} cached={usage.cached_input_tokens} "
        f"output={usage.output_tokens} total={usage.total_tokens}"
    )


def _print_grounded_response(response, output: TextIO) -> None:
    """Print Phase 10 claims, citations, grounding decisions, and measurements."""
    print("=" * 64, file=output)
    print("PHASE 10 — GROUNDED RAG ANSWER", file=output)
    print("=" * 64, file=output)
    print(f"Grounding status: {response.status.value}", file=output)
    print(f"Generation model: {response.generation_model}", file=output)
    print(f"Verification model: {response.verification_model or 'not called'}", file=output)
    print("\nANSWER", file=output)
    print(response.answer, file=output)
    print("\nCITATIONS", file=output)
    if not response.citations:
        print("  (none)", file=output)
    for citation in response.citations:
        section = " > ".join(citation.section) or "(root)"
        if citation.start_page is None:
            page = "page unavailable"
        elif citation.end_page in (None, citation.start_page):
            page = f"page {citation.start_page}"
        else:
            page = f"pages {citation.start_page}-{citation.end_page}"
        print(
            f"  [{citation.citation_number}] {citation.document_id} "
            f"v{citation.document_version} | {section} | {page}",
            file=output,
        )
        print(f"      Chunk: {citation.chunk_id} ({citation.evidence_id})", file=output)
    diagnostics = response.grounding
    print("\nGROUNDING DIAGNOSTICS", file=output)
    print(f"  Claims generated: {diagnostics.generated_claim_count}", file=output)
    print(f"  Claims verified: {diagnostics.verified_claim_count}", file=output)
    print(f"  Claims supported: {diagnostics.supported_claim_count}", file=output)
    print(f"  Claims unsupported: {diagnostics.unsupported_claim_count}", file=output)
    print(f"  Claims contradicted: {diagnostics.contradicted_claim_count}", file=output)
    print(f"  Invalid citations: {diagnostics.invalid_citation_count}", file=output)
    print(f"  Limitation: {diagnostics.verification_limitation}", file=output)
    print("\nPERFORMANCE", file=output)
    print(
        f"  Latency ms: retrieval={response.latency.retrieval_ms:.3f} "
        f"reranking={response.latency.reranking_ms:.3f} "
        f"generation={response.latency.generation_ms:.3f} "
        f"verification={response.latency.verification_ms if response.latency.verification_ms is not None else 'not measured'} "
        f"total={response.latency.total_ms:.3f}",
        file=output,
    )
    print("\nUSAGE", file=output)
    print(f"  Generation tokens: {_format_optional_tokens(response.generation_usage)}", file=output)
    print(f"  Verification tokens: {_format_optional_tokens(response.verification_usage)}", file=output)
    known_costs = [
        item.estimated_cost_usd
        for item in (response.generation_cost, response.verification_cost)
        if item is not None
    ]
    cost_label = f"{sum(known_costs):.10f}" if len(known_costs) == response.request_count else "unavailable"
    print(f"  Estimated total API cost (USD): {cost_label}", file=output)
    final = "PASSED" if response.status == GroundingStatus.GROUNDED else "FAILED"
    print(f"FINAL STATUS: {final}", file=output)


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
        client = OpenAIResponsesClient(settings)
        service = (
            GroundedRagService(pipeline, client, settings)
            if arguments.mode == "grounded"
            else RagGenerationService(pipeline, client, settings)
        )
        response = await service.answer_question(
            request,
            top_k=arguments.top_k,
            request_id=str(uuid4()),
        )
    finally:
        await connection.close()
    if arguments.mode == "grounded":
        _print_grounded_response(response, output)
    else:
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
