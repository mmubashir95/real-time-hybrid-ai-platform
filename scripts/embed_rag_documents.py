#!/usr/bin/env python3
"""Generate pinned local embeddings from the validated Phase 4 chunk artifact.

The command never reruns or modifies Phases 1–4. It performs a full tokenizer
compatibility preflight before inference and publishes artifacts only after a
successful write/read validation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, TextIO

# Keep direct execution independent of the caller's current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.embeddings import (  # noqa: E402
    EmbeddingConfig,
    SentenceTransformerEncoder,
    generate_embeddings,
    load_chunks,
    run_semantic_experiment,
    write_embedding_artifacts,
)
from app.rag.embeddings.models import (  # noqa: E402
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
)
from app.rag.embeddings.service import (  # noqa: E402
    embedding_run_directory,
    evaluate_recall,
    load_evaluation_question,
    load_reusable_run,
    publish_latest_pointer,
    question_search_scope,
)


DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "rag" / "processed" / "chunks.jsonl"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "rag" / "processed" / "embeddings"
DEFAULT_EVALUATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "rag"
    / "evaluation"
    / "question_bank_SYNTHETIC_REVIEW_REQUIRED.csv"
)


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse reproducible model, inference, artifact, and experiment settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run model-tokenizer compatibility checks without encoding or writing.",
    )
    parser.add_argument(
        "--question-id",
        help="Run one scoped experiment using an existing evaluation question.",
    )
    parser.add_argument("--evaluation-file", type=Path, default=DEFAULT_EVALUATION_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--evaluate-split",
        choices=("train_candidate", "validation", "heldout_test"),
        help="Report coarse evidence-anchor Recall@5 and Recall@10.",
    )
    return parser.parse_args(arguments)


def _print_header(output: TextIO) -> None:
    """Print the stable report title."""
    print("=" * 56, file=output)
    print("              RAG EMBEDDING GENERATION", file=output)
    print("=" * 56, file=output)


def print_compatibility(
    chunks: tuple[Any, ...],
    encoder: SentenceTransformerEncoder,
    output: TextIO,
) -> list[int]:
    """Print measured model-token lengths and fail with chunk-safe details."""
    prepared = [encoder.prepare_document(chunk.text) for chunk in chunks]
    lengths = encoder.count_tokens(prepared)
    overlength = [
        (chunk, length)
        for chunk, length in zip(chunks, lengths, strict=True)
        if length > encoder.model_max_length
    ]
    print("\nINPUT", file=output)
    document_count = len(
        {
            (chunk.tenant_id, chunk.document_id, chunk.document_version)
            for chunk in chunks
        }
    )
    print(f"  Documents: {document_count}", file=output)
    print(f"  Chunks: {len(chunks)}", file=output)
    print(f"  Overlength chunks: {len(overlength)}", file=output)
    print("\nMODEL", file=output)
    print(f"  Model identifier: {encoder.config.model_id}", file=output)
    print(f"  Immutable revision: {encoder.config.model_revision}", file=output)
    print(f"  Dimensions: {encoder.config.expected_dimensions}", file=output)
    print(f"  Device: {encoder.config.device}", file=output)
    print(f"  Input token limit: {encoder.model_max_length}", file=output)
    print(f"  Batch size: {encoder.config.batch_size}", file=output)
    if lengths:
        print("\nTOKEN COMPATIBILITY", file=output)
        print(f"  Minimum model tokens: {min(lengths)}", file=output)
        print(f"  Average model tokens: {sum(lengths) / len(lengths):.2f}", file=output)
        print(f"  Maximum model tokens: {max(lengths)}", file=output)
    if overlength:
        print("\nOVERLENGTH CHUNKS", file=output)
        for chunk, length in overlength:
            print(
                f"  {chunk.chunk_id}: Phase 4={chunk.token_count}, "
                f"embedding model={length}, limit={encoder.model_max_length}",
                file=output,
            )
        raise ValueError(
            "Compatibility blocker: adjust approved chunking or select another "
            "model with owner approval; no text was truncated"
        )
    return lengths


def _generate_and_publish(
    chunks: tuple[Any, ...],
    encoder: SentenceTransformerEncoder,
    corpus_sha256: str,
    output_directory: Path,
    config: EmbeddingConfig,
    output: TextIO,
) -> tuple[Any, tuple[Any, ...]]:
    """Encode every chunk, publish the run, print measurements, return rows."""
    result = generate_embeddings(chunks, encoder, corpus_sha256)
    vectors_path, metadata_path, report_path = write_embedding_artifacts(
        output_directory, result, config
    )
    throughput = (
        len(chunks) / result.encoding_seconds if result.encoding_seconds else 0.0
    )
    print("\nOUTPUT", file=output)
    print(f"  Generated vectors: {len(result.records)}", file=output)
    print("  Failed vectors: 0", file=output)
    print("  Invalid vectors: 0", file=output)
    print(f"  Vectors: {vectors_path}", file=output)
    print(f"  Metadata: {metadata_path}", file=output)
    print(f"  Report: {report_path}", file=output)
    print("\nPERFORMANCE", file=output)
    print(f"  Model loading duration: {result.model_load_seconds:.3f}s", file=output)
    print(f"  Encoding duration: {result.encoding_seconds:.3f}s", file=output)
    print(f"  Chunks per second: {throughput:.2f}", file=output)
    return result.vectors, result.records


def main(
    arguments: list[str] | None = None,
    output: TextIO | None = None,
    error_output: TextIO | None = None,
) -> int:
    """Run compatibility, generation, persistence, and optional experiment."""
    if output is None:
        output = sys.stdout
    if error_output is None:
        error_output = sys.stderr
    _print_header(output)
    try:
        args = parse_args(arguments)
        config = EmbeddingConfig(
            model_id=args.model_id,
            model_revision=args.model_revision,
            batch_size=args.batch_size,
            device=args.device,
        )
        chunks, corpus_sha256 = load_chunks(args.chunks)
        encoder = SentenceTransformerEncoder(config)
        print_compatibility(chunks, encoder, output)
        if args.check_only:
            print("\nFINAL STATUS: PASSED (COMPATIBILITY ONLY)", file=output)
            return 0

        run_directory = embedding_run_directory(
            args.output_directory, corpus_sha256, config
        )
        # The run identity is known before inference, so an identical run is
        # validated and reused instead of re-encoding and failing to publish.
        # Reuse also repairs latest.json if a previous publication failed late.
        if run_directory.exists():
            vectors, records, _ = load_reusable_run(
                run_directory, corpus_sha256, config
            )
            publish_latest_pointer(args.output_directory, run_directory)
            print("\nOUTPUT", file=output)
            print(f"  Reused validated run: {run_directory}", file=output)
            print(f"  Vectors: {len(records)}", file=output)
        else:
            vectors, records = _generate_and_publish(
                chunks, encoder, corpus_sha256, args.output_directory, config, output
            )

        if args.question_id:
            question = load_evaluation_question(args.evaluation_file, args.question_id)
            document_id, document_version = question_search_scope(question)
            ranked = run_semantic_experiment(
                question["query"],
                question["tenant_id"],
                encoder,
                vectors,
                records,
                top_k=args.top_k,
                document_id=document_id,
                document_version=document_version,
            )
            scope = "gold document/version" if document_id else "all active tenant documents"
            print(f"\nSEMANTIC EXPERIMENT: {args.question_id}", file=output)
            print(f"  Search scope: {question['tenant_id']}, {scope}", file=output)
            for rank, item in enumerate(ranked, start=1):
                gold_match = (
                    item["document_id"] == question["gold_document_id"]
                    and item["document_version"] == question["gold_version"]
                    and question["gold_section_key"] in item["section"]
                )
                print(
                    f"  {rank}. score={item['score']:.4f} "
                    f"chunk={item['chunk_id']} document={item['document_id']} "
                    f"version={item['document_version']} section={item['section']} "
                    f"gold_anchor_match={gold_match}",
                    file=output,
                )
        if args.evaluate_split:
            metrics = evaluate_recall(
                args.evaluation_file,
                args.evaluate_split,
                encoder,
                vectors,
                records,
            )
            print(f"\nEVALUATION: {args.evaluate_split}", file=output)
            print(
                f"  Evaluated questions: {metrics['evaluated_questions']}",
                file=output,
            )
            print(
                f"  Skipped no-answer questions: "
                f"{metrics['skipped_no_answer_questions']}",
                file=output,
            )
            print(f"  Recall@5: {metrics['recall_at_5']:.4f}", file=output)
            print(f"  Recall@10: {metrics['recall_at_10']:.4f}", file=output)
            print(f"  Match level: {metrics['match_level']}", file=output)
        print("\nFINAL STATUS: PASSED", file=output)
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Embedding generation failed: {error}", file=error_output)
        print("\nFINAL STATUS: FAILED", file=output)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
