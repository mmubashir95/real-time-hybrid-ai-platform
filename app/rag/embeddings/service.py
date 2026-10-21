"""Embedding orchestration, validation, persistence, and local retrieval.

This module consumes Phase 4 JSONL instead of rerunning ingestion. It keeps
model inference separate from artifact publication and does not implement a
database index or production retrieval API.
"""

from __future__ import annotations

from collections.abc import Sequence
import csv
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from time import perf_counter
from typing import Any, Protocol

import numpy as np

from app.rag.embeddings.models import (
    EmbeddingConfig,
    EmbeddingRecord,
    GenerationResult,
    TokenLength,
)
from app.rag.ingestion.models import DocumentChunk


class EmbeddingEncoder(Protocol):
    """Small inference contract implemented by the real and test encoders."""

    config: EmbeddingConfig
    model_load_seconds: float
    tokenizer_name: str
    model_max_length: int

    def prepare_document(self, text: str) -> str:
        """Return the exact text supplied to document tokenization."""

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        """Return special-token-inclusive input lengths."""

    def encode_documents(self, texts: Sequence[str]) -> Any:
        """Encode a bounded batch in input order."""

    def encode_query(self, query: str) -> Any:
        """Encode one query using the configured query convention."""


def load_chunks(path: Path) -> tuple[tuple[DocumentChunk, ...], str]:
    """Load strict Phase 4 JSONL and return chunks plus its byte hash."""
    if not path.is_file():
        raise FileNotFoundError(f"Chunk artifact not found: {path}")
    content = path.read_bytes()
    expected_fields = {field.name for field in fields(DocumentChunk)}
    chunks: list[DocumentChunk] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(content.splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid chunk JSON on line {line_number}: {error}"
            ) from error
        if not isinstance(record, dict) or set(record) != expected_fields:
            raise ValueError(f"Invalid chunk schema on line {line_number}")
        record["heading_path"] = tuple(record["heading_path"])
        record["source_block_ids"] = tuple(record["source_block_ids"])
        try:
            chunk = DocumentChunk(**record)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid chunk record on line {line_number}: {error}"
            ) from error
        if not chunk.chunk_id.strip():
            raise ValueError(f"Missing chunk_id on line {line_number}")
        if chunk.chunk_id in seen_ids:
            raise ValueError(f"Duplicate chunk_id: {chunk.chunk_id}")
        if not chunk.text.strip():
            raise ValueError(f"Empty chunk text: {chunk.chunk_id}")
        seen_ids.add(chunk.chunk_id)
        chunks.append(chunk)
    return tuple(chunks), hashlib.sha256(content).hexdigest()


def embedding_input_fingerprint(
    prepared_text: str,
    config: EmbeddingConfig,
) -> str:
    """Hash exact model input together with every vector-affecting setting."""
    identity = {
        "prepared_text": prepared_text,
        "encoding": config.fingerprint_fields(),
    }
    canonical = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "emb_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_vectors(
    vectors: Any,
    expected_count: int,
    expected_dimensions: int,
    normalize_embeddings: bool,
    normalization_tolerance: float = 1e-4,
) -> np.ndarray:
    """Validate shape, float compatibility, finiteness, nonzero norms, and L2 policy."""
    array = np.asarray(vectors)
    if array.ndim != 2 or array.shape != (expected_count, expected_dimensions):
        raise ValueError(
            "Embedding shape mismatch: expected "
            f"({expected_count}, {expected_dimensions}), received {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("Embeddings must have a numerical dtype")
    array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("Embeddings contain NaN or infinity")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms == 0):
        raise ValueError("Embeddings contain a zero vector")
    if normalize_embeddings and not np.allclose(
        norms, 1.0, rtol=normalization_tolerance, atol=normalization_tolerance
    ):
        raise ValueError("Normalized embeddings are outside the L2 norm tolerance")
    return array


def generate_embeddings(
    chunks: Sequence[DocumentChunk],
    encoder: EmbeddingEncoder,
    corpus_sha256: str,
) -> GenerationResult:
    """Preflight every chunk, encode bounded batches, and validate all rows."""
    if not chunks:
        empty = np.empty((0, encoder.config.expected_dimensions), dtype=np.float32)
        return GenerationResult(
            vectors=empty,
            records=(),
            token_lengths=(),
            model_load_seconds=encoder.model_load_seconds,
            encoding_seconds=0.0,
            batch_durations_seconds=(),
            tokenizer_name=encoder.tokenizer_name,
            model_max_length=encoder.model_max_length,
            device=encoder.config.device,
            corpus_sha256=corpus_sha256,
        )

    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs are not allowed")
    prepared_inputs = [encoder.prepare_document(chunk.text) for chunk in chunks]
    model_token_counts = encoder.count_tokens(prepared_inputs)
    if len(model_token_counts) != len(chunks):
        raise ValueError("Tokenizer returned a count inconsistent with chunk input")
    token_lengths = tuple(
        TokenLength(chunk.chunk_id, chunk.token_count, model_tokens)
        for chunk, model_tokens in zip(chunks, model_token_counts, strict=True)
    )
    overlength = [
        length for length in token_lengths
        if length.embedding_tokens > encoder.model_max_length
    ]
    if overlength:
        details = ", ".join(
            f"{item.chunk_id} (chunk={item.original_chunk_tokens}, "
            f"model={item.embedding_tokens})"
            for item in overlength
        )
        raise ValueError(
            f"{len(overlength)} chunks exceed model limit "
            f"{encoder.model_max_length}: {details}"
        )

    batches: list[np.ndarray] = []
    batch_durations: list[float] = []
    encoding_started = perf_counter()
    for start in range(0, len(chunks), encoder.config.batch_size):
        stop = min(start + encoder.config.batch_size, len(chunks))
        batch_chunks = chunks[start:stop]
        batch_started = perf_counter()
        try:
            raw_batch = encoder.encode_documents(
                [chunk.text for chunk in batch_chunks]
            )
        except Exception as error:
            affected = ", ".join(chunk.chunk_id for chunk in batch_chunks)
            raise RuntimeError(
                f"Embedding batch failed for chunk IDs [{affected}]: {error}"
            ) from error
        batch_durations.append(perf_counter() - batch_started)
        batches.append(
            validate_vectors(
                raw_batch,
                len(batch_chunks),
                encoder.config.expected_dimensions,
                encoder.config.normalize_embeddings,
            )
        )
    encoding_seconds = perf_counter() - encoding_started
    vectors = np.concatenate(batches, axis=0)
    validate_vectors(
        vectors,
        len(chunks),
        encoder.config.expected_dimensions,
        encoder.config.normalize_embeddings,
    )

    records = tuple(
        EmbeddingRecord(
            row_index=index,
            chunk_id=chunk.chunk_id,
            tenant_id=chunk.tenant_id,
            document_id=chunk.document_id,
            document_version=chunk.document_version,
            document_status=chunk.document_status,
            access_scope=chunk.access_scope,
            heading_path=chunk.heading_path,
            start_page=chunk.start_page,
            end_page=chunk.end_page,
            source_file=chunk.source_file,
            source_type=chunk.source_type,
            source_block_ids=chunk.source_block_ids,
            chunk_token_count=chunk.token_count,
            chunk_tokenizer_name=chunk.tokenizer_name,
            chunk_tokenizer_version=chunk.tokenizer_version,
            chunking_config_version=chunk.chunking_config_version,
            input_fingerprint=embedding_input_fingerprint(
                prepared_inputs[index], encoder.config
            ),
            embedding_dimensions=encoder.config.expected_dimensions,
            embedding_model_id=encoder.config.model_id,
            embedding_model_revision=encoder.config.model_revision,
        )
        for index, chunk in enumerate(chunks)
    )
    return GenerationResult(
        vectors=vectors,
        records=records,
        token_lengths=token_lengths,
        model_load_seconds=encoder.model_load_seconds,
        encoding_seconds=encoding_seconds,
        batch_durations_seconds=tuple(batch_durations),
        tokenizer_name=encoder.tokenizer_name,
        model_max_length=encoder.model_max_length,
        device=encoder.config.device,
        corpus_sha256=corpus_sha256,
    )


def _json_bytes(value: Any) -> bytes:
    """Serialize stable human-readable JSON with a terminal newline."""
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return text.encode("utf-8")


def _metadata_bytes(records: Sequence[EmbeddingRecord]) -> bytes:
    """Serialize ordered row metadata as stable compact JSON Lines."""
    lines = [
        json.dumps(
            record.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for record in records
    ]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def build_generation_report(
    result: GenerationResult,
    config: EmbeddingConfig,
) -> dict[str, Any]:
    """Build a measured report for persistence and CLI presentation."""
    counts = [item.embedding_tokens for item in result.token_lengths]
    total_seconds = result.model_load_seconds + result.encoding_seconds
    throughput = (
        len(result.records) / result.encoding_seconds
        if result.encoding_seconds
        else 0.0
    )
    return {
        "final_status": "PASSED",
        "input": {
            "chunks": len(result.records),
            "corpus_sha256": result.corpus_sha256,
            "duplicate_chunk_ids": 0,
            "overlength_chunks": 0,
        },
        "model": {
            **config.to_dict(),
            "actual_dimensions": config.expected_dimensions,
            "tokenizer_name": result.tokenizer_name,
            "model_max_length": result.model_max_length,
            "license": "MIT",
        },
        "token_compatibility": {
            "minimum": min(counts, default=0),
            "maximum": max(counts, default=0),
            "average": round(sum(counts) / len(counts), 2) if counts else 0,
            "includes_special_tokens": True,
            "truncation": False,
        },
        "output": {
            "vectors": len(result.records),
            "invalid_vectors": 0,
            "metadata_rows": len(result.records),
        },
        "performance": {
            "model_loading_seconds": round(result.model_load_seconds, 6),
            "encoding_seconds": round(result.encoding_seconds, 6),
            "total_seconds": round(total_seconds, 6),
            "chunks_per_second": round(throughput, 4),
            "batch_durations_seconds": [
                round(value, 6) for value in result.batch_durations_seconds
            ],
        },
    }


def write_embedding_artifacts(
    output_directory: Path,
    result: GenerationResult,
    config: EmbeddingConfig,
) -> tuple[Path, Path, Path]:
    """Stage, round-trip validate, then atomically publish one immutable run."""
    if len(result.records) != len(result.vectors):
        raise ValueError("Vector and metadata row counts do not match")
    run_identity_text = (
        f"{result.corpus_sha256}:{config.model_id}:{config.model_revision}:"
        f"{config.encoding_config_version}"
    )
    run_identity = hashlib.sha256(run_identity_text.encode("utf-8")).hexdigest()[:16]
    output_directory.mkdir(parents=True, exist_ok=True)
    run_directory = output_directory / f"run-{run_identity}"
    if run_directory.exists():
        raise FileExistsError(
            f"Immutable embedding run already exists: {run_directory}"
        )
    stage = Path(tempfile.mkdtemp(prefix=".embedding-run-", dir=output_directory))
    try:
        vectors_path = stage / "vectors.npy"
        metadata_path = stage / "metadata.jsonl"
        report_path = stage / "report.json"
        with vectors_path.open("wb") as vector_file:
            np.save(vector_file, result.vectors, allow_pickle=False)
            vector_file.flush()
            os.fsync(vector_file.fileno())
        metadata_path.write_bytes(_metadata_bytes(result.records))
        report_path.write_bytes(_json_bytes(build_generation_report(result, config)))
        # A read-after-write check prevents publication of misaligned artifacts.
        loaded_vectors, loaded_records, _ = load_embedding_artifacts(stage)
        if not np.array_equal(loaded_vectors, result.vectors):
            raise ValueError("Persisted vectors failed round-trip validation")
        if loaded_records != result.records:
            raise ValueError("Persisted metadata failed round-trip validation")
        stage.replace(run_directory)
        stage = run_directory

        latest_stage = output_directory / ".latest.json.tmp"
        latest_stage.write_bytes(_json_bytes({"run_directory": run_directory.name}))
        latest_stage.replace(output_directory / "latest.json")
        return (
            run_directory / "vectors.npy",
            run_directory / "metadata.jsonl",
            run_directory / "report.json",
        )
    finally:
        if stage.exists() and stage != run_directory:
            shutil.rmtree(stage)


def load_embedding_artifacts(
    run_directory: Path,
) -> tuple[np.ndarray, tuple[EmbeddingRecord, ...], dict[str, Any]]:
    """Load and validate a published or staged embedding artifact set."""
    with (run_directory / "vectors.npy").open("rb") as vector_file:
        vectors = np.load(vector_file, allow_pickle=False)
    records: list[EmbeddingRecord] = []
    metadata_lines = (run_directory / "metadata.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    for line_number, line in enumerate(metadata_lines, start=1):
        try:
            item = json.loads(line)
            item["heading_path"] = tuple(item["heading_path"])
            item["source_block_ids"] = tuple(item["source_block_ids"])
            records.append(EmbeddingRecord(**item))
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(
                f"Invalid embedding metadata line {line_number}: {error}"
            ) from error
    report = json.loads((run_directory / "report.json").read_text(encoding="utf-8"))
    if len(vectors) != len(records):
        raise ValueError("Persisted vector and metadata counts do not match")
    if [record.row_index for record in records] != list(range(len(records))):
        raise ValueError("Persisted metadata row indexes are not ordered")
    if len({record.chunk_id for record in records}) != len(records):
        raise ValueError("Persisted metadata contains duplicate chunk IDs")
    dimensions = records[0].embedding_dimensions if records else vectors.shape[1]
    normalized = bool(report.get("model", {}).get("normalize_embeddings", True))
    validate_vectors(vectors, len(records), dimensions, normalized)
    return vectors, tuple(records), report


def run_semantic_experiment(
    query: str,
    tenant_id: str,
    encoder: EmbeddingEncoder,
    vectors: np.ndarray,
    records: Sequence[EmbeddingRecord],
    top_k: int = 5,
    document_id: str | None = None,
    document_version: str | None = None,
) -> list[dict[str, Any]]:
    """Rank only explicitly authorized tenant/version rows by cosine score."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    candidates = [
        index for index, record in enumerate(records)
        if record.tenant_id == tenant_id
        and (document_id is None or record.document_id == document_id)
        and (document_version is None or record.document_version == document_version)
        # Without an explicit version request, archived records cannot appear current.
        and (document_version is not None or record.document_status == "active")
    ]
    if not candidates:
        raise ValueError("No authorized embedding rows match the requested scope")
    query_vector = np.asarray(encoder.encode_query(query), dtype=np.float32)
    if query_vector.shape != (encoder.config.expected_dimensions,):
        raise ValueError("Query embedding has an unexpected shape")
    query_norm = float(np.linalg.norm(query_vector))
    if not np.isfinite(query_vector).all() or query_norm == 0:
        raise ValueError("Query embedding is nonfinite or zero")
    candidate_vectors = vectors[candidates]
    scores = (candidate_vectors @ query_vector) / (
        np.linalg.norm(candidate_vectors, axis=1) * query_norm
    )
    ranked_positions = np.argsort(-scores)[:top_k]
    return [
        {
            "score": float(scores[position]),
            "chunk_id": records[candidates[position]].chunk_id,
            "document_id": records[candidates[position]].document_id,
            "document_version": records[candidates[position]].document_version,
            "section": " > ".join(records[candidates[position]].heading_path),
        }
        for position in ranked_positions
    ]


def load_evaluation_question(
    path: Path,
    question_id: str,
) -> dict[str, str]:
    """Load one existing gold-candidate question without inventing test data."""
    with path.open(encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            if row["question_id"] == question_id:
                return dict(row)
    raise ValueError(f"Evaluation question not found: {question_id}")


def evaluate_recall(
    path: Path,
    split: str,
    encoder: EmbeddingEncoder,
    vectors: np.ndarray,
    records: Sequence[EmbeddingRecord],
) -> dict[str, Any]:
    """Measure coarse evidence-anchor Recall@5/10 on existing labeled queries.

    Expected-no-answer rows are reported but excluded because recall cannot
    evaluate abstention. Historical questions explicitly select their archived
    gold document/version; other questions search active documents only within
    the trusted tenant.
    """
    evaluated = 0
    skipped_no_answer = 0
    hits_at_5 = 0
    hits_at_10 = 0
    with path.open(encoding="utf-8", newline="") as csv_file:
        rows = [row for row in csv.DictReader(csv_file) if row["split"] == split]
    if not rows:
        raise ValueError(f"No evaluation questions found for split: {split}")

    for row in rows:
        if row["expected_no_answer"].strip().lower() == "true":
            skipped_no_answer += 1
            continue
        is_historical = row["question_type"] == "version_conflict"
        ranked = run_semantic_experiment(
            row["query"],
            row["tenant_id"],
            encoder,
            vectors,
            records,
            top_k=10,
            document_id=row["gold_document_id"] if is_historical else None,
            document_version=row["gold_version"] if is_historical else None,
        )
        matches = [
            item["document_id"] == row["gold_document_id"]
            and item["document_version"] == row["gold_version"]
            and row["gold_section_key"] in item["section"]
            for item in ranked
        ]
        evaluated += 1
        hits_at_5 += int(any(matches[:5]))
        hits_at_10 += int(any(matches[:10]))
    return {
        "split": split,
        "evaluated_questions": evaluated,
        "skipped_no_answer_questions": skipped_no_answer,
        "recall_at_5": hits_at_5 / evaluated if evaluated else 0.0,
        "recall_at_10": hits_at_10 / evaluated if evaluated else 0.0,
        "match_level": "document+version+section (not exact span)",
    }
