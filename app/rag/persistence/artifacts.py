"""Map validated Phase 4/5 artifacts into immutable database import records.

This module performs every file, checksum, ownership, provenance, and embedding
compatibility check before the database importer opens a write transaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from app.rag.embeddings.models import (
    DEFAULT_DIMENSIONS,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    ENCODING_CONFIG_VERSION,
    EmbeddingConfig,
)
from app.rag.embeddings.service import (
    embedding_input_fingerprint,
    load_chunks,
    load_embedding_artifacts,
)
from app.rag.ingestion.discovery import validate_manifest
from app.rag.ingestion.models import DocumentChunk


CHUNK_ID_PATTERN = re.compile(r"^chk_[0-9a-f]{64}$")


@dataclass(frozen=True)
class DocumentImport:
    """One tenant-scoped logical document identity."""

    tenant_id: str
    document_id: str


@dataclass(frozen=True)
class VersionImport:
    """One immutable document-version metadata record."""

    tenant_id: str
    document_id: str
    version: str
    title: str
    effective_date: date
    status: str
    visibility: str
    source_sha256: str
    source_path: str
    source_format: str
    synthetic: bool
    topic_family: str
    word_count: int
    pdf_pages: int | None
    section_count: int
    scope_note: str


@dataclass(frozen=True)
class ChunkImport:
    """One stable chunk plus its complete Phase 4 provenance."""

    tenant_id: str
    document_id: str
    document_version: str
    chunk_id: str
    chunk_index: int
    text: str
    token_count: int
    heading_path: tuple[str, ...]
    start_page: int | None
    end_page: int | None
    source_block_ids: tuple[int, ...]
    tokenizer_name: str
    tokenizer_version: str
    chunking_config_version: str
    chunking_configuration: dict[str, Any]
    chunking_config_fingerprint: str


@dataclass(frozen=True)
class EmbeddingImport:
    """One vector row and the exact Phase 5 encoding identity."""

    chunk_id: str
    vector: np.ndarray
    model_id: str
    model_revision: str
    embedding_config_fingerprint: str
    input_fingerprint: str
    document_instruction: str
    query_instruction: str
    normalize_embeddings: bool
    encoding_config_version: str


@dataclass(frozen=True)
class ImportPlan:
    """A fully validated, deterministic database import scope."""

    run_directory: Path
    documents: tuple[DocumentImport, ...]
    versions: tuple[VersionImport, ...]
    chunks: tuple[ChunkImport, ...]
    embeddings: tuple[EmbeddingImport, ...]
    model_id: str
    model_revision: str
    dimensions: int


def resolve_embedding_run(
    embedding_root: Path,
    explicit_run: Path | None = None,
) -> Path:
    """Resolve an explicit run or the safe relative target in latest.json."""
    if explicit_run is not None:
        run_directory = explicit_run.resolve()
    else:
        pointer_path = embedding_root / "latest.json"
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            run_name = pointer["run_directory"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"Invalid embedding latest pointer: {error}") from error
        if (
            not isinstance(run_name, str)
            or not run_name.startswith("run-")
            or Path(run_name).name != run_name
        ):
            raise ValueError("latest.json must name one run-* directory")
        run_directory = (embedding_root / run_name).resolve()
        try:
            run_directory.relative_to(embedding_root.resolve())
        except ValueError as error:
            raise ValueError("Embedding run resolves outside its artifact root") from error
    if not run_directory.is_dir():
        raise FileNotFoundError(f"Embedding run directory not found: {run_directory}")
    return run_directory


def _canonical_fingerprint(value: dict[str, Any], *, ensure_ascii: bool = True) -> str:
    """Hash one canonical JSON object using the documented contract."""
    serialized = json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_chunking_contract(
    report_path: Path,
    chunks_sha256: str,
    expected_chunks: int,
) -> tuple[dict[str, Any], str]:
    """Load and validate the Phase 4 report and configuration fingerprint."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        configuration = report["chunking_configuration"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"Invalid chunking report: {error}") from error
    if report.get("final_status") != "PASSED":
        raise ValueError("Chunking report is not marked PASSED")
    if report.get("chunks_jsonl_sha256") != chunks_sha256:
        raise ValueError("Chunk artifact checksum does not match chunking report")
    if report.get("total_chunks") != expected_chunks:
        raise ValueError("Chunking report count does not match chunk artifact")
    if report.get("total_chunks") != sum(report.get("chunk_counts_per_document", {}).values()):
        raise ValueError("Chunking report document counts do not reconcile")
    if not isinstance(configuration, dict):
        raise ValueError("Chunking configuration must be a JSON object")
    return configuration, _canonical_fingerprint(configuration)


def _embedding_config(report: dict[str, Any]) -> tuple[EmbeddingConfig, str]:
    """Validate the approved model contract and derive its database fingerprint."""
    try:
        model = report["model"]
        config = EmbeddingConfig(
            model_id=model["model_id"],
            model_revision=model["model_revision"],
            expected_dimensions=model["expected_dimensions"],
            batch_size=model.get("batch_size", 1),
            device=model.get("device", "cpu"),
            normalize_embeddings=model["normalize_embeddings"],
            query_instruction=model["query_instruction"],
            document_instruction=model["document_instruction"],
            encoding_config_version=model["encoding_config_version"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid embedding model contract: {error}") from error
    if not isinstance(config.document_instruction, str) or not isinstance(
        config.query_instruction, str
    ):
        raise ValueError("Embedding instructions must be strings")
    if model.get("actual_dimensions") != config.expected_dimensions:
        raise ValueError("Reported actual and expected embedding dimensions differ")
    expected = (
        DEFAULT_MODEL_ID,
        DEFAULT_MODEL_REVISION,
        DEFAULT_DIMENSIONS,
        True,
        ENCODING_CONFIG_VERSION,
    )
    actual = (
        config.model_id,
        config.model_revision,
        config.expected_dimensions,
        config.normalize_embeddings,
        config.encoding_config_version,
    )
    if actual != expected:
        raise ValueError(f"Unsupported embedding contract: {actual}")
    fingerprint = _canonical_fingerprint(
        config.fingerprint_fields(), ensure_ascii=False
    )
    return config, fingerprint


def _version_from_manifest(document: Any) -> VersionImport:
    """Convert one validated manifest row into typed database metadata."""
    metadata = document.metadata
    try:
        synthetic = {"true": True, "false": False}[metadata["synthetic"].lower()]
        pdf_pages = int(metadata["pdf_pages"]) if metadata["pdf_pages"] else None
        version = VersionImport(
            tenant_id=document.tenant_id,
            document_id=document.document_id,
            version=document.version,
            title=document.title,
            effective_date=date.fromisoformat(metadata["effective_date"]),
            status=document.status,
            visibility=metadata["visibility"],
            source_sha256=document.sha256,
            source_path=metadata["path"],
            source_format=document.format,
            synthetic=synthetic,
            topic_family=metadata["topic_family"],
            word_count=int(metadata["word_count"]),
            pdf_pages=pdf_pages,
            section_count=int(metadata["section_count"]),
            scope_note=metadata["scope_note"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Manifest contains invalid typed metadata for "
            f"{document.tenant_id}/{document.document_id}/{document.version}: {error}"
        ) from error
    if version.status not in {"active", "archived"}:
        raise ValueError(f"Unsupported lifecycle status: {version.status}")
    if version.visibility not in {"public", "internal"}:
        raise ValueError(f"Unsupported visibility: {version.visibility}")
    if version.word_count < 0 or version.section_count < 0:
        raise ValueError("Manifest counts must be nonnegative")
    if version.source_format == "pdf" and version.pdf_pages is None:
        raise ValueError("PDF manifest rows must declare pdf_pages")
    if version.source_format != "pdf" and version.pdf_pages is not None:
        raise ValueError("Non-PDF manifest rows must not declare pdf_pages")
    return version


def _verify_chunk_parent(chunk: DocumentChunk, version: VersionImport) -> None:
    """Ensure a chunk preserves its manifest ownership and version metadata."""
    expected = (
        version.status,
        version.visibility,
        version.effective_date.isoformat(),
        version.source_path,
        version.source_format,
    )
    actual = (
        chunk.document_status,
        chunk.access_scope,
        chunk.effective_date,
        chunk.source_file,
        chunk.source_type,
    )
    if actual != expected:
        raise ValueError(f"Chunk {chunk.chunk_id} conflicts with its manifest version")
    if not CHUNK_ID_PATTERN.fullmatch(chunk.chunk_id):
        raise ValueError(f"Invalid stable chunk ID: {chunk.chunk_id}")
    if chunk.chunk_index < 0 or chunk.token_count <= 0 or not chunk.source_block_ids:
        raise ValueError(f"Invalid sequence, token count, or provenance: {chunk.chunk_id}")
    if any(block_id < 0 for block_id in chunk.source_block_ids):
        raise ValueError(f"Negative source block ID: {chunk.chunk_id}")
    pages_absent = chunk.start_page is None and chunk.end_page is None
    pages_valid = (
        chunk.start_page is not None
        and chunk.end_page is not None
        and chunk.start_page > 0
        and chunk.end_page >= chunk.start_page
    )
    if not pages_absent and not pages_valid:
        raise ValueError(f"Invalid page range: {chunk.chunk_id}")
    if version.source_format != "pdf" and not pages_absent:
        raise ValueError(f"Non-PDF chunk contains page bounds: {chunk.chunk_id}")


def _verify_metadata(chunk: DocumentChunk, record: Any) -> None:
    """Verify the Phase 5 row still names the exact source chunk and provenance."""
    expected = (
        chunk.tenant_id,
        chunk.document_id,
        chunk.document_version,
        chunk.document_status,
        chunk.access_scope,
        chunk.heading_path,
        chunk.start_page,
        chunk.end_page,
        chunk.source_file,
        chunk.source_type,
        chunk.source_block_ids,
        chunk.token_count,
        chunk.tokenizer_name,
        chunk.tokenizer_version,
        chunk.chunking_config_version,
    )
    actual = (
        record.tenant_id,
        record.document_id,
        record.document_version,
        record.document_status,
        record.access_scope,
        record.heading_path,
        record.start_page,
        record.end_page,
        record.source_file,
        record.source_type,
        record.source_block_ids,
        record.chunk_token_count,
        record.chunk_tokenizer_name,
        record.chunk_tokenizer_version,
        record.chunking_config_version,
    )
    if actual != expected:
        raise ValueError(f"Embedding metadata conflicts with chunk {chunk.chunk_id}")


def load_import_plan(
    *,
    manifest_path: Path,
    raw_directory: Path,
    chunks_path: Path,
    chunking_report_path: Path,
    embedding_root: Path,
    run_directory: Path | None = None,
) -> ImportPlan:
    """Load all source artifacts and return a fully cross-validated import plan."""
    manifest = validate_manifest(manifest_path, raw_directory)
    if not manifest.passed:
        details = "; ".join(error.message for error in manifest.errors[:5])
        raise ValueError(f"Manifest validation failed: {details}")
    chunks, chunks_sha256 = load_chunks(chunks_path)
    configuration, chunking_fingerprint = _load_chunking_contract(
        chunking_report_path, chunks_sha256, len(chunks)
    )
    selected_run = resolve_embedding_run(embedding_root, run_directory)
    vectors, metadata, embedding_report = load_embedding_artifacts(selected_run)
    if embedding_report.get("final_status") != "PASSED":
        raise ValueError("Embedding report is not marked PASSED")
    if embedding_report.get("input", {}).get("corpus_sha256") != chunks_sha256:
        raise ValueError("Embedding run was produced from another chunk artifact")
    if embedding_report.get("input", {}).get("chunks") != len(chunks):
        raise ValueError("Embedding report input count does not match chunk artifact")
    output_counts = embedding_report.get("output", {})
    if (
        output_counts.get("vectors") != len(vectors)
        or output_counts.get("metadata_rows") != len(metadata)
        or output_counts.get("invalid_vectors") != 0
    ):
        raise ValueError("Embedding report output counts do not match artifacts")
    config, embedding_fingerprint = _embedding_config(embedding_report)

    versions = tuple(_version_from_manifest(document) for document in manifest.documents)
    version_by_identity = {
        (version.tenant_id, version.document_id, version.version): version
        for version in versions
    }
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    chunk_version_identities = {
        (chunk.tenant_id, chunk.document_id, chunk.document_version)
        for chunk in chunks
    }
    if chunk_version_identities != set(version_by_identity):
        raise ValueError("Manifest and chunk document-version identity sets do not match")
    if set(chunk_by_id) != {record.chunk_id for record in metadata}:
        raise ValueError("Chunk and embedding metadata identity sets do not match")

    chunk_rows: list[ChunkImport] = []
    embedding_rows: list[EmbeddingImport] = []
    for record, vector in zip(metadata, vectors, strict=True):
        chunk = chunk_by_id[record.chunk_id]
        version_key = (chunk.tenant_id, chunk.document_id, chunk.document_version)
        version = version_by_identity.get(version_key)
        if version is None:
            raise ValueError(f"Chunk {chunk.chunk_id} has no manifest version")
        _verify_chunk_parent(chunk, version)
        _verify_metadata(chunk, record)
        if (
            chunk.chunking_config_version != configuration.get("config_version")
            or chunk.tokenizer_name != configuration.get("tokenizer_name")
            or chunk.tokenizer_version != configuration.get("tokenizer_version")
        ):
            raise ValueError(f"Chunk {chunk.chunk_id} conflicts with chunking configuration")
        prepared_text = config.document_instruction + chunk.text
        if record.input_fingerprint != embedding_input_fingerprint(prepared_text, config):
            raise ValueError(f"Input fingerprint mismatch for chunk {chunk.chunk_id}")
        chunk_rows.append(
            ChunkImport(
                tenant_id=chunk.tenant_id,
                document_id=chunk.document_id,
                document_version=chunk.document_version,
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                token_count=chunk.token_count,
                heading_path=chunk.heading_path,
                start_page=chunk.start_page,
                end_page=chunk.end_page,
                source_block_ids=chunk.source_block_ids,
                tokenizer_name=chunk.tokenizer_name,
                tokenizer_version=chunk.tokenizer_version,
                chunking_config_version=chunk.chunking_config_version,
                chunking_configuration=configuration,
                chunking_config_fingerprint=chunking_fingerprint,
            )
        )
        embedding_rows.append(
            EmbeddingImport(
                chunk_id=record.chunk_id,
                vector=np.asarray(vector, dtype=np.float32),
                model_id=config.model_id,
                model_revision=config.model_revision,
                embedding_config_fingerprint=embedding_fingerprint,
                input_fingerprint=record.input_fingerprint,
                document_instruction=config.document_instruction,
                query_instruction=config.query_instruction,
                normalize_embeddings=config.normalize_embeddings,
                encoding_config_version=config.encoding_config_version,
            )
        )

    documents = tuple(
        DocumentImport(*identity)
        for identity in sorted(
            {(version.tenant_id, version.document_id) for version in versions}
        )
    )
    return ImportPlan(
        run_directory=selected_run,
        documents=documents,
        versions=versions,
        chunks=tuple(chunk_rows),
        embeddings=tuple(embedding_rows),
        model_id=config.model_id,
        model_revision=config.model_revision,
        dimensions=config.expected_dimensions,
    )
