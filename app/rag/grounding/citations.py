"""Build and validate request-local citation identities from authorized evidence."""

from __future__ import annotations

import re
from collections.abc import Iterable
from types import MappingProxyType
from typing import Mapping

from app.rag.grounding.models import EvidenceRecord, GeneratedClaim
from app.rag.reranking.models import RerankedPassage


# Citations are rendered only by the server; numbered markers, E IDs, or source
# labels inside model text would display provenance the registry never issued.
EMBEDDED_CITATION_PATTERN = re.compile(r"\[\s*\d+\s*\]|\bE\d+\b|\bsources?\s*:", re.IGNORECASE)
# Claim text is pasted into the verifier prompt next to the server's evidence.
# Line breaks (every separator str.splitlines() honors) or the prompt's own
# labels would let a manipulated draft append fake "evidence" for the verifier.
VERIFIER_PROMPT_STRUCTURE_PATTERN = re.compile(
    r"[\n\r\v\f\x1c-\x1e\x85  ]|\bCLAIM\s+\d+\s*:|\bCITED\s+EVIDENCE\b",
    re.IGNORECASE,
)


class EvidenceRegistry:
    """Hold an immutable E-number mapping created from passages sent to the model."""

    def __init__(self, passages: Iterable[RerankedPassage], tenant_id: str) -> None:
        """Validate provenance and assign deterministic IDs in supplied order."""
        records: dict[str, EvidenceRecord] = {}
        chunk_ids: set[str] = set()
        for number, reranked in enumerate(passages, start=1):
            passage = reranked.passage
            if passage.tenant_id != tenant_id:
                raise ValueError("Evidence tenant does not match the trusted request")
            if passage.chunk_id in chunk_ids:
                raise ValueError("Evidence registry cannot contain duplicate chunks")
            if passage.status not in {"active", "archived"}:
                raise ValueError("Evidence registry contains an unauthorized lifecycle status")
            evidence_id = f"E{number}"
            records[evidence_id] = EvidenceRecord(
                evidence_id=evidence_id,
                chunk_id=passage.chunk_id,
                tenant_id=passage.tenant_id,
                document_id=passage.document_id,
                document_version=passage.document_version,
                document_title=passage.title,
                source_reference=passage.source_reference,
                source_format=passage.source_format,
                section=passage.section,
                start_page=passage.start_page,
                end_page=passage.end_page,
                passage_text=passage.text,
            )
            chunk_ids.add(passage.chunk_id)
        self._records: Mapping[str, EvidenceRecord] = MappingProxyType(records)

    @property
    def records(self) -> tuple[EvidenceRecord, ...]:
        """Return registry entries in stable citation order."""
        return tuple(self._records.values())

    def validate_claim(self, claim: GeneratedClaim) -> tuple[EvidenceRecord, ...]:
        """Resolve only IDs present in this request-specific registry.

        Raises ValueError for unknown IDs, claim text carrying its own citations,
        or claim text that could alter the verifier prompt structure.
        """
        if EMBEDDED_CITATION_PATTERN.search(claim.text):
            raise ValueError("Claim text must not contain citation markers or source labels")
        if VERIFIER_PROMPT_STRUCTURE_PATTERN.search(claim.text):
            raise ValueError("Claim text must be one line without verifier prompt labels")
        unknown = [item for item in claim.evidence_ids if item not in self._records]
        if unknown:
            raise ValueError(f"Unknown evidence IDs: {', '.join(unknown)}")
        return tuple(self._records[item] for item in claim.evidence_ids)
