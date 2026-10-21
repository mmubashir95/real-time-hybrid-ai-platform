# Phase 5 — Production Embedding Generation

## Scope and model contract

Phase 5 converts the accepted Phase 4 chunks into aligned local vectors. It
does not create a PostgreSQL schema, pgvector index, retrieval API, BM25 index,
reranker, or answer generator.

The selected model is `BAAI/bge-small-en-v1.5` at immutable Hugging Face commit
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. The model was selected because it
is a compact, locally executable English retrieval model with 384-dimensional
output and a declared MIT license. The pinned model card and tokenizer config
are the source of truth; changing the commit is an embedding-contract change
and requires regeneration.

The runtime verifies rather than assumes:

- the model output has 384 dimensions;
- the loaded tokenizer identity and model maximum sequence length;
- every prepared chunk fits before any inference begins;
- token counts include model special tokens and use no truncation;
- all returned vectors are finite, nonzero `float32` values;
- normalized vectors have L2 norm approximately 1 (`rtol=atol=1e-4`).

Phase 4 counts tokens with `o200k_base`. BGE uses a BERT tokenizer with a
declared 512-token maximum, so its counts can differ. A Phase 4 chunk being
below 500 tokens is not proof that it fits BGE. An overlength preflight fails
the entire run and reports chunk IDs, both token counts, and the model limit.
It never silently truncates, drops, or splits a chunk.

## Document and query encoding

Document inputs use the normalized Phase 4 `chunk.text` exactly, with no new
normalization, rewriting, summary, or repeated heading. The existing chunk text
already contains relevant heading context where Phase 4 attached it.

Queries use BGE's published retrieval instruction:

```text
Represent this sentence for searching relevant passages:
```

Document and query methods are deliberately separate. The document convention,
query convention, normalization choice, model ID, revision, and Phase 5 config
version are recorded in the report.

## Identity, metadata, and fingerprints

Each matrix row has one JSONL metadata record with the same zero-based row
index. It references the original stable chunk ID and preserves tenant,
document, version, status, access scope, heading path, page range, source file,
source blocks, Phase 4 tokenizer, and chunking config version. The frozen
`DocumentChunk` model is reused and never mutated.

The input fingerprint is:

```text
"emb_" + sha256(canonical_json({
  "prepared_text": exact_document_input,
  "encoding": {
    "document_instruction": ...,
    "encoding_config_version": ...,
    "model_id": ...,
    "model_revision": ...,
    "normalize_embeddings": ...
  }
}))
```

Canonical JSON is UTF-8, key-sorted, and uses compact separators. A relevant
text, weight revision, instruction, normalization, or config-version change
therefore invalidates reuse.

## Running Phase 5

Install the pinned dependencies:

```bash
.venv/bin/pip install -r requirements.txt
```

Run tokenizer compatibility only (recommended first):

```bash
.venv/bin/python scripts/embed_rag_documents.py --check-only
```

Generate all embeddings with the conservative default batch size of 32:

```bash
.venv/bin/python scripts/embed_rag_documents.py
```

Explicit device and batch configuration are supported:

```bash
.venv/bin/python scripts/embed_rag_documents.py --device cpu --batch-size 16
```

There is no silent device fallback. CPU is the default; an unsupported or
unavailable requested device fails model initialization without changing model
identity.

## Artifacts and inspection

The default output root is `data/rag/processed/embeddings/`, already covered by
the generated-data ignore rule. Each successful corpus/model contract is stored
under an immutable `run-<identity>/` directory:

- `vectors.npy`: ordered `float32` matrix, loadable without the model;
- `metadata.jsonl`: ordered chunk-to-row and source metadata mapping;
- `report.json`: model, tokenizer, compatibility, timing, and count data;
- `latest.json`: atomically published pointer written only after validation.

Files are written to a temporary run directory. The writer reloads and checks
the matrix and metadata before atomically renaming the complete run and then
publishing `latest.json`. A failed attempt cannot overwrite a prior immutable
run or advertise an incomplete new run.

Inspect one vector without loading the model:

```bash
.venv/bin/python -c "import json,numpy as np,pathlib; p=pathlib.Path('data/rag/processed/embeddings'); r=json.loads((p/'latest.json').read_text())['run_directory']; d=p/r; v=np.load(d/'vectors.npy',allow_pickle=False); m=json.loads((d/'metadata.jsonl').read_text().splitlines()[0]); print(v.shape,m['chunk_id'],v[0][:8])"
```

## Semantic similarity experiment

Use an existing evaluation question rather than inventing a new benchmark:

```bash
.venv/bin/python scripts/embed_rag_documents.py --question-id AST-REF-001-Q01 --top-k 5
```

The experiment encodes the query with the query instruction and computes cosine
similarity in memory. Candidates are restricted to the question's trusted
tenant, gold document, and gold version. When callers do not explicitly select
a version, the reusable experiment function excludes archived records. Results
show score, chunk ID, document ID, version, section, and a section-anchor match.
A section match is only a coarse check, not proof that the exact gold span was
retrieved. Similarities are ranking signals, not calibrated correctness
probabilities. The synthetic labels still require human review, so a few good
examples do not establish retrieval quality.

Run the existing reviewed-candidate split through a coarse anchor evaluation:

```bash
.venv/bin/python scripts/embed_rag_documents.py --evaluate-split validation
```

This reports Recall@5 and Recall@10 using document + version + section anchors.
Expected-no-answer rows are counted and excluded from recall because recall
does not measure abstention. Normal questions search active records in their
trusted tenant; explicitly historical questions select their archived gold
document/version. These metrics do not claim exact-span retrieval quality.

## Tests

Routine tests use a small deterministic fake encoder and make no model calls:

```bash
.venv/bin/python -m unittest tests.test_rag_embedding_generation
```

Run the opt-in real-model smoke test only when the pinned weights are available
locally or model download is allowed:

```bash
RUN_REAL_EMBEDDING_TEST=1 .venv/bin/python -m unittest tests.test_rag_embedding_generation.RealModelSmokeTests
```

Without that environment variable, the real-model test is explicitly skipped,
not passed. The smoke test verifies dimensions, finite nonzero vectors, query
and document compatibility, and a basic related-sentence similarity.

## Limitations and Phase 6

The current corpus is synthetic and unusually regular. Phase 5 records timings
only from an actual run; this document does not claim unmeasured performance.
The CLI provides only a small exact-vector learning baseline. A formal benchmark
with exact-span resolution, abstention metrics, statistical comparison, and
production authorization testing belongs in the retrieval/evaluation phases.

Phase 6 should add PostgreSQL and pgvector migrations, model/revision/dimension
constraints, transactionally safe ingestion of these aligned artifacts, exact
dense search first, tenant/access/version filters applied before exposure,
index lifecycle handling, and corpus-level retrieval evaluation. It must not
mix vectors produced by different embedding contracts.
