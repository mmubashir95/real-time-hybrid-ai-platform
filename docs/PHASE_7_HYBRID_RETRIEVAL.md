# Phase 7 — Hybrid Retrieval

## Status

The code path is implemented for keyword, semantic, and hybrid retrieval. Live
quality numbers are intentionally not recorded here: the checked-in question
bank is marked `synthetic_seed_needs_review`, and the final database-backed
commands are reserved for the project owner to run.

## Architecture

`RetrievalService.search()` accepts one typed request and returns the same
`RetrievedPassage` structure in every mode. The trusted caller context owns the
tenant, visibility, and optional exact historical-version authorization. User
filters can only narrow that scope.

- Keyword mode loads authorized chunks from PostgreSQL, constructs a bounded
  in-memory Okapi BM25 index, and ranks positive matches.
- Semantic mode reuses the Phase 5 BGE query encoder and Phase 6 exact pgvector
  cosine query.
- Hybrid mode retrieves independent candidate lists and combines their ranks
  using reciprocal rank fusion (RRF).

There is no public HTTP endpoint in this phase because the application does not
yet expose an authenticated principal-to-tenant authorization mechanism. The
CLI constructs an explicitly internal development context.

## BM25 design

The implementation uses genuine Okapi BM25 with `k1=1.5` and `b=0.75`. This
avoids another dependency and is appropriate for the current 806-chunk bounded
corpus. Tokens are case-folded alphanumeric terms. Hyphenated identifiers such
as `AST-REF-001` are indexed both whole and by component, which supports exact
policy lookup and ordinary word lookup.

The lexical index is rebuilt from the current database scope per search. This is
simple and makes additions, updates, deletions, version publication, tenant
scope, and visibility changes immediately consistent. It is not suitable for a
large corpus; at larger scale, use a maintained shared BM25 engine or a cached
index keyed by an immutable corpus snapshot and authorization scope.

BM25 scores are relevance scores, not probabilities or confidence values.

## Semantic design

The query is encoded by `BAAI/bge-small-en-v1.5` at pinned revision
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, with normalized 384-dimensional
vectors and the Phase 5 query instruction. PostgreSQL compares it with compatible
stored vectors using exact cosine distance (`<=>`). Exact search is the
correctness baseline for this corpus size.

The SQL query applies tenant, visibility, lifecycle, document, section, and
effective-date constraints before ranking and before passage text leaves the
database.

## Hybrid design

Raw BM25 and cosine scores are on unrelated scales, so they are never added.
RRF contributes `1 / (60 + rank)` from each branch. Candidates are deduplicated
by stable chunk ID, retain both branch ranks/scores, and use branch ranks plus
chunk ID for deterministic tie-breaking. A branch failure raises a retrieval
error; it is not silently treated as an empty result.

## Evaluation

The evaluator uses source anchors `(tenant, document, version, section)` rather
than chunk IDs. It calculates Recall@5/10/20, MRR@10, binary-label NDCG@10,
p50/p95 latency, empty-result rate, and cross-tenant violations. No-answer rows
are excluded from relevance metrics but remain represented by the empty-result
rate. The report records a dataset checksum and retrieval configuration.

By default, evaluation refuses the current unreviewed labels. The override is
explicit and labels the output `EXPLORATORY_UNREVIEWED`; such results must not be
presented as a final benchmark.

## Commands for the project owner

Search (replace the fingerprint with the imported Phase 5 profile):

```bash
.venv/bin/python scripts/search_rag_documents.py \
  --query "duplicate payment policy" --mode hybrid --tenant asterdesk \
  --k 10 --config-fingerprint <64-character-fingerprint>
```

Evaluation after human label review:

```bash
.venv/bin/python scripts/evaluate_rag_retrieval.py \
  --split heldout_test --config-fingerprint <64-character-fingerprint>
```

Use `--allow-unreviewed-labels` only for an exploratory run of the current seed
bank. Both commands work independently of the current working directory.

## Known limitations

- BM25 rebuild cost grows linearly with every authorized chunk.
- Tokenization is intentionally language-simple and has no stemming or synonyms.
- One shared asyncpg connection means hybrid branches execute sequentially.
- Semantic search requires the PostgreSQL/pgvector corpus and local model files.
- Current labels need human review before defensible comparative claims.
- No approximate vector index is introduced; exact cosine search is deliberate.
