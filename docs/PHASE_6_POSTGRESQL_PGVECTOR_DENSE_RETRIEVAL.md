# Phase 6 — PostgreSQL, pgvector, and Dense Retrieval

## Scope and system boundary

Phase 6 moves the validated Phase 5 embedding artifacts into PostgreSQL and adds
production-oriented dense retrieval. It provides:

- PostgreSQL 17 with pgvector 0.8.7;
- ordered, checksum-protected database migrations;
- normalized RAG storage with source provenance;
- transactional and idempotent embedding import;
- staged, active, archived, and withdrawn document lifecycle states;
- exact PostgreSQL cosine-distance retrieval;
- tenant, visibility, lifecycle, and embedding-profile filtering before Top-K;
- artifact-to-database integrity reconciliation;
- PostgreSQL-versus-NumPy ranking parity checks;
- Recall@5, Recall@10, and latency reporting.

Phase 6 does not add BM25, hybrid retrieval, reciprocal-rank fusion, reranking,
answer generation, citation rendering, a public retrieval API, row-level
security, or user authentication. The CLI tools accept trusted development and
evaluation scope; they are not proof that an end user is authorized.

## Architecture and data flow

```text
Phase 5 immutable artifacts
  manifest.csv
  chunks.jsonl + chunking_report.json
  vectors.npy + metadata.jsonl + report.json
                     |
                     v
          validation and import plan
                     |
                     v
PostgreSQL 17 + pgvector
  documents
      -> document_versions
          -> document_chunks
              -> chunk_embeddings vector(384)
                     |
                     v
trusted search context -> query encoding -> SQL filters -> cosine ranking
                     |
                     v
source-preserving dense results and evaluation reports
```

The checked-in Phase 5 source scope contains 75 logical documents, 79 document
versions, 806 chunks, and 806 embeddings across AsterDesk and Brookhaven. These
are observed artifact values, not hardcoded database-wide assertions. A database
may legitimately contain additional tenants, runs, document versions, or model
configurations.

The selected embedding space remains:

```text
Model: BAAI/bge-small-en-v1.5
Revision: 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
Dimensions: 384
Normalization: enabled
Encoding contract: phase5-v1
Query instruction: Represent this sentence for searching relevant passages:
```

Changing the model, revision, document instruction, normalization policy, or
encoding version creates a different embedding configuration. Equal dimensions
alone never make two embedding spaces compatible.

## PostgreSQL and pgvector

Compose uses the official pgvector image
`pgvector/pgvector:0.8.7-pg17-bookworm`, pinned by digest. PostgreSQL remains on
major version 17 and keeps the existing `postgres_data` named volume.

Installing pgvector server files and enabling the extension are separate steps.
The image supplies the files; migration 001 runs:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Extensions are database-scoped. Enabling `vector` in `realtime_ai` does not
enable it in `realtime_ai_test`.

Never use `docker compose down -v` to upgrade or test this phase. Preserve the
named volume and create a logical backup before changing a non-disposable
database:

```bash
docker compose exec -T postgres pg_dump \
  -U realtime_ai -d realtime_ai -Fc > realtime_ai_before_phase6.dump
```

## Ordered database migrations

The project uses a small ordered SQL runner instead of Alembic:

| Revision | Source | Responsibility |
|---|---|---|
| `000` | `schema.sql` | Month 1 `analyses` table |
| `001` | `migrations/001_enable_pgvector.sql` | Enable database-scoped pgvector |
| `002` | `migrations/002_create_rag_schema.sql` | Create normalized RAG tables |
| `003` | `migrations/003_add_document_lifecycle.sql` | Add lifecycle and staged completeness |

Successful revisions are recorded in `public.schema_migrations` with their
names, SHA-256 checksums, and application times. The runner rejects unknown
revisions, missing history, modified applied files, an incompatible Month 1
schema, and partial/manual RAG installations.

Run migrations against a directly reachable database:

```bash
export DATABASE_URL='postgresql+asyncpg://USER:PASSWORD@HOST:5432/realtime_ai'
.venv/bin/python -m scripts.migrate_database history
.venv/bin/python -m scripts.migrate_database status
.venv/bin/python -m scripts.migrate_database upgrade
.venv/bin/python -m scripts.migrate_database status
```

Replace the uppercase fields with real values; they are not literal connection
credentials. Applied migrations must never be edited. Corrections require a new
forward migration.

Compose runs the same migration runner through the short-lived `migrate`
service before starting the API:

```text
postgres healthy -> migrate exits successfully -> api starts
```

```bash
docker compose up --build -d
docker compose ps -a
docker compose logs migrate
```

A migration failure prevents API startup. Each revision uses a PostgreSQL
transaction, so unsuccessful DDL is not recorded as applied.

## Database storage contract

### Documents and versions

`documents` stores the immutable tenant-scoped logical identity
`(tenant_id, document_id)`. A trigger rejects reassignment to another tenant or
external identity.

`document_versions` stores version-specific title, effective date, lifecycle,
visibility, source hash/path/format, corpus metadata, and staged completeness.
`(document_fk, version)` is unique, and a partial unique index permits at most
one active version per logical document.

The lifecycle states are:

| Status | Meaning | Current search | Historical search |
|---|---|---:|---:|
| `staged` | Complete candidate awaiting publication | no | no |
| `active` | Current published version | yes | no |
| `archived` | Superseded historical version | no | exact authorized version only |
| `withdrawn` | Explicitly deactivated version | no | no |

Visibility is either `public` or `internal`. It is always evaluated together
with trusted caller scope; the label alone is not a complete authorization
system.

### Chunks and embeddings

`document_chunks` preserves the Phase 4 stable chunk ID, zero-based position,
text, token count, heading path, page bounds, source block IDs, tokenizer
identity, full chunking configuration, and its canonical fingerprint. Equal text
is allowed across distinct valid chunks.

`chunk_embeddings` stores `public.vector(384)` plus the model ID, immutable
revision, embedding-configuration fingerprint, input fingerprint, instructions,
normalization flag, and encoding version. Its uniqueness identity is:

```text
(chunk_id, model_id, model_revision, embedding_config_fingerprint)
```

The vector type enforces 384 dimensions. Additional checks reject zero vectors,
malformed fingerprints, invalid relationships, and conflicting identities. All
foreign keys use restrictive updates and deletes; no parent deletion silently
removes provenance or historical content.

No HNSW or IVFFlat index is created. Exact search is intentional for the current
corpus size and establishes the correctness baseline before approximate search
is considered.

## Artifact validation and transactional import

The importer reuses the Phase 1 manifest validator, Phase 4 chunk loader, and
Phase 5 artifact loader. Before opening a database write transaction it verifies:

- source and artifact checksums;
- vector/metadata row alignment and unique stable chunk IDs;
- complete tenant, document, version, source, access, and provenance agreement;
- chunking configuration and canonical fingerprint;
- model ID, immutable revision, dimensions, normalization, and encoding version;
- every input fingerprint against the exact chunk text and encoding contract;
- finite, nonzero, correctly normalized vectors.

Validate artifacts without connecting to PostgreSQL:

```bash
.venv/bin/python -m scripts.import_rag_embeddings --check-only
```

Import into a directly reachable migrated database:

```bash
export DATABASE_URL='postgresql+asyncpg://USER:PASSWORD@HOST:5432/realtime_ai'
.venv/bin/python -m scripts.import_rag_embeddings
```

For Compose, the database hostname is available inside the Compose network and
generated artifacts must be mounted into the container:

```bash
docker compose run --rm migrate
docker compose run --rm \
  --volume "$PWD/data:/app/data:ro" \
  migrate python -m scripts.import_rag_embeddings
```

One import uses one transaction for the complete source scope:

```text
lock document identities
  -> insert or verify documents
  -> insert or verify versions
  -> insert or verify chunks
  -> insert or verify embeddings
  -> reconcile expected identities and dimensions
  -> commit
```

The importer distinguishes missing, identical, and conflicting identities.
Missing rows are inserted, identical rows are reused, and conflicting content
raises `RagImportConflict` and rolls back the whole transaction. Sorted advisory
locks serialize concurrent imports that share documents without imposing a
database-wide lock.

On an empty schema, the selected artifact should insert 75 documents, 79
versions, 806 chunks, and 806 embeddings. An identical retry should insert zero
new embeddings and verify/reuse all 806. These are expected values; only actual
CLI or test output establishes what happened in a database.

## Document lifecycle

`stage_document_version` accepts one validated import plan for exactly one
tenant, document, and version. It calculates an authoritative chunk-set
fingerprint from sorted unique source chunk IDs and persists the candidate as
`staged` through the existing importer. The expected set is never inferred from
whatever rows happen to exist in PostgreSQL.

`publish_document_version` locks the parent document and, in one transaction:

1. verifies the expected current active version;
2. requires the target to be staged;
3. reconciles stored chunks with the staged count and fingerprint;
4. requires one compatible embedding for every expected chunk;
5. archives the previous active version, when present;
6. activates the target and verifies the final state.

Any failure rolls back both status changes. Concurrent publishers serialize on
the document row; only a worker with the still-current precondition may commit.
Safe retries return an already-published result rather than repeating changes.

`withdraw_document_version` supports only `active -> withdrawn`, preserves all
content and provenance, and uses the same lock and expected-active precondition.
There is no hard-delete lifecycle operation.

## Exact dense retrieval

`DenseRetriever` encodes only the incoming question and reuses stored document
vectors. Query vectors are validated as finite, nonzero, 384-dimensional values.
The service then executes exact cosine search inside PostgreSQL:

```text
cosine_distance = embedding <=> query_vector
similarity = 1 - cosine_distance
```

The vector is supplied through asyncpg parameter binding and explicitly cast to
`public.vector`. PostgreSQL applies tenant, visibility, lifecycle, historical
scope, and complete embedding-profile filters before ordering and limiting.
Unauthorized or incompatible high-similarity rows therefore cannot consume an
authorized Top-K slot or enter Python application memory.

Equal distances use ascending `chunk_id` as the deterministic secondary key.
An unknown tenant, unknown version, or scope with no eligible records returns an
empty result; there is no cross-tenant or archived fallback.

Results preserve:

- stable chunk ID and original chunk text;
- tenant, document, version, and title;
- heading path, page range, and source block IDs;
- source path, source format, and visibility;
- embedding model, revision, and configuration identity;
- cosine distance and similarity.

Similarity is a ranking signal, not calibrated confidence or proof that a
passage answers the question.

## Trusted retrieval scope

`TrustedSearchContext` requires a nonblank tenant, one or more approved
visibilities, and a `SearchContextOrigin`. Historical search additionally
requires an exact document ID/version and explicit historical permission.

The type records that authorization was established elsewhere; constructing it
does not authenticate anyone. A future HTTP boundary must derive this context
from authenticated server-side policy. It must never trust a client-provided
tenant ID as authorization.

Run a trusted development query after inspecting the imported configuration
fingerprint:

```sql
SELECT embedding_model_id, embedding_model_revision,
       embedding_config_fingerprint, normalize_embeddings,
       encoding_config_version, query_instruction, count(*)
FROM chunk_embeddings
GROUP BY 1, 2, 3, 4, 5, 6;
```

```bash
.venv/bin/python scripts/search_rag_dense.py \
  --question "Can I get a refund if I was charged twice?" \
  --tenant asterdesk \
  --visibility public \
  --top-k 5 \
  --config-fingerprint <64-character-imported-fingerprint>
```

Historical development search requires all three explicit controls:

```bash
  --historical-document-id AST-REF-001 \
  --historical-version 1.0 \
  --authorize-historical
```

These arguments are trusted operator filters, not end-user authentication.

## Evaluation and integrity verification

The Phase 6.8 CLI is read-only. It never migrates, imports, deletes, publishes,
withdraws, or recreates database objects. Mutation and rollback guarantees run
separately in disposable integration-test schemas.

Evaluation identifies the immutable artifact through its run directory and the
SHA-256 checksums of `vectors.npy` and `metadata.jsonl`. It hashes the evaluation
question bank and records the model/profile, schema, visibility policy,
tie-breaking rule, tolerances, PostgreSQL/pgvector versions, and runtime details.

Integrity reconciliation compares the exact artifact identity set rather than a
global database count. It checks vectors, dimensions, embedding configuration,
chunk text and provenance, document/version ownership, lifecycle, visibility,
and source metadata. Missing or conflicting records stop evaluation.

For ranking parity, each question is encoded once. That exact query vector is
used for both:

1. NumPy cosine ranking over the authorized artifact candidates; and
2. PostgreSQL exact cosine ranking over the corresponding persisted candidates.

Both paths use identical scope and ascending chunk-ID tie-breaking. The report
separates exact matches, explainable numerical ties, and unexplained differences.

Retrieval quality retains the Phase 5 section-anchor definition: a hit requires
the labeled document, version, and section within Top-K. Expected-no-answer rows
are recorded but excluded from the recall denominator because Recall@K cannot
measure abstention.

Two views are reported:

- matched scope proves PostgreSQL/NumPy implementation parity;
- production scope applies the selected current visibility policy.

The historical Phase 5 values—20.75% Recall@5 and 33.96% Recall@10 over 106
questions—remain reference figures until reproduced from the selected immutable
artifacts and labels. The evaluation report never presents them as a current
measured result.

Run the final evaluation yourself against a populated development database:

```bash
.venv/bin/python scripts/evaluate_rag_dense.py \
  --schema public \
  --split validation \
  --visibility public
```

Successful runs create a unique directory beneath
`data/rag/evaluation/results/` containing:

- `report.json` — execution identity, integrity, parity, quality, latency,
  environment, and limitations;
- `retrieval_results.jsonl` — per-question scope and ranking outcomes;
- `ranking_comparison.json` — compact parity results.

The report includes model-load time separately and summarizes query encoding,
SQL retrieval, and end-to-end duration with sample count, median, nearest-rank
p95, minimum, and maximum. Poor recall alone does not fail the command because
there is no approved quality threshold. Integrity failure or an unexplained
ranking mismatch returns a nonzero status.

## Disposable PostgreSQL tests

The lightweight evaluation helpers require no transformer or database:

```bash
.venv/bin/python -m unittest tests.test_rag_dense_evaluation
```

For the repository's Compose environment, create the disposable database and
enable pgvector once:

```bash
docker compose up -d postgres
docker compose exec postgres \
  createdb --username realtime_ai --owner realtime_ai realtime_ai_test
docker compose exec postgres \
  psql --username realtime_ai --dbname realtime_ai_test \
  --command 'CREATE EXTENSION IF NOT EXISTS vector;'
```

If `createdb` reports that the database already exists, confirm it is the
disposable test database you previously created. Do not delete an unknown
database.

The production image intentionally omits tests and test artifacts. Mount the
current repository read-only and run the integration suite inside the Compose
network:

```bash
docker compose build api

docker compose run --rm --no-deps \
  --volume "$PWD:/workspace:ro" --workdir /workspace \
  -e TEST_DATABASE_URL='postgresql+asyncpg://realtime_ai:local_development_only@postgres:5432/realtime_ai_test' \
  api python -m unittest \
  tests.test_rag_embedding_import \
  tests.test_rag_document_lifecycle \
  tests.test_rag_dense_retrieval
```

The database integration tests create unique random schemas and remove only
those schemas. They cover:

- schema constraints and migration repeatability;
- initial import, identical retry, conflicts, concurrency, and rollback;
- lifecycle completeness, publication, withdrawal, and concurrent publication;
- cosine distance, Top-K, empty results, and stable tie-breaking;
- tenant, visibility, lifecycle, and embedding-profile isolation;
- unauthorized high-similarity candidates filtered before Top-K;
- exact artifact reconciliation and provenance-drift detection.

A missing `TEST_DATABASE_URL` causes skips. Skipped integration tests are not
successful PostgreSQL verification.

## Operational verification

After migration and import, inspect the actual database rather than relying only
on expected corpus counts:

```sql
SELECT version, name, checksum_sha256, applied_at
FROM public.schema_migrations
ORDER BY version;

SELECT extname, extversion
FROM pg_extension
WHERE extname = 'vector';

SELECT count(*) AS documents FROM documents;
SELECT status, count(*) FROM document_versions GROUP BY status ORDER BY status;
SELECT count(*) AS chunks FROM document_chunks;
SELECT count(*) AS embeddings FROM chunk_embeddings;

SELECT embedding_model_id, embedding_model_revision,
       embedding_config_fingerprint, public.vector_dims(embedding), count(*)
FROM chunk_embeddings
GROUP BY 1, 2, 3, 4;
```

Global counts are supporting context only. Exact artifact reconciliation is the
authoritative check when the database contains more than one legitimate scope.

## Failure diagnosis

- `invalid literal for int() with base 10: 'PORT'`: a documentation placeholder
  was used literally; supply a real numeric port and connection values.
- `No module named tests` in Compose: mount the repository read-only and set the
  container working directory to `/workspace` as shown above.
- `extension "vector" is not available`: the PostgreSQL server lacks compatible
  pgvector files; use the pinned image or install the server package.
- `type "vector" does not exist`: enable pgvector in the exact connected database.
- migration checksum mismatch: restore the original applied migration and create
  a new forward revision.
- `RagImportConflict`: an existing stable identity differs from the validated
  artifact; inspect the conflict instead of overwriting it.
- no retrieval results: verify tenant, visibility, lifecycle, embedding profile,
  and selected schema before assuming a ranking failure.
- ranking mismatch: confirm identical query vectors, candidate identities,
  lifecycle/visibility scope, and tie-breaking before diagnosing pgvector.

## Limitations and Phase 7 boundary

The corpus and question bank are synthetic and still require human review.
Recall@K measures retrieval of coarse section anchors, not exact-span quality,
answer correctness, groundedness, or abstention. Performance figures are valid
only for the recorded run, database, hardware, scope, and sample count.

Application-level filters do not replace database row-level security or
production authentication. No public retrieval route exists. A future API must
resolve identity and permissions server-side before constructing
`TrustedSearchContext`.

Dense-only retrieval can miss exact identifiers, rare terms, and lexical
matches. Phase 7 may add PostgreSQL keyword retrieval and hybrid fusion, but it
must preserve the same tenant, visibility, lifecycle, and compatibility filters
for every candidate source. Approximate vector indexes should be justified by
measured scale and latency rather than added prematurely.
