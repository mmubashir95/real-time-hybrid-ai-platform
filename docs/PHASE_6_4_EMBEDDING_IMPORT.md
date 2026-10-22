# Phase 6.4: Transactional embedding import

## Purpose and boundary

Task 6.4 imports the already generated Phase 5 artifacts into the approved
PostgreSQL/pgvector schema. It does not run the embedding model, alter lifecycle
statuses, create retrieval APIs, or add vector indexes.

The importer resolves `data/rag/processed/embeddings/latest.json` by default. It
loads the selected run's `vectors.npy`, `metadata.jsonl`, and `report.json`, then
joins them to `data/rag/processed/chunks.jsonl`, `chunking_report.json`, and
`manifest.csv` through stable identities.

```text
manifest.csv                 -> documents and document_versions
chunks.jsonl                 -> document_chunks
chunking_report.json         -> chunking configuration identity
metadata.jsonl + vectors.npy -> chunk_embeddings
report.json                  -> embedding model/configuration identity
```

The current published run contains 75 logical documents, 79 versions, 806 chunks,
and 806 embeddings. These are observed artifact values, not hardcoded importer
limits.

## Validation before database writes

`load_import_plan` reuses the Phase 1 manifest validator, Phase 4 chunk loader,
and Phase 5 embedding loader. Before a database connection is required, it checks:

- source-file and artifact checksums;
- vector/metadata count and row-index alignment;
- unique stable chunk IDs;
- exact tenant, document, version, source, access, and provenance agreement;
- the full chunking configuration and its canonical fingerprint;
- model ID, immutable revision, 384 dimensions, normalization, and encoding version;
- each input fingerprint against the exact chunk text and encoding settings;
- finite, nonzero, correctly normalized vectors.

Matching dimensions alone are insufficient. The accepted artifact must use
`BAAI/bge-small-en-v1.5`, revision
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, and the approved Phase 5 encoding
contract.

## Transaction and retry behavior

One import uses one PostgreSQL transaction for the complete source scope:

```text
lock document identities
  -> insert/verify documents
  -> insert/verify versions
  -> insert/verify chunks
  -> insert/verify embeddings
  -> reconcile exact expected identities and dimensions
  -> commit
```

Document advisory locks are acquired in sorted identity order. Concurrent imports
that share a logical document serialize, while unrelated documents can proceed.
Database primary, foreign, and unique constraints remain the final integrity
boundary.

For every identity, the importer distinguishes:

- missing: insert it;
- present and identical: reuse it;
- present but different: raise `RagImportConflict` and roll back.

`ON CONFLICT DO NOTHING` is used only to resolve concurrent insertion races; the
existing row is always read and compared before it counts as reused. Version
status is compared exactly, so archived sources stay archived and the importer
never activates or archives a version.

## CLI usage

First validate all source contracts without connecting to PostgreSQL:

```bash
.venv/bin/python -m scripts.import_rag_embeddings --check-only
```

For a directly reachable PostgreSQL database:

```bash
export DATABASE_URL=postgresql+asyncpg://localhost:5432/realtime_ai
.venv/bin/python -m scripts.import_rag_embeddings
```

The Compose PostgreSQL service is not published to the host. Run the importer on
the Compose network and mount generated artifacts read-only:

```bash
docker compose build migrate
docker compose run --rm migrate
docker compose run --rm \
  -v "$PWD/data:/app/data:ro" \
  migrate python -m scripts.import_rag_embeddings
```

The first `docker compose run --rm migrate` applies pending schema revisions.
The second invocation overrides that service's normal command with the importer.

Use `--run-directory PATH` to select a reviewed immutable run explicitly. The
default follows `latest.json`. `--schema` defaults to `public`. Output redacts the
database password and reports inserted and reused counts separately.

On an initially empty RAG schema, the expected first import reflects the actual
artifact scope: 75 documents, 79 versions, 806 chunks, and 806 embeddings inserted.
An identical second import should report those identities as reused and insert
zero embeddings. Treat the CLI output as the execution record; do not infer a
successful import from these expected values.

## Verification queries

Run after a successful import:

```sql
SELECT count(*) AS documents FROM documents;
SELECT count(*) AS versions FROM document_versions;
SELECT status, count(*) FROM document_versions GROUP BY status ORDER BY status;
SELECT count(*) AS chunks FROM document_chunks;
SELECT count(*) AS embeddings FROM chunk_embeddings;

SELECT embedding_model_id, embedding_model_revision,
       embedding_config_fingerprint, public.vector_dims(embedding), count(*)
FROM chunk_embeddings
GROUP BY 1, 2, 3, 4;

SELECT count(*) AS orphan_embeddings
FROM chunk_embeddings e
LEFT JOIN document_chunks c ON c.chunk_id=e.chunk_id
WHERE c.chunk_id IS NULL;
```

Counts may be greater than the imported scope if other valid RAG records already
exist. The importer itself reconciles the exact source identity set rather than
trusting global counts.

## PostgreSQL integration and regression tests

Use the disposable pgvector database described in the Phase 6.3 migration guide:

```bash
export TEST_DATABASE_URL="postgresql+asyncpg://realtime_ai:${RAG_TEST_PASSWORD}@127.0.0.1:55432/realtime_ai_test"
.venv/bin/python -m unittest tests.test_rag_embedding_import -v
.venv/bin/python -m unittest \
  tests.test_rag_embedding_generation \
  tests.test_rag_schema_integration \
  tests.test_database_migrations -v
.venv/bin/python -m unittest discover -s tests -v
```

The integration tests use random schemas. They verify initial commit, identical
retry, archived status preservation, conflicting version/chunk/vector rejection,
transaction rollback, concurrent identical imports, relationship integrity, and
Month 1 row preservation. A missing `TEST_DATABASE_URL` produces skips, which are
not successful PostgreSQL verification.

## Failure handling and limitations

Artifact validation failures happen before database writes. Database conflicts or
PostgreSQL failures roll back the entire import transaction. Existing committed
application and RAG rows remain unchanged. The command never creates a database,
drops tables, truncates rows, or modifies source artifacts.

The importer itself does not publish versions. Task 6.5 now reuses
`load_import_plan` and `import_rag_embeddings` to stage a validated version while
leaving ordinary corpus-import behavior unchanged. See
`docs/PHASE_6_5_DOCUMENT_LIFECYCLE.md`. Vector similarity search and indexes remain
later Phase 6 work.
