# Phase 6.8 — Dense Retrieval Evaluation

## Purpose and boundaries

Task 6.8 verifies the existing Phase 5 artifacts and Phase 6 PostgreSQL retrieval
path. It does not introduce another importer or search engine. The evaluation CLI
is read-only: it does not migrate, import, delete, publish, withdraw, or otherwise
change database rows. Mutation guarantees are tested separately in disposable
schemas through the existing importer and lifecycle integration tests.

The database scope supplied by the CLI is a trusted development/test scope. It
does not authenticate a user and must not be described as production identity
verification.

## Evaluation contract

The selected immutable embedding run supplies the model ID, pinned revision,
encoding configuration, normalization policy, vector dimension, input
fingerprints, and artifact checksums. The question-bank byte hash identifies the
labels. PostgreSQL searches use cosine distance (`embedding <=> query`) and order
by ascending distance, then ascending `chunk_id`. NumPy uses the identical query
vector and candidate identities, orders by descending cosine similarity, then
ascending `chunk_id`, and compares scores with an absolute tolerance of `1e-6`.

The matched-scope view permits `public` and `internal` artifacts to establish
implementation parity with the Phase 5 corpus. The production-scope view uses
the visibility arguments passed to the CLI (default: `public`). Current searches
include only active versions. Explicit version-conflict labels use the existing,
exact archived document/version rule; staged and withdrawn versions are never
eligible.

Recall@5 and Recall@10 retain the Phase 5 coarse matching rule: document ID,
document version, and section anchor must match. Expected-no-answer questions are
reported but excluded from the recall denominator. Recall is passage retrieval
quality, not answer accuracy, and similarity is not calibrated confidence.

## Integrity reconciliation

`reconcile_import_plan` selects the exact chunk identities and embedding profile
from the approved artifact. It compares vector dimensions and finite values,
vectors within tolerance, embedding configuration, chunk text and provenance,
document/version ownership, lifecycle, visibility, and source metadata. Extra
legitimate database runs do not fail reconciliation because the check is not a
database-wide row count.

The report records expected, verified, missing, duplicate, dimension, vector,
metadata, and relationship mismatch identities. A failed reconciliation stops
ranking evaluation rather than producing metrics from incompatible data.

## Running the owner-controlled verification

Set `DATABASE_URL` to a populated development database. Run the commands below
yourself from the repository root; implementation did not execute the final CLI.

```bash
python scripts/evaluate_rag_dense.py \
  --schema public \
  --split validation \
  --visibility public
```

To evaluate a named immutable run, add:

```bash
--run-directory data/rag/processed/embeddings/run-b12282b9784ea9fb
```

Each run creates a unique directory under `data/rag/evaluation/results/` with:

- `report.json`: execution identity, integrity, parity, recall, latency,
  PostgreSQL/pgvector versions, environment, and limitations.
- `retrieval_results.jsonl`: per-question IDs, status, matched-scope comparison,
  and production-scope result IDs.
- `ranking_comparison.json`: compact parity outcome.

Poor recall alone does not fail the command because no approved quality threshold
exists. Missing/corrupt artifacts, database failure, integrity failure, or an
unexplained ranking mismatch produces a nonzero exit status.

## Tests

The first command is inexpensive and does not load the transformer. The remaining
commands require a disposable PostgreSQL database whose name contains `test`.
Values such as `USER`, `PASSWORD`, `HOST`, `PORT`, and `DB_test` are descriptive
placeholders, not literal connection values.

```bash
python -m unittest tests.test_rag_dense_evaluation
```

For this repository's Compose environment, create the isolated database and
enable pgvector in it once:

```bash
docker compose up -d postgres
docker compose exec postgres \
  createdb --username realtime_ai --owner realtime_ai realtime_ai_test
docker compose exec postgres \
  psql --username realtime_ai --dbname realtime_ai_test \
  --command 'CREATE EXTENSION IF NOT EXISTS vector;'
```

If `createdb` reports that `realtime_ai_test` already exists, do not delete it
blindly. Confirm that it is the disposable test database you previously created.

Build the current application image, then run the integration tests inside the
Compose network. Here `postgres` is the service hostname and `5432` is the
container port:

```bash
docker compose build api

docker compose run --rm --no-deps \
  --volume "$PWD:/workspace:ro" --workdir /workspace \
  -e TEST_DATABASE_URL='postgresql+asyncpg://realtime_ai:local_development_only@postgres:5432/realtime_ai_test' \
  api python -m unittest \
  tests.test_rag_embedding_import \
  tests.test_rag_document_lifecycle \
  tests.test_rag_dense_retrieval

docker compose run --rm --no-deps \
  --volume "$PWD:/workspace:ro" --workdir /workspace \
  -e TEST_DATABASE_URL='postgresql+asyncpg://realtime_ai:local_development_only@postgres:5432/realtime_ai_test' \
  api python -m unittest discover -s tests
```

The read-only volume is required because the production Dockerfile intentionally
does not copy the test suite and all test artifacts into the runtime image. It
exposes the current repository for this one container without changing or
enlarging the production image. The `/workspace` working directory also ensures
Python imports the mounted source and tests rather than an older image copy.

Integration tests create random schemas and remove only those schemas. Never point
`TEST_DATABASE_URL` at production. The importer tests verify identical retries,
conflict rejection, and real transaction rollback after earlier writes. Retrieval
tests verify cosine math, bounded Top-K, empty/wrong-tenant behavior, lifecycle,
visibility, model compatibility, and unauthorized high-similarity candidates.

If you use a separately installed host PostgreSQL instead of Compose, substitute
its real username, password, hostname, numeric port, and disposable database name.
The repository's Compose service does not currently publish PostgreSQL to a host
port, so `localhost:5432` is not the Compose connection from the Mac shell.

## Practical exercise 1 — five synthetic vectors

Objective: see pgvector cosine ordering and compare it with NumPy. Use a disposable
test database and `psql`; the table is temporary and disappears on disconnect.

```sql
CREATE TEMP TABLE five_vectors (id text PRIMARY KEY, embedding vector(384));
INSERT INTO five_vectors VALUES
  ('same', (ARRAY[1::real] || array_fill(0::real, ARRAY[383]))::vector),
  ('near', (ARRAY[1::real, 0.1::real] || array_fill(0::real, ARRAY[382]))::vector),
  ('orthogonal', (ARRAY[0::real, 1::real] || array_fill(0::real, ARRAY[382]))::vector),
  ('opposite', (ARRAY[-1::real] || array_fill(0::real, ARRAY[383]))::vector),
  ('mixed', (ARRAY[0.7::real, 0.7::real] || array_fill(0::real, ARRAY[382]))::vector);
SELECT id,
       embedding <=> ('[1,' || array_to_string(array_fill(0, ARRAY[383]), ',') || ']')::vector AS distance,
       1 - (embedding <=> ('[1,' || array_to_string(array_fill(0, ARRAY[383]), ',') || ']')::vector) AS similarity
FROM five_vectors
ORDER BY distance, id;
DROP TABLE five_vectors;
```

Expected concept: `same` ranks first, `near` follows closely, orthogonal has zero
similarity, and opposite has negative similarity. Recreate the five arrays in a
small NumPy session and sort cosine scores descending to confirm the same order.

## Practical exercise 2 — five real Phase 5 embeddings

Objective: preserve row alignment and provenance with real artifacts. First run
the artifact-only importer check, which opens no database connection:

```bash
python scripts/import_rag_embeddings.py --check-only
```

Then inspect the first five metadata rows and their corresponding matrix rows in
a Python shell, encode one existing question with the established
`SentenceTransformerEncoder`, and insert those five vectors into a `TEMP TABLE`
with columns `(chunk_id text, source_path text, embedding vector(384))`. Query with
`ORDER BY embedding <=> $query_vector, chunk_id LIMIT 5`. The returned chunk IDs
must remain paired with the source metadata from the same row indexes. Disconnect
or `DROP TABLE` to clean up. Do not copy rows into production RAG tables.

## Practical exercise 3 — filtering before Top-K

Objective: prove access filters run before the limit. In a temporary table create
five rows with `(tenant_id, status, visibility, embedding)`: a perfect foreign-
tenant match, a perfect archived match, a perfect internal match, and two weaker
active/public matches for the authorized tenant.

```sql
SELECT id, 1 - (embedding <=> :query_vector::vector) AS similarity
FROM disposable_filter_vectors
WHERE tenant_id = 'asterdesk'
  AND status = 'active'
  AND visibility = 'public'
ORDER BY embedding <=> :query_vector::vector, id
LIMIT 2;
```

Expected concept: both weaker authorized rows fill Top-2; none of the three perfect
but unauthorized rows consumes a slot. Remove one predicate at a time to observe
how the candidate set changes, restore it, and finish with `DROP TABLE
disposable_filter_vectors`. This demonstrates filtering semantics only; production
authorization must still establish the trusted tenant and visibility scope.

## Known limitations

- The checked-in question bank is synthetic and explicitly review-required.
- Historical reference values of 20.75% Recall@5 and 33.96% Recall@10 are not
  treated as verified until reproduced from the immutable artifacts and labels.
- Exact search is intentional at this corpus size; no HNSW or IVFFlat index is
  introduced by this task.
- Cold model load is reported separately. Median and nearest-rank p95 describe
  only the queries and environment in the generated report.
