# Phase 6.3: Database migration verification and deployment

## Architecture and history

The repository uses a small, ordered SQL migration runner rather than Alembic.
It wraps the existing Month 1 schema and published Phase 6 migrations; it does not
redesign them or import embeddings.

| Revision | Source | Purpose | Dependency |
|---|---|---|---|
| `000` | `schema.sql` | Create the Month 1 `analyses` table | none |
| `001` | `migrations/001_enable_pgvector.sql` | Enable database-scoped pgvector | `000` |
| `002` | `migrations/002_create_rag_schema.sql` | Create the approved RAG tables | `001` |
| `003` | `migrations/003_add_document_lifecycle.sql` | Add lifecycle states and staged completeness metadata | `002` |

`scripts/migrate_database.py` records successful revisions in
`public.schema_migrations` by default. Each record contains the revision, name,
SHA-256 checksum, and application time. The runner rejects unknown revisions,
history gaps, and edited files. Revision 002 retains its original
`rag_schema_migrations` marker for compatibility; never remove it or rewrite an
applied SQL migration.

## Deployment sequence

```text
Start compatible PostgreSQL 17 + pgvector server files
        -> verify connectivity
        -> apply pending migrations
        -> verify migration revision
        -> verify pgvector and RAG schema
        -> start the application
```

Back up every non-disposable database first. Adapt this example to the deployment's
secret-management and connection conventions:

```bash
pg_dump -Fc -d realtime_ai -f realtime_ai_before_migration.dump
```

Do not delete PostgreSQL volumes to apply migrations. PostgreSQL data directories
are major-version-specific; keep existing PostgreSQL 17 volumes on PostgreSQL 17.

## Migration commands

The runner accepts the application's `postgresql+asyncpg://` URL and does not need
an OpenAI key:

```bash
export DATABASE_URL=postgresql+asyncpg://localhost:5432/realtime_ai
python -m scripts.migrate_database history
python -m scripts.migrate_database status
python -m scripts.migrate_database upgrade
python -m scripts.migrate_database status
```

`history` prints repository order. `status` reports `pending`, `applied`, or
`checksum-mismatch`. `upgrade` executes pending revisions only; a second successful
run prints `Applied migrations: none`.

On a fresh database, revision 000 creates `analyses`, 001 enables pgvector, 002
creates the empty RAG schema, and 003 adds the four-state lifecycle contract. On
an existing Month 1 database, the runner
validates the complete `analyses` column, primary-key, and latency-check contract
before recording 000. It does not replace the table or modify existing rows.

The extension files must already exist on the server. Revision 001 checks
`pg_available_extensions`, requires PostgreSQL 17, and runs:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

The deployment role needs `CREATE` on the database and permission to install the
trusted `vector` extension. If policy reserves extension installation for a DBA,
the DBA may enable it first and the deployment role can safely run the same upgrade.

## Docker Compose

### Automated migrations before application startup

Task 6.3 automates database migrations as part of the local Compose deployment.
Previously, an engineer had to apply SQL files manually, and PostgreSQL init
scripts only handled a newly created data volume. Compose now runs the repository's
ordered migration runner through a dedicated, short-lived `migrate` service.

When you run `docker compose up --build -d`, Compose builds the application and
migration images and follows this startup sequence:

```text
postgres starts -> PostgreSQL health check passes
                -> migrate applies pending revisions and exits successfully
                -> api starts on http://localhost:8000
```

The migration container executes `python -m scripts.migrate_database upgrade`.
The runner checks `schema_migrations` and skips recorded revisions. Existing
application records and the named database volume are retained. If a migration
fails, the service exits with an error and Compose blocks API startup; inspect
`docker compose logs migrate`, correct the cause, and rerun the deployment command.

This automation applies registered SQL migrations. Editing a SQLAlchemy model
alone does not change the database: schema changes still need a new migration file
and an entry in the runner's ordered `MIGRATIONS` manifest. Keep applied migration
files unchanged. A standalone `docker compose restart api` does not run migrations.

Set `OPENAI_API_KEY` in the local `.env` file before starting the complete stack.
Compose validates that setting even though the migration runner itself does not
call OpenAI. The application is deployed locally in Docker; these commands do not
publish it to a cloud server.

Compose pins `pgvector/pgvector:0.8.7-pg17-bookworm` by multi-platform digest.
The `migrate` service waits for PostgreSQL health and must finish successfully
before `api` starts. One-time PostgreSQL init scripts are deliberately not the
migration mechanism because they do not run for existing data volumes.

```bash
docker compose up --build -d
docker compose ps -a
docker compose logs migrate
```

On the first runner-managed installation, successful output is normally
`Applied migrations: 000, 001, 002`. If all revisions are already recorded, output
is `Applied migrations: none`. The migration service exiting with code 0 is
expected: it completes its job rather than running continuously.

An existing named volume is preserved. When invoked again, the runner reads its
ledger and applies only pending revisions. Check persistence without deleting volumes:

```bash
docker compose restart postgres
docker compose run --rm migrate python -m scripts.migrate_database status
```

The Dockerfile's CPU-only PyTorch installation is unchanged by migration packaging.

## Database verification

Run these queries against the intended database after `upgrade`:

```sql
SELECT version, name, checksum_sha256, applied_at
FROM public.schema_migrations
ORDER BY version;

SELECT current_setting('server_version') AS postgres_version;

SELECT extname, extversion
FROM pg_extension
WHERE extname = 'vector';

SELECT to_regclass('public.analyses'),
       to_regclass('public.documents'),
       to_regclass('public.document_versions'),
       to_regclass('public.document_chunks'),
       to_regclass('public.chunk_embeddings');
```

`tests/test_rag_schema_integration.py` verifies the published column, constraint,
trigger, and index contract through PostgreSQL catalogs, including `vector(384)`
and the absence of HNSW/IVFFlat indexes.

## Integration and regression tests

Use only a disposable database whose name contains `test`:

```bash
export RAG_TEST_PASSWORD=choose-a-temporary-url-safe-password
docker run -d --name realtime-ai-migration-test \
  -p 127.0.0.1:55432:5432 \
  -e POSTGRES_DB=realtime_ai_test \
  -e POSTGRES_USER=realtime_ai \
  -e POSTGRES_PASSWORD="$RAG_TEST_PASSWORD" \
  pgvector/pgvector:0.8.7-pg17-bookworm@sha256:ac08538c6f8b9904c33c8224c5e5706dbe760aca29db1d096972b4052c22a75d
docker exec realtime-ai-migration-test pg_isready -U realtime_ai -d realtime_ai_test
export TEST_DATABASE_URL="postgresql+asyncpg://realtime_ai:${RAG_TEST_PASSWORD}@127.0.0.1:55432/realtime_ai_test"
.venv/bin/python -m unittest tests.test_database_migrations -v
.venv/bin/python -m unittest tests.test_postgres_integration tests.test_rag_schema_integration -v
.venv/bin/python -m unittest discover -s tests -v
```

The migration tests create and remove random schemas. They cover fresh setup,
legacy-data preservation, repeatability, status, checksum immutability,
transactional failure, and retry. Missing `TEST_DATABASE_URL` is a skip, not a
successful integration result.

After reviewing results, remove only the disposable container:

```bash
docker stop realtime-ai-migration-test
docker rm -v realtime-ai-migration-test
unset TEST_DATABASE_URL RAG_TEST_PASSWORD
```

Never use `docker compose down -v` for migration testing or deployment.

## Failure diagnosis and recovery

Each revision runs in its own PostgreSQL transaction and is recorded only after
its SQL succeeds. Earlier committed revisions remain recorded when a later one
fails. Correct the cause and rerun `upgrade`; never insert ledger rows manually.

- `pgvector is not installed`: use the pinned image or install a PostgreSQL
  17-compatible server package, then retry.
- permission error during `CREATE EXTENSION`: use an authorized role to enable
  `vector`, or grant the deployment role the required database privilege.
- existing `analyses` validation failure: compare it with `schema.sql`; do not
  stamp an incompatible database.
- unexpected RAG object: inspect the partial/manual schema. Revision 002 refuses
  to hide this conflict.
- checksum mismatch: restore the published file and use a new forward revision
  for corrections instead of editing applied history.
- wrong URL or database: correct `DATABASE_URL`; connection and SQL failures do
  not record a revision.

PostgreSQL supports transactional DDL for the operations used here, so failed
table/function changes roll back together. `CREATE DATABASE` and some concurrent
index operations are exceptions, but these migrations use neither.

Known limitations: there is no destructive downgrade; one schema (normally
`public`) is targeted per invocation; installing pgvector server packages remains
an infrastructure responsibility; and runtime behavior remains unverified until
the real-PostgreSQL commands above complete.
