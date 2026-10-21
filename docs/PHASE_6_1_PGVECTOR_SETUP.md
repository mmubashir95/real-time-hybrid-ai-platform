# Phase 6.1: PostgreSQL and pgvector setup

## Purpose and selected image

pgvector adds a PostgreSQL vector type and distance operators. Phase 6 needs it so
later tasks can store the 384-dimensional BGE embeddings and execute similarity
queries in PostgreSQL. This task does not create RAG tables, import embeddings, or
add retrieval indexes or APIs.

Compose uses the official pgvector project image
`pgvector/pgvector:0.8.7-pg17-bookworm`, pinned to multi-platform digest
`sha256:ac08538c6f8b9904c33c8224c5e5706dbe760aca29db1d096972b4052c22a75d`.
It is built from the same PostgreSQL 17 Bookworm image family previously used by
the project, includes pgvector 0.8.7 server files, and retains PostgreSQL's
`/var/lib/postgresql/data` data directory. PostgreSQL major version 17, service
name `postgres`, database name, credentials, health check, and `postgres_data`
volume mapping are unchanged.

Installing pgvector on a server and enabling it are separate operations. The
image supplies the server-side extension control and SQL files. The migration
then runs `CREATE EXTENSION IF NOT EXISTS vector` in one connected database.
Extensions are database-scoped, so enabling it in `realtime_ai` does not enable
it in another database such as `realtime_ai_test`.

## Protect an existing Compose database

Never use `docker compose down -v` during this upgrade. Before changing a running
container, confirm the cluster major version and create a logical backup:

```bash
docker compose exec postgres postgres --version
docker compose exec -T postgres pg_dump \
  -U realtime_ai -d realtime_ai -Fc > realtime_ai_before_pgvector.dump
docker compose exec -T postgres psql -U realtime_ai -d realtime_ai \
  -c "SELECT count(*) AS analyses_count FROM analyses;"
docker volume inspect real-time-hybrid-ai-platform_postgres_data
```

The exact Compose volume prefix can differ; find it with `docker volume ls`.
Check that the backup is non-empty and store it outside the Docker volume. A
logical dump is the portable rollback artifact; a raw data-directory copy is not
a substitute for a tested backup.

Validate the image separately before using the existing named volume:

```bash
docker volume create realtime_ai_pgvector_validation
docker run --rm -d --name realtime-ai-pgvector-validation \
  -e POSTGRES_PASSWORD=validation-only \
  -e POSTGRES_DB=validation_db \
  -v realtime_ai_pgvector_validation:/var/lib/postgresql/data \
  pgvector/pgvector:0.8.7-pg17-bookworm@sha256:ac08538c6f8b9904c33c8224c5e5706dbe760aca29db1d096972b4052c22a75d
docker exec realtime-ai-pgvector-validation pg_isready -U postgres -d validation_db
docker exec realtime-ai-pgvector-validation \
  psql -U postgres -d validation_db -c "CREATE EXTENSION vector;"
docker exec realtime-ai-pgvector-validation \
  psql -U postgres -d validation_db -c \
  "SELECT current_setting('server_version'), extversion FROM pg_extension WHERE extname = 'vector';"
docker stop realtime-ai-pgvector-validation
```

Remove only that explicitly named disposable validation volume after confirming
its identity. Do not remove the project volume.

Then recreate only the database container and run the ordered migration service
against the already initialized application database:

```bash
docker compose pull postgres
docker compose up -d --no-deps postgres
docker compose run --rm migrate python -m scripts.migrate_database upgrade
```

The migration rejects non-PostgreSQL-17 servers and reports an actionable error
when the server does not have pgvector installed. Re-running it is safe. It does
not modify `analyses` or create a production RAG schema.

Rollback means restoring the previous Compose image reference and recreating the
container against the unchanged PostgreSQL 17 volume. Keep the extension enabled:
the old plain PostgreSQL image lacks pgvector's shared library, so objects using
the extension would not work. If a database restore is required, create a clean
PostgreSQL 17 instance and restore `realtime_ai_before_pgvector.dump`; do not
overwrite or reinitialize the existing volume in place.

## Direct local setup and migration

A non-Docker PostgreSQL 17 server must have a compatible pgvector package installed
first. Then enable it in each intended database:

```bash
psql -v ON_ERROR_STOP=1 -d realtime_ai -f migrations/001_enable_pgvector.sql
psql -d realtime_ai -c \
  "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
```

The connected role must be allowed to create the trusted `vector` extension. Do
not put passwords in commands or committed files; use normal PostgreSQL credential
mechanisms.

## Functional smoke test

Run this in `realtime_ai` or another pgvector-enabled disposable database:

```sql
CREATE TEMP TABLE vector_smoke_test (
    id INTEGER PRIMARY KEY,
    embedding vector(3) NOT NULL
);
INSERT INTO vector_smoke_test (id, embedding) VALUES
    (1, '[1,0,0]'), (2, '[0.9,0.1,0]'), (3, '[0,1,0]');
SELECT id, 1 - (embedding <=> '[1,0,0]'::vector) AS similarity
FROM vector_smoke_test
ORDER BY embedding <=> '[1,0,0]'::vector
LIMIT 3;
CREATE TEMP TABLE vector_384_smoke_test (embedding vector(384));
```

The cosine ranking must be `1, 2, 3`. The `<=>` operator returns cosine distance,
so `1 - distance` is cosine similarity. BGE-small produces 384 values per vector,
which is why later storage will use `vector(384)`; PostgreSQL rejects vectors with
a different number of values.

## Integration tests

Use a disposable database, not production data. Install and enable pgvector there,
then run:

```bash
createdb realtime_ai_test
psql -v ON_ERROR_STOP=1 -d realtime_ai_test \
  -f migrations/001_enable_pgvector.sql
TEST_DATABASE_URL=postgresql+asyncpg://localhost:5432/realtime_ai_test \
  python -m unittest tests.test_postgres_integration -v
```

Tests create temporary tables and unique schemas. The extension itself is enabled
at database scope and intentionally is not dropped during teardown. Without
`TEST_DATABASE_URL`, the integration class is skipped; a skip is not a successful
pgvector validation.

## Troubleshooting

- `pgvector is not installed`: install server files compatible with PostgreSQL 17
  or use the pinned Compose image, then rerun the migration.
- `extension "vector" is not available`: the client reached PostgreSQL, but that
  server image/package does not contain pgvector.
- `permission denied to create extension`: connect as the database owner or a role
  permitted to install this trusted extension.
- `type "vector" does not exist`: enable the extension in the exact database named
  by the connection URL; enabling it in another database has no effect.
- `expected 384 dimensions`: the inserted embedding length does not match the
  declared model dimension.
- Existing volumes do not run init scripts again. Use the ordered migration runner
  rather than restarting PostgreSQL repeatedly.

## Remaining Phase 6 work

Task 6.2 added the production RAG schema, and Task 6.3 added ordered migration
tracking. Task 6.4 will import the validated embeddings. Retrieval and indexes
remain later work and are not part of this infrastructure change.
