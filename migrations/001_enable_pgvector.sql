-- Enable pgvector in one PostgreSQL 17 application database.
-- This migration is intentionally database-scoped and safe to run repeatedly.

DO $$
BEGIN
    IF current_setting('server_version_num')::INTEGER NOT BETWEEN 170000 AND 179999 THEN
        RAISE EXCEPTION
            'pgvector migration requires PostgreSQL 17; connected server is %',
            current_setting('server_version');
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_available_extensions
        WHERE name = 'vector'
    ) THEN
        RAISE EXCEPTION
            'pgvector is not installed on this PostgreSQL server; install a PostgreSQL 17-compatible pgvector package or use the configured pgvector image';
    END IF;
END
$$;

CREATE EXTENSION IF NOT EXISTS vector;
