-- Extend document-version lifecycle and persist the authoritative staged chunk set.
-- Legacy active/archived rows intentionally keep nullable completeness metadata.
DO $migration$
DECLARE
    target_schema TEXT := current_schema();
BEGIN
    IF target_schema IS NULL OR target_schema LIKE 'pg_%' THEN
        RAISE EXCEPTION
            'Select a writable application schema before applying the lifecycle migration';
    END IF;

    PERFORM set_config('search_path', format('%I, public', target_schema), true);
    PERFORM pg_advisory_xact_lock(
        hashtext(current_database()), hashtext(target_schema || ':rag_schema')
    );

    IF NOT EXISTS (
        SELECT 1 FROM rag_schema_migrations
        WHERE version = '002_create_rag_schema'
    ) THEN
        RAISE EXCEPTION 'Apply migrations/002_create_rag_schema.sql before migration 003';
    END IF;
    IF EXISTS (
        SELECT 1 FROM rag_schema_migrations
        WHERE version = '003_add_document_lifecycle'
    ) THEN
        RAISE NOTICE '003_add_document_lifecycle already applied; skipping';
        RETURN;
    END IF;

    ALTER TABLE document_versions
        DROP CONSTRAINT document_versions_status_check,
        ADD CONSTRAINT document_versions_status_check
            CHECK (status IN ('staged', 'active', 'archived', 'withdrawn')),
        ADD COLUMN expected_chunk_count INTEGER
            CHECK (expected_chunk_count > 0),
        ADD COLUMN expected_chunk_ids_fingerprint TEXT
            CHECK (expected_chunk_ids_fingerprint ~ '^[0-9a-f]{64}$'),
        ADD CONSTRAINT ck_document_versions_staged_completeness CHECK (
            status <> 'staged'
            OR (
                expected_chunk_count IS NOT NULL
                AND expected_chunk_ids_fingerprint IS NOT NULL
            )
        ),
        ADD CONSTRAINT ck_document_versions_completeness_pair CHECK (
            (expected_chunk_count IS NULL)
            = (expected_chunk_ids_fingerprint IS NULL)
        );

    COMMENT ON COLUMN document_versions.expected_chunk_count IS
        'Authoritative count from the validated source chunk set; required for staged versions.';
    COMMENT ON COLUMN document_versions.expected_chunk_ids_fingerprint IS
        'SHA-256 of compact JSON encoding of sorted unique source chunk IDs.';

    INSERT INTO rag_schema_migrations (version)
    VALUES ('003_add_document_lifecycle');
END
$migration$;
