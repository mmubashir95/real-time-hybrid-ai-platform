"""Apply and inspect the repository's ordered PostgreSQL migration history.

The runner deliberately stays small: SQL remains in the existing migration files,
while this module supplies durable revision tracking, checksum protection, ordered
execution, and safe adoption of the pre-migration Month 1 schema.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import asyncpg
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCHEMA = "public"
MIGRATION_TABLE = "schema_migrations"
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class MigrationError(RuntimeError):
    """Report invalid migration state or an unsafe baseline adoption."""


@dataclass(frozen=True)
class Migration:
    """Describe one immutable migration and its direct predecessor."""

    version: str
    name: str
    path: Path
    depends_on: str | None
    database_scoped: bool = False

    @property
    def checksum(self) -> str:
        """Return the SHA-256 digest used to detect edited migration history."""
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    @property
    def sql(self) -> str:
        """Load migration SQL only when the revision is about to be applied."""
        return self.path.read_text(encoding="utf-8")


MIGRATIONS: tuple[Migration, ...] = (
    Migration("000", "initial_schema", ROOT / "schema.sql", None),
    Migration(
        "001",
        "enable_pgvector",
        ROOT / "migrations" / "001_enable_pgvector.sql",
        "000",
        True,
    ),
    Migration(
        "002",
        "create_rag_schema",
        ROOT / "migrations" / "002_create_rag_schema.sql",
        "001",
    ),
)


def normalize_database_url(database_url: str) -> str:
    """Convert the application's SQLAlchemy async URL into an asyncpg DSN."""
    try:
        url = make_url(database_url)
    except ArgumentError as error:
        raise MigrationError("Database URL is not valid.") from error
    if url.drivername not in {"postgresql", "postgresql+asyncpg"}:
        raise MigrationError("Database URL must use PostgreSQL with asyncpg.")
    return url.set(drivername="postgresql").render_as_string(hide_password=False)


def validate_migration_graph(migrations: Sequence[Migration]) -> None:
    """Reject duplicate or out-of-order revision declarations before connecting."""
    seen: set[str] = set()
    previous: str | None = None
    for migration in migrations:
        if migration.version in seen:
            raise MigrationError(f"Duplicate migration version: {migration.version}")
        if migration.depends_on != previous:
            raise MigrationError(
                f"Migration {migration.version} depends on {migration.depends_on!r}; "
                f"expected {previous!r}."
            )
        if not migration.path.is_file():
            raise MigrationError(f"Migration file is missing: {migration.path}")
        seen.add(migration.version)
        previous = migration.version


class MigrationRunner:
    """Apply migrations transactionally and expose their recorded state."""

    def __init__(
        self,
        database_url: str,
        *,
        schema: str = DEFAULT_SCHEMA,
        migrations: Sequence[Migration] = MIGRATIONS,
    ) -> None:
        """Configure a runner without opening a database connection."""
        if not IDENTIFIER_PATTERN.fullmatch(schema):
            raise MigrationError("Schema must be an unquoted PostgreSQL identifier.")
        validate_migration_graph(migrations)
        self.database_url = normalize_database_url(database_url)
        self.schema = schema
        self.migrations = tuple(migrations)

    async def upgrade(self) -> list[str]:
        """Apply pending revisions in order and return newly applied versions."""
        connection = await asyncpg.connect(self.database_url)
        try:
            await self._ensure_schema_and_ledger(connection)
            applied = await self._read_applied(connection)
            self._validate_applied_history(applied)
            newly_applied: list[str] = []
            for migration in self.migrations:
                if migration.version in applied:
                    continue
                await self._apply_migration(connection, migration)
                newly_applied.append(migration.version)
            return newly_applied
        finally:
            await connection.close()

    async def status(self) -> list[tuple[Migration, str]]:
        """Return each revision with an applied, pending, or drifted state."""
        connection = await asyncpg.connect(self.database_url)
        try:
            applied = await self._read_applied(connection, ledger_may_be_absent=True)
            self._validate_applied_history(applied, allow_checksum_drift=True)
            return [
                (
                    migration,
                    "pending"
                    if migration.version not in applied
                    else (
                        "applied"
                        if applied[migration.version] == migration.checksum
                        else "checksum-mismatch"
                    ),
                )
                for migration in self.migrations
            ]
        finally:
            await connection.close()

    async def _ensure_schema_and_ledger(self, connection: asyncpg.Connection) -> None:
        """Create only the metadata needed to begin a migration transaction."""
        quoted_schema = _quote_identifier(self.schema)
        await connection.execute(f"CREATE SCHEMA IF NOT EXISTS {quoted_schema}")
        await connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {quoted_schema}.{MIGRATION_TABLE} (
                version TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                checksum_sha256 TEXT NOT NULL CHECK (checksum_sha256 ~ '^[0-9a-f]{{64}}$'),
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )

    async def _read_applied(
        self,
        connection: asyncpg.Connection,
        *,
        ledger_may_be_absent: bool = False,
    ) -> dict[str, str]:
        """Read recorded checksums, optionally treating a missing ledger as empty."""
        qualified_table = f"{_quote_identifier(self.schema)}.{MIGRATION_TABLE}"
        if ledger_may_be_absent:
            exists = await connection.fetchval("SELECT to_regclass($1)", qualified_table)
            if exists is None:
                return {}
        rows = await connection.fetch(
            f"SELECT version, checksum_sha256 FROM {qualified_table} ORDER BY version"
        )
        return {row["version"]: row["checksum_sha256"] for row in rows}

    def _validate_applied_history(
        self,
        applied: dict[str, str],
        *,
        allow_checksum_drift: bool = False,
    ) -> None:
        """Require recorded revisions to be a checksum-valid history prefix."""
        known_versions = [migration.version for migration in self.migrations]
        unknown = sorted(set(applied) - set(known_versions))
        if unknown:
            raise MigrationError(f"Unknown applied migration versions: {', '.join(unknown)}")
        applied_versions = [version for version in known_versions if version in applied]
        if applied_versions != known_versions[: len(applied_versions)]:
            raise MigrationError("Applied migrations are not a contiguous history prefix.")
        if allow_checksum_drift:
            return
        for migration in self.migrations:
            recorded = applied.get(migration.version)
            if recorded is not None and recorded != migration.checksum:
                raise MigrationError(
                    f"Migration {migration.version} checksum differs from recorded history."
                )

    async def _apply_migration(
        self, connection: asyncpg.Connection, migration: Migration
    ) -> None:
        """Apply and record one revision atomically; never record failed SQL."""
        quoted_schema = _quote_identifier(self.schema)
        qualified_table = f"{quoted_schema}.{MIGRATION_TABLE}"
        async with connection.transaction():
            # Serialize deployers so two processes cannot apply the same revision.
            await connection.execute(
                "SELECT pg_advisory_xact_lock(hashtext(current_database()), hashtext($1))",
                f"{self.schema}:schema_migrations",
            )
            already_applied = await connection.fetchval(
                f"SELECT EXISTS (SELECT 1 FROM {qualified_table} WHERE version = $1)",
                migration.version,
            )
            if already_applied:
                return
            if migration.version == "000" and await self._analyses_exists(connection):
                await self._validate_existing_month1_schema(connection)
            else:
                search_path = (
                    "public"
                    if migration.database_scoped
                    else f"{quoted_schema}, public"
                )
                await connection.execute(f"SET LOCAL search_path TO {search_path}")
                await connection.execute(migration.sql)
            await connection.execute(
                f"INSERT INTO {qualified_table} (version, name, checksum_sha256) "
                "VALUES ($1, $2, $3)",
                migration.version,
                migration.name,
                migration.checksum,
            )

    async def _analyses_exists(self, connection: asyncpg.Connection) -> bool:
        """Check for the legacy Month 1 table only in the configured schema."""
        qualified = f"{_quote_identifier(self.schema)}.analyses"
        return await connection.fetchval("SELECT to_regclass($1) IS NOT NULL", qualified)

    async def _validate_existing_month1_schema(
        self, connection: asyncpg.Connection
    ) -> None:
        """Adopt an existing baseline only when its required contract is intact."""
        columns = await connection.fetch(
            """
            SELECT column_name, data_type, is_nullable, character_maximum_length,
                   is_identity, identity_generation, column_default
            FROM information_schema.columns
            WHERE table_schema = $1 AND table_name = 'analyses'
            """,
            self.schema,
        )
        actual = {
            row["column_name"]: (
                row["data_type"],
                row["is_nullable"],
                row["character_maximum_length"],
                row["is_identity"],
                row["identity_generation"],
                row["column_default"],
            )
            for row in columns
        }
        expected = {
            "id": ("integer", "NO", None, "YES", "BY DEFAULT", None),
            "message": ("text", "NO", None, "NO", None, None),
            "category": ("character varying", "NO", 100, "NO", None, None),
            "urgency": ("character varying", "NO", 50, "NO", None, None),
            "summary": ("text", "NO", None, "NO", None, None),
            "entities": ("jsonb", "NO", None, "NO", None, None),
            "sentiment": ("character varying", "NO", 50, "NO", None, None),
            "created_at": (
                "timestamp with time zone",
                "NO",
                None,
                "NO",
                None,
                "now()",
            ),
            "latency_ms": ("integer", "NO", None, "NO", None, None),
        }
        if actual != expected:
            raise MigrationError(
                "Existing analyses table does not match the Month 1 baseline; "
                "refusing to mark migration 000 as applied."
            )
        # PostgreSQL cannot infer an asyncpg parameter's type through format()'s
        # variadic arguments, so schema parameters need an explicit text cast.
        primary_key = await connection.fetchval(
            """
            SELECT count(*) = 1
            FROM pg_constraint
            WHERE conrelid = format('%I.analyses', $1::text)::regclass
              AND contype = 'p'
              AND conkey = ARRAY[(SELECT attnum FROM pg_attribute
                                  WHERE attrelid = format('%I.analyses', $1::text)::regclass
                                    AND attname = 'id')]
            """,
            self.schema,
        )
        latency_check = await connection.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conrelid = format('%I.analyses', $1::text)::regclass
                  AND conname = 'ck_analyses_latency_ms_nonnegative'
                  AND pg_get_constraintdef(oid) = 'CHECK ((latency_ms >= 0))'
            )
            """,
            self.schema,
        )
        if not primary_key or not latency_check:
            raise MigrationError(
                "Existing analyses table is missing its primary key or latency constraint."
            )


def _quote_identifier(identifier: str) -> str:
    """Quote a previously validated PostgreSQL identifier."""
    return f'"{identifier}"'


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for deployment and operator inspection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("upgrade", "status", "history"))
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", ""))
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    return parser


async def _run_command(arguments: argparse.Namespace) -> int:
    """Execute one CLI command and render a stable, secret-free result."""
    if arguments.command == "history":
        validate_migration_graph(MIGRATIONS)
        for migration in MIGRATIONS:
            print(f"{migration.version} {migration.name}")
        return 0
    if not arguments.database_url:
        raise MigrationError("DATABASE_URL or --database-url is required.")
    runner = MigrationRunner(arguments.database_url, schema=arguments.schema)
    if arguments.command == "upgrade":
        applied = await runner.upgrade()
        print("Applied migrations: " + (", ".join(applied) if applied else "none"))
        return 0
    statuses = await runner.status()
    for migration, state in statuses:
        print(f"{migration.version} {migration.name} [{state}]")
    return 1 if any(state == "checksum-mismatch" for _, state in statuses) else 0


def main() -> int:
    """Run the migration CLI and convert expected failures into concise output."""
    arguments = _build_parser().parse_args()
    try:
        return asyncio.run(_run_command(arguments))
    except (MigrationError, asyncpg.PostgresError, OSError, TimeoutError) as error:
        print(f"Migration failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
