"""Real PostgreSQL checks for schema.sql and the analysis repository.

Skipped unless TEST_DATABASE_URL points at a disposable database, for example:

    createdb realtime_ai_test
    TEST_DATABASE_URL=postgresql+asyncpg://localhost:5432/realtime_ai_test \
        python -m unittest tests.test_postgres_integration

Each test works inside throwaway schemas that are dropped afterwards.
"""

import os
import unittest
from pathlib import Path
from uuid import uuid4

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from app.database import DatabaseError
from app.models import Base
from app.repositories.analysis_repository import get_analysis_by_id, save_analysis
from app.schemas import AnalysisResult

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "")
SCHEMA_SQL = (Path(__file__).resolve().parent.parent / "schema.sql").read_text()


@unittest.skipUnless(TEST_DATABASE_URL, "TEST_DATABASE_URL is not set")
class PostgresIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.admin_engine = create_async_engine(TEST_DATABASE_URL)
        self.schemas: list[str] = []
        self.engines: list[AsyncEngine] = []
        self.analysis = AnalysisResult(
            category="billing",
            urgency="high",
            summary="Duplicate charge.",
            entities={"amount": "$10", "order_id": "A-1"},
            sentiment="negative",
        )

    async def asyncTearDown(self) -> None:
        for engine in self.engines:
            await engine.dispose()
        async with self.admin_engine.begin() as connection:
            for schema in self.schemas:
                await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await self.admin_engine.dispose()

    async def _engine_in_new_schema(self) -> AsyncEngine:
        schema = f"test_{uuid4().hex}"
        async with self.admin_engine.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        self.schemas.append(schema)
        engine = create_async_engine(
            TEST_DATABASE_URL,
            connect_args={"server_settings": {"search_path": schema}},
        )
        self.engines.append(engine)
        return engine

    async def _schema_sql_engine(self) -> AsyncEngine:
        engine = await self._engine_in_new_schema()
        async with engine.begin() as connection:
            await connection.exec_driver_sql(SCHEMA_SQL)
        return engine

    async def test_saved_analysis_round_trips_through_postgres(self) -> None:
        sessions = async_sessionmaker(
            await self._schema_sql_engine(), expire_on_commit=False
        )

        async with sessions() as session:
            saved = await save_analysis(
                session, "I was charged twice.", self.analysis, 125
            )

        # A fresh session proves the row was committed, not just held in memory.
        async with sessions() as session:
            stored = await get_analysis_by_id(session, saved.id)
            missing = await get_analysis_by_id(session, saved.id + 1000)

        self.assertIsNotNone(stored)
        self.assertEqual(stored.message, "I was charged twice.")
        self.assertEqual(stored.entities, self.analysis.entities)
        self.assertEqual(stored.latency_ms, 125)
        self.assertIsNotNone(stored.created_at.tzinfo)
        self.assertEqual(stored.created_at, saved.created_at)
        self.assertIsNone(missing)

    async def test_failed_write_rolls_back_and_session_stays_usable(self) -> None:
        sessions = async_sessionmaker(
            await self._schema_sql_engine(), expire_on_commit=False
        )

        async with sessions() as session:
            with self.assertLogs(
                "app.repositories.analysis_repository", "ERROR"
            ) as logs:
                with self.assertRaises(DatabaseError):
                    await save_analysis(session, "Hello", self.analysis, -1)
            saved = await save_analysis(session, "Hello", self.analysis, 1)

        self.assertIn("CheckViolationError", logs.output[0])
        self.assertIsNotNone(saved.id)

    async def test_missing_table_is_controlled_and_diagnosable(self) -> None:
        sessions = async_sessionmaker(await self._engine_in_new_schema())

        async with sessions() as session:
            with self.assertLogs(
                "app.repositories.analysis_repository", "ERROR"
            ) as logs:
                with self.assertRaises(DatabaseError):
                    await get_analysis_by_id(session, 1)

        self.assertIn("UndefinedTableError", logs.output[0])

    async def test_schema_sql_matches_sqlalchemy_model(self) -> None:
        from_sql = await self._schema_sql_engine()
        from_model = await self._engine_in_new_schema()
        async with from_model.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        self.assertEqual(
            await _describe_analyses(from_sql),
            await _describe_analyses(from_model),
        )


async def _describe_analyses(engine: AsyncEngine) -> dict:
    def describe(sync_connection) -> dict:
        inspector = inspect(sync_connection)
        schema = sync_connection.execute(text("SELECT current_schema()")).scalar()
        return {
            "columns": [
                (
                    column["name"],
                    str(column["type"]),
                    column["nullable"],
                    column.get("default"),
                    bool(column.get("identity")),
                )
                for column in inspector.get_columns("analyses", schema=schema)
            ],
            "primary_key": inspector.get_pk_constraint("analyses", schema=schema)[
                "constrained_columns"
            ],
            "checks": sorted(
                (check["name"], check["sqltext"])
                for check in inspector.get_check_constraints(
                    "analyses", schema=schema
                )
            ),
        }

    async with engine.connect() as connection:
        return await connection.run_sync(describe)


if __name__ == "__main__":
    unittest.main()
