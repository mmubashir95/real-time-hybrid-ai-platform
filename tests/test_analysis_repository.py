import unittest
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import DatabaseError
from app.models import AnalysisRecord
from app.repositories.analysis_repository import (
    get_analysis_by_id,
    save_analysis,
)
from app.schemas import AnalysisResult


class AnalysisRepositoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.session = Mock(spec=AsyncSession)
        self.session.flush = AsyncMock()
        self.session.commit = AsyncMock()
        self.session.refresh = AsyncMock()
        self.session.rollback = AsyncMock()
        self.session.execute = AsyncMock()
        self.analysis = AnalysisResult(
            category="billing",
            urgency="high",
            summary="Duplicate charge.",
            entities={"amount": "$10"},
            sentiment="negative",
        )

    async def test_save_analysis_persists_request_result_and_metadata(self) -> None:
        created_at = datetime(2026, 9, 26, 10, 0, tzinfo=UTC)

        async def populate_database_values(record: AnalysisRecord) -> None:
            record.id = 42
            record.created_at = created_at

        self.session.refresh.side_effect = populate_database_values

        with patch(
            "app.repositories.analysis_repository.perf_counter",
            side_effect=(20.0, 20.03),
        ), self.assertLogs(
            "app.repositories.analysis_repository", "INFO"
        ) as logs:
            record = await save_analysis(
                self.session,
                "I was charged twice.",
                self.analysis,
                125,
                request_id="request-123",
            )

        self.session.add.assert_called_once_with(record)
        self.session.flush.assert_awaited_once_with()
        self.session.refresh.assert_awaited_once_with(record)
        self.session.commit.assert_awaited_once_with()
        self.session.rollback.assert_not_awaited()
        self.assertEqual(record.id, 42)
        self.assertEqual(record.message, "I was charged twice.")
        self.assertEqual(record.category, self.analysis.category)
        self.assertEqual(record.urgency, self.analysis.urgency)
        self.assertEqual(record.summary, self.analysis.summary)
        self.assertEqual(record.entities, self.analysis.entities)
        self.assertEqual(record.sentiment, self.analysis.sentiment)
        self.assertEqual(record.created_at, created_at)
        self.assertEqual(record.latency_ms, 125)
        self.assertIn("request_id=request-123", logs.output[0])
        self.assertIn("database_latency_ms=30.000", logs.output[0])
        self.assertIn("status=success", logs.output[0])

    async def test_save_failure_rolls_back_and_is_controlled(self) -> None:
        self.session.commit.side_effect = SQLAlchemyError("connection secret")

        with self.assertRaisesRegex(DatabaseError, "Failed to save"):
            await save_analysis(self.session, "Hello", self.analysis, 1)

        self.session.rollback.assert_awaited_once_with()

    async def test_unreachable_database_on_save_is_controlled(self) -> None:
        # asyncpg raises a bare OSError (not a SQLAlchemyError) on connection refused.
        self.session.flush.side_effect = OSError("Connect call failed")

        with self.assertRaisesRegex(DatabaseError, "Failed to save"):
            await save_analysis(self.session, "Hello", self.analysis, 1)

        self.session.commit.assert_not_awaited()
        self.session.rollback.assert_awaited_once_with()

    async def test_get_analysis_returns_database_record(self) -> None:
        expected = AnalysisRecord(
            id=42,
            message="Hello",
            category="general",
            urgency="low",
            summary="Greeting.",
            entities={},
            sentiment="neutral",
            latency_ms=10,
        )
        result = Mock()
        result.scalar_one_or_none.return_value = expected
        self.session.execute.return_value = result

        record = await get_analysis_by_id(self.session, 42)

        self.assertIs(record, expected)
        self.session.execute.assert_awaited_once()

    async def test_get_analysis_failure_is_controlled(self) -> None:
        self.session.execute.side_effect = SQLAlchemyError("connection secret")

        with self.assertRaisesRegex(DatabaseError, "Failed to retrieve"):
            await get_analysis_by_id(self.session, 42)

    async def test_unreachable_database_on_get_is_controlled(self) -> None:
        self.session.execute.side_effect = OSError("Connect call failed")

        with self.assertRaisesRegex(DatabaseError, "Failed to retrieve"):
            await get_analysis_by_id(self.session, 42)
