import asyncio
import logging
from time import perf_counter

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import DatabaseError
from app.models import AnalysisRecord
from app.schemas import AnalysisResult

logger = logging.getLogger(__name__)

# asyncpg raises a raw OSError when PostgreSQL is unreachable; SQLAlchemy does not
# wrap connection-refused errors, so both must be caught to return a controlled 503.
DATABASE_ERRORS = (SQLAlchemyError, OSError)


def _error_type(error: BaseException) -> str:
    """Name the driver-level cause (e.g. UndefinedTableError) without leaking details.

    SQLAlchemy wraps the DBAPI error in `error.orig`, and its asyncpg adapter in turn
    keeps the original asyncpg exception as `orig.__cause__`.
    """
    original = getattr(error, "orig", None)
    return type(getattr(original, "__cause__", None) or original or error).__name__


async def save_analysis(
    session: AsyncSession,
    message: str,
    analysis: AnalysisResult,
    latency_ms: int,
    *,
    request_id: str = "-",
) -> AnalysisRecord:
    record = AnalysisRecord(
        message=message,
        category=analysis.category,
        urgency=analysis.urgency,
        summary=analysis.summary,
        entities=analysis.entities,
        sentiment=analysis.sentiment,
        latency_ms=latency_ms,
    )
    database_started_at = perf_counter()
    database_status = "error"
    try:
        session.add(record)
        await session.flush()
        await session.refresh(record)
        await session.commit()
        database_status = "success"
    except asyncio.CancelledError:
        database_status = "cancelled"
        raise
    except DATABASE_ERRORS as error:
        try:
            await session.rollback()
        except DATABASE_ERRORS as rollback_error:
            logger.error(
                "Failed to roll back analysis transaction error_type=%s",
                _error_type(rollback_error),
            )
        logger.error("Failed to save analysis error_type=%s", _error_type(error))
        raise DatabaseError("Failed to save analysis.") from error
    finally:
        logger.info(
            "performance request_id=%s operation=database_save "
            "database_latency_ms=%.3f status=%s",
            request_id,
            (perf_counter() - database_started_at) * 1000,
            database_status,
        )

    return record


async def get_analysis_by_id(
    session: AsyncSession, analysis_id: int
) -> AnalysisRecord | None:
    try:
        result = await session.execute(
            select(AnalysisRecord).where(AnalysisRecord.id == analysis_id)
        )
    except DATABASE_ERRORS as error:
        logger.error(
            "Failed to retrieve analysis error_type=%s", _error_type(error)
        )
        raise DatabaseError("Failed to retrieve analysis.") from error

    return result.scalar_one_or_none()
