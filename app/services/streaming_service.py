import asyncio
import json
import logging
from collections.abc import AsyncIterator
from time import perf_counter

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import DatabaseError
from app.repositories.analysis_repository import save_analysis
from app.services.ai_service import AIServiceError, analyze_message

logger = logging.getLogger(__name__)


def format_sse(event: str, payload: dict[str, object]) -> str:
    """Format one named SSE event with a compact JSON data payload."""
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _error_event(code: str, message: str) -> str:
    return format_sse("error", {"code": code, "message": message})


async def stream_analysis(
    request: Request,
    session: AsyncSession,
    message: str,
) -> AsyncIterator[str]:
    """Run the existing analysis flow while reporting truthful stage boundaries.

    Client disconnects are mainly handled by Starlette: under uvicorn,
    StreamingResponse cancels this generator at whatever it is awaiting (for
    example the model call), and CancelledError propagates so cleanup runs.
    """
    try:
        yield format_sse("processing_started", {"status": "processing"})

        started_at = perf_counter()
        try:
            analysis = await analyze_message(message)
        except AIServiceError as error:
            logger.warning(
                "Streaming AI analysis failed error_type=%s", type(error).__name__
            )
            yield _error_event(
                "AI_UNAVAILABLE", "AI analysis is temporarily unavailable."
            )
            return

        yield format_sse("analysis_complete", analysis.model_dump())

        # Servers that do not cancel on disconnect only notice it when a send fails.
        # Checking here means a result is never stored for a client that has left.
        if await request.is_disconnected():
            logger.info("Analysis stream client disconnected before persistence")
            return

        latency_ms = max(0, round((perf_counter() - started_at) * 1000))
        try:
            record = await save_analysis(session, message, analysis, latency_ms)
        except DatabaseError as error:
            logger.error(
                "Streaming analysis persistence failed error_type=%s",
                type(error).__name__,
            )
            yield _error_event(
                "STORAGE_UNAVAILABLE", "Analysis storage is temporarily unavailable."
            )
            return

        yield format_sse("saved", {"analysis_id": record.id})
        yield format_sse(
            "completed",
            {"status": "completed", "analysis_id": record.id},
        )
    except asyncio.CancelledError:
        # Raised on client disconnect or server shutdown; re-raise so cleanup runs.
        logger.info("Analysis stream cancelled")
        raise
    except Exception as error:
        # The 200 headers are already sent, so an unexpected failure can only be
        # reported in-band. CancelledError is a BaseException and is not caught here.
        logger.error(
            "Streaming analysis failed unexpectedly error_type=%s",
            type(error).__name__,
        )
        yield _error_event("INTERNAL_ERROR", "Analysis failed.")
