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


async def stream_analysis(
    request: Request,
    session: AsyncSession,
    message: str,
) -> AsyncIterator[str]:
    """Run the existing analysis flow while reporting truthful stage boundaries."""
    try:
        if await request.is_disconnected():
            return

        yield format_sse("processing_started", {"status": "processing"})

        if await request.is_disconnected():
            return

        started_at = perf_counter()
        try:
            analysis = await analyze_message(message)
        except AIServiceError as error:
            logger.warning(
                "Streaming AI analysis failed error_type=%s", type(error).__name__
            )
            if not await request.is_disconnected():
                yield format_sse(
                    "error",
                    {
                        "code": "AI_UNAVAILABLE",
                        "message": "AI analysis is temporarily unavailable.",
                    },
                )
            return

        if await request.is_disconnected():
            return

        yield format_sse("analysis_complete", analysis.model_dump())

        if await request.is_disconnected():
            return

        latency_ms = max(0, round((perf_counter() - started_at) * 1000))
        try:
            record = await save_analysis(session, message, analysis, latency_ms)
        except DatabaseError as error:
            logger.error(
                "Streaming analysis persistence failed error_type=%s",
                type(error).__name__,
            )
            if not await request.is_disconnected():
                yield format_sse(
                    "error",
                    {
                        "code": "STORAGE_UNAVAILABLE",
                        "message": "Analysis storage is temporarily unavailable.",
                    },
                )
            return

        if await request.is_disconnected():
            return

        yield format_sse("saved", {"analysis_id": record.id})

        if await request.is_disconnected():
            return

        yield format_sse(
            "completed",
            {"status": "completed", "analysis_id": record.id},
        )
    except asyncio.CancelledError:
        # StreamingResponse cancellation should also cancel an awaited provider or
        # database operation when its client supports cooperative cancellation.
        logger.info("Analysis stream cancelled after client disconnect")
        raise
