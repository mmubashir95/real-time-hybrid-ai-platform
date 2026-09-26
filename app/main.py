import asyncio
import logging
from time import perf_counter
from typing import Annotated
from uuid import uuid4

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Path,
    Request,
    Response,
    status,
)
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import DatabaseError, get_session
from app.models import AnalysisRecord
from app.repositories.analysis_repository import (
    get_analysis_by_id,
    save_analysis,
)
from app.schemas import AnalysisRequest, AnalysisResult, StoredAnalysis
from app.services.ai_service import AIServiceError, analyze_message
from app.services.streaming_service import stream_analysis

# Uvicorn only configures its own loggers; without this, app INFO logs (such as
# the /async-demo start/finish lines) are dropped.
logging.basicConfig(level=logging.INFO)

app = FastAPI()
logger = logging.getLogger(__name__)

ASYNC_DEMO_DELAY_SECONDS = 3.0
POSTGRES_INTEGER_MAX = 2_147_483_647


@app.exception_handler(DatabaseError)
async def database_error_handler(
    _request: Request, error: DatabaseError
) -> JSONResponse:
    logger.error("Database operation unavailable error_type=%s", type(error).__name__)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Analysis storage is temporarily unavailable."},
    )


async def fake_async_model_call() -> str:
    """Simulate model I/O without blocking the event-loop thread."""
    await asyncio.sleep(ASYNC_DEMO_DELAY_SECONDS)
    return "async result"


@app.get("/async-demo")
async def async_demo() -> dict[str, str]:
    request_id = uuid4().hex
    started_at = perf_counter()
    logger.info("Async demo started request_id=%s", request_id)

    result = await fake_async_model_call()

    logger.info(
        "Async demo finished request_id=%s duration_seconds=%.3f",
        request_id,
        perf_counter() - started_at,
    )
    return {"result": result}


@app.post(
    "/analyze",
    response_model=AnalysisResult,
    responses={503: {"description": "AI analysis or storage is unavailable."}},
)
async def analyze(
    analysis_request: AnalysisRequest,
    response: Response,
    session: AsyncSession = Depends(get_session),
) -> AnalysisResult:
    request_id = uuid4().hex
    # Returned to the caller so a slow response can be matched to its
    # model/database performance log lines.
    response.headers["X-Request-ID"] = request_id
    request_started_at = perf_counter()
    request_status = "error"
    try:
        analysis_started_at = perf_counter()
        try:
            analysis = await analyze_message(
                analysis_request.message,
                request_id=request_id,
            )
        except AIServiceError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AI analysis is temporarily unavailable.",
                headers={"X-Request-ID": request_id},
            ) from error

        analysis_latency_ms = max(
            0, round((perf_counter() - analysis_started_at) * 1000)
        )
        record = await save_analysis(
            session,
            analysis_request.message,
            analysis,
            analysis_latency_ms,
            request_id=request_id,
        )
        response.headers["X-Analysis-ID"] = str(record.id)
        response.headers["Location"] = f"/analyses/{record.id}"
        request_status = "success"
        return analysis
    except asyncio.CancelledError:
        request_status = "cancelled"
        raise
    finally:
        logger.info(
            "performance request_id=%s operation=analyze_request "
            "end_to_end_latency_ms=%.3f status=%s",
            request_id,
            (perf_counter() - request_started_at) * 1000,
            request_status,
        )


@app.post(
    "/analyze/stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "Analysis progress as Server-Sent Events.",
        },
        503: {"description": "Analysis storage is unavailable before streaming."},
    },
)
async def analyze_stream(
    analysis_request: AnalysisRequest,
    request: Request,
    # Request scope keeps the session open until StreamingResponse finishes.
    session: AsyncSession = Depends(get_session, scope="request"),
) -> StreamingResponse:
    return StreamingResponse(
        stream_analysis(request, session, analysis_request.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get(
    "/analyses/{analysis_id}",
    response_model=StoredAnalysis,
    responses={
        404: {"description": "Analysis not found."},
        503: {"description": "Analysis storage is temporarily unavailable."},
    },
)
async def get_analysis(
    # Bounded to the PostgreSQL INTEGER range so out-of-range IDs are rejected
    # with 422 instead of reaching the database and surfacing as a storage 503.
    analysis_id: Annotated[int, Path(ge=1, le=POSTGRES_INTEGER_MAX)],
    session: AsyncSession = Depends(get_session),
) -> AnalysisRecord:
    record = await get_analysis_by_id(session, analysis_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found.",
        )
    return record
