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
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import DatabaseError, get_session
from app.models import AnalysisRecord
from app.repositories.analysis_repository import (
    get_analysis_by_id,
    save_analysis,
)
from app.schemas import AnalysisRequest, AnalysisResult, StoredAnalysis
from app.services.ai_service import AIServiceError, analyze_message

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
    started_at = perf_counter()
    try:
        analysis = await analyze_message(analysis_request.message)
    except AIServiceError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI analysis is temporarily unavailable.",
        ) from error

    latency_ms = max(0, round((perf_counter() - started_at) * 1000))
    record = await save_analysis(
        session,
        analysis_request.message,
        analysis,
        latency_ms,
    )
    response.headers["X-Analysis-ID"] = str(record.id)
    response.headers["Location"] = f"/analyses/{record.id}"
    return analysis


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
