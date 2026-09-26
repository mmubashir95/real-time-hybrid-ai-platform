import asyncio
import logging
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, status

from app.schemas import AnalysisRequest, AnalysisResult
from app.services.ai_service import AIServiceError, analyze_message

# Uvicorn only configures its own loggers; without this, app INFO logs (such as
# the /async-demo start/finish lines) are dropped.
logging.basicConfig(level=logging.INFO)

app = FastAPI()
logger = logging.getLogger(__name__)

ASYNC_DEMO_DELAY_SECONDS = 3.0


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
    responses={503: {"description": "AI analysis is temporarily unavailable."}},
)
async def analyze(analysis_request: AnalysisRequest) -> AnalysisResult:
    try:
        return await analyze_message(analysis_request.message)
    except AIServiceError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI analysis is temporarily unavailable.",
        ) from error
