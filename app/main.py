from fastapi import FastAPI, HTTPException, status

from app.schemas import AnalysisRequest, AnalysisResult
from app.services.ai_service import AIServiceError, analyze_message

app = FastAPI()


@app.post("/analyze", response_model=AnalysisResult)
def analyze(analysis_request: AnalysisRequest) -> AnalysisResult:
    try:
        return analyze_message(analysis_request.message)
    except AIServiceError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI analysis is temporarily unavailable.",
        ) from error
