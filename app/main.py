from fastapi import FastAPI

from app.schemas import AnalysisRequest, AnalysisResult
from app.services.ai_service import analyze_message

app = FastAPI()


@app.post("/analyze", response_model=AnalysisResult)
def analyze(analysis_request: AnalysisRequest) -> AnalysisResult:
    return analyze_message(analysis_request.message)
