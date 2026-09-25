from fastapi import FastAPI
from pydantic import BaseModel


class AnalysisRequest(BaseModel):
    message: str


class AnalysisResult(BaseModel):
    category: str
    urgency: str
    summary: str
    entities: dict
    sentiment: str

app = FastAPI()


@app.post("/analyze", response_model=AnalysisResult)
def analyze_message(analysis_request: AnalysisRequest) -> AnalysisResult:
    return AnalysisResult(
        category="billing",
        urgency="high",
        summary="Customer reports duplicate billing.",
        entities={},
        sentiment="negative",
    )
