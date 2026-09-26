from datetime import datetime

from pydantic import BaseModel, ConfigDict


class AnalysisRequest(BaseModel):
    message: str


class AnalysisResult(BaseModel):
    category: str
    urgency: str
    summary: str
    entities: dict
    sentiment: str


class StoredAnalysis(AnalysisResult):
    model_config = ConfigDict(from_attributes=True)

    id: int
    message: str
    created_at: datetime
    latency_ms: int
