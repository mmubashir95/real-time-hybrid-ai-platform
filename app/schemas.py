from pydantic import BaseModel


class AnalysisRequest(BaseModel):
    message: str


class AnalysisResult(BaseModel):
    category: str
    urgency: str
    summary: str
    entities: dict
    sentiment: str
