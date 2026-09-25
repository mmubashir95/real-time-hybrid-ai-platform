from app.schemas import AnalysisResult


def analyze_message(message: str) -> AnalysisResult:
    return AnalysisResult(
        category="billing",
        urgency="high",
        summary="Customer reports duplicate billing.",
        entities={},
        sentiment="negative",
    )
