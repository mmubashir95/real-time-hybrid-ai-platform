from openai import OpenAI

from app.schemas import AnalysisResult

MODEL = "gpt-4o-mini"
ANALYSIS_INSTRUCTIONS = """Analyze the customer support message.

Return the category, urgency, summary, entities, and sentiment.
"""
ANALYSIS_SCHEMA = AnalysisResult.model_json_schema()
ANALYSIS_SCHEMA["additionalProperties"] = False
# OpenAI Structured Outputs requires every object to reject undeclared fields.
# Keep the public entities contract as a dict while constraining this phase to {}.
ANALYSIS_SCHEMA["properties"]["entities"]["properties"] = {}
ANALYSIS_SCHEMA["properties"]["entities"]["required"] = []
ANALYSIS_SCHEMA["properties"]["entities"]["additionalProperties"] = False
_client = OpenAI()


def analyze_message(message: str) -> AnalysisResult:
    response = _client.responses.create(
        model=MODEL,
        instructions=ANALYSIS_INSTRUCTIONS,
        input=message,
        text={
            "format": {
                "type": "json_schema",
                "name": "analysis_result",
                "strict": True,
                "schema": ANALYSIS_SCHEMA,
            }
        },
    )
    return AnalysisResult.model_validate_json(response.output_text)
