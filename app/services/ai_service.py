import json

from openai import OpenAI

from app.schemas import AnalysisResult

MODEL = "gpt-4o-mini"
ANALYSIS_INSTRUCTIONS = """Analyze the customer support message.

Return the category, urgency, summary, entities, and sentiment.

Urgency must be low, medium, or high. Use high only when the customer was
charged incorrectly, is locked out of their account, or cannot use the service
at all.

Entities are specific values stated in the message, such as order IDs, email
addresses, product names, monetary amounts, and dates. Use short snake_case
entity types such as order_id, email, product, amount, and date. Return an
empty list when the message contains no such values.
"""
ANALYSIS_SCHEMA = AnalysisResult.model_json_schema()
ANALYSIS_SCHEMA["additionalProperties"] = False
# OpenAI Structured Outputs requires every object to declare its keys, so a
# free-form dict cannot be requested directly. The model returns entities as
# type/value pairs, which analyze_message converts to the public dict contract.
ANALYSIS_SCHEMA["properties"]["entities"] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "type": {"type": "string"},
            "value": {"type": "string"},
        },
        "required": ["type", "value"],
        "additionalProperties": False,
    },
}
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
    analysis = json.loads(response.output_text)
    analysis["entities"] = {
        entity["type"]: entity["value"] for entity in analysis["entities"]
    }
    return AnalysisResult.model_validate(analysis)
