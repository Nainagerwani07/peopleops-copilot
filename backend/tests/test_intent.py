"""Intent classification spec. These call the real LLM, so run them with: pytest -m llm"""

import pytest

from app.services.ai.intent import classify_intent

pytestmark = pytest.mark.llm


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        # Routing table from the assignment
        ("What is the leave policy?", "POLICY_QA"),
        ("Who is assigned to Project X?", "SQL_QUERY"),
        ("Apply leave for tomorrow.", "HR_ACTION"),
        ("Create a ticket for VPN issue.", "HR_ACTION"),
        ("Show employees who know LangChain.", "SQL_QUERY"),
        # The tricky pair: same topic, different source of truth
        ("How many sick leaves do I get per year?", "POLICY_QA"),  # rule → policy documents
        ("How many sick leaves do I have left?", "SQL_QUERY"),  # my data → database
        # Not an HR request at all
        ("What's the weather in Paris today?", "UNKNOWN"),
    ],
)
async def test_classify_intent(message, expected):
    result = await classify_intent(message)
    assert result.intent == expected, f"{message!r} → {result.intent} ({result.reason})"
    assert 0.0 <= result.confidence <= 1.0
    assert result.reason
