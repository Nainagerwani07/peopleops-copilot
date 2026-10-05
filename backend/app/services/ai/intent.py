"""Intent classification: decide which agent should handle a chat message.

YOUR TASK (M1 chunk 1). Spec: tests/test_intent.py (run with `.venv/bin/pytest -m llm`)

1. Define `IntentResult`, a Pydantic model with:
     intent:     one of "POLICY_QA" | "SQL_QUERY" | "HR_ACTION" | "UNKNOWN"   (hint: typing.Literal)
     confidence: float between 0 and 1                                        (hint: Field(ge=..., le=...))
     reason:     str, one short sentence
   Field descriptions are sent to the LLM as part of the schema, so write them for the model.

2. Write a system prompt that explains each intent. Think about the tricky pair in the tests:
   "how many sick leaves do I GET" (a rule) vs "how many do I HAVE LEFT" (my data).

3. Implement classify_intent:
     llm = get_chat_model().with_structured_output(IntentResult)
     return await llm.ainvoke([("system", ...), ("human", message)])
"""

from app.services.ai.llm import get_chat_model  # noqa: F401


async def classify_intent(message: str):
    raise NotImplementedError
