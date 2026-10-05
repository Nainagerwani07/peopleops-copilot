"""Intent classification: decide which agent (policy RAG, SQL, HR action) should handle a chat message."""

from typing import Literal

from pydantic import BaseModel, Field

from app.services.ai.llm import get_chat_model


class IntentResult(BaseModel):
    intent: Literal["POLICY_QA", "SQL_QUERY", "HR_ACTION", "UNKNOWN"] = Field(
        description="The intent category that best matches the employee's request."
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="How confident you are in the classification, between 0 and 1.",
    )
    reason: str = Field(
        description="One short sentence explaining why this intent was selected."
    )


SYSTEM_PROMPT = """
You are an intent classifier for an internal company copilot used by employees.

Classify each employee message into exactly one of these intents:

POLICY_QA:
The employee is asking about company rules, policies, entitlements, eligibility,
or procedures documented in company policy.
This asks what is generally allowed, required, or provided.

Examples:
- "How many earned leaves am I entitled to each year?"
- "Am I allowed to carry forward annual leave?"
- "What is the maternity leave policy?"
- "How do I apply for parental leave?"

SQL_QUERY:
The employee wants to read or look up existing company records or personal data.
This includes employees, departments, projects, skills, leave balances,
requests, tickets, and other stored records.

Typical wording includes "show me", "who is", "how many do I have",
"how many are left", "what is my balance", or questions about existing data.

Examples:
- "How much earned leave have I used so far?"
- "Show me my open tickets."
- "Who works on Project Alpha?"
- "Which employees know Python?"

HR_ACTION:
The employee wants the system to create, update, approve, reject, assign,
or otherwise change something.

Examples:
- "Request two days of casual leave next week."
- "Raise a ticket because my laptop will not charge."
- "Approve Ravi's leave request."
- "Assign Priya to Project Alpha."
- "Post an announcement."

UNKNOWN:
The message does not fit POLICY_QA, SQL_QUERY, or HR_ACTION.

Important distinctions:
- Asking what someone GETS or is ENTITLED TO is usually POLICY_QA.
- Asking what someone HAS, HAS LEFT, USED, or what currently EXISTS is SQL_QUERY.
- Asking to CREATE, CHANGE, UPDATE, APPROVE, REJECT, ASSIGN, or POST something is HR_ACTION.

For example:
"How many earned leaves do I get?" -> POLICY_QA
"How many earned leaves do I have left?" -> SQL_QUERY

Choose the intent based on what the employee wants the system to do,
not merely on topic words such as "leave", "ticket", or "employee".
"""


async def classify_intent(message: str) -> IntentResult:
    llm = get_chat_model().with_structured_output(IntentResult)

    return await llm.ainvoke([
        ("system", SYSTEM_PROMPT),
        ("human", message),
    ])
