"""AI audit log: one row per AI interaction (who asked what, what the AI did, how it ended).

This writes only to the AI layer's own ai_audit_logs table, never to HR business tables.
"""

import json
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.ai_audit_log import AiAuditLog
from app.models.employee import Employee

# SUCCESS: answered / action done   DENIED: permission check refused   BLOCKED: guardrail stopped it
# NO_ANSWER: RAG had no grounded answer   ERROR: something failed (LLM, API, DB)
AuditStatus = Literal["SUCCESS", "DENIED", "BLOCKED", "NO_ANSWER", "ERROR"]

MAX_MESSAGE_CHARS = 2000


def redact(text: str) -> str:
    """Mask secrets and sensitive identifiers a user might paste into a chat message.

    YOUR TASK (M1 chunk 3). Spec: tests/test_audit.py::test_redact_*
    Replace, using regular expressions (`re.sub`):
      - JWTs (three base64url parts joined by dots, starting with "eyJ")  → "[REDACTED_TOKEN]"
      - PAN numbers (5 uppercase letters, 4 digits, 1 uppercase letter)   → "[REDACTED_PAN]"
      - long digit runs, 9 or more digits (bank accounts, phone numbers)  → "[REDACTED_NUMBER]"
    Ordinary text, dates ("2026-11-02") and short ids ("ticket 42") must stay unchanged.
    """
    raise NotImplementedError


async def log_ai_event(
    db: AsyncSession,
    *,
    user: Employee,
    message: str,
    status: AuditStatus,
    intent: str | None = None,
    tool_name: str | None = None,
    records_accessed: list[int] | None = None,
) -> AiAuditLog:
    entry = AiAuditLog(
        user_id=user.id,
        role=user.role.value,
        message=redact(message)[:MAX_MESSAGE_CHARS],
        intent=intent,
        tool_name=tool_name,
        action_status=status,
        records_accessed=json.dumps(records_accessed) if records_accessed else None,
    )
    db.add(entry)
    await db.commit()
    return entry
