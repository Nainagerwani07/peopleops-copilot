"""AI audit log: one row per AI interaction (who asked what, what the AI did, how it ended).

This writes only to the AI layer's own ai_audit_logs table, never to HR business tables.
"""

import json
import re
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

    Masks by shape, not meaning: spaced-out numbers ("1234 5678 9012") and lowercase PANs
    are not caught. Masking slightly too much is the safe direction.
    """
    text = re.sub(
        r"eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+",
        "[REDACTED_TOKEN]",
        text
    )
    text = re.sub(
        r"\b[A-Z]{5}\d{4}[A-Z]\b",
        "[REDACTED_PAN]",
        text
    )
    text = re.sub(
        r"\b\d{9,}\b",
        "[REDACTED_NUMBER]",
        text
    )
    return text


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
