"""AI audit log spec."""

import json

import pytest
from sqlalchemy import select

from app.core.security import create_access_token
from app.models.ai_audit_log import AiAuditLog
from app.services.ai.audit import log_ai_event, redact

# ---- redact() -----------------------------------------------------------------------------


def test_redact_jwt():
    token = create_access_token("3", "EMPLOYEE")
    out = redact(f"my token is {token} please help")
    assert token not in out
    assert out == "my token is [REDACTED_TOKEN] please help"


def test_redact_pan():
    assert redact("What is the PAN ABCDE1234F for?") == "What is the PAN [REDACTED_PAN] for?"


@pytest.mark.parametrize("number", ["00031000000002", "123456789", "9876543210"])
def test_redact_long_numbers(number):
    assert redact(f"Account {number} is wrong") == "Account [REDACTED_NUMBER] is wrong"


@pytest.mark.parametrize(
    "text",
    [
        "How many sick leaves do I get?",
        "Apply leave from 2026-11-02 to 2026-11-03",
        "Mark ticket 42 as resolved",
        "Who is on project 12345678?",  # 8 digits: below the threshold, kept
    ],
)
def test_redact_keeps_normal_text(text):
    assert redact(text) == text


# ---- log_ai_event() -----------------------------------------------------------------------


async def test_log_ai_event_writes_redacted_row(session_factory, org):
    async with session_factory() as db:
        await log_ai_event(
            db,
            user=org.employee_a,
            message="Show salary for PAN ABCDE1234F",
            status="DENIED",
            intent="SQL_QUERY",
            tool_name="sql_agent",
            records_accessed=[7, 9],
        )
        row = (await db.execute(select(AiAuditLog))).scalar_one()

    assert row.user_id == org.employee_a.id
    assert row.role == "EMPLOYEE"
    assert row.message == "Show salary for PAN [REDACTED_PAN]"
    assert row.action_status == "DENIED"
    assert json.loads(row.records_accessed) == [7, 9]
    assert row.created_at is not None
