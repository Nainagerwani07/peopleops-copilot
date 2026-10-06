"""POST /api/v1/chat/router wiring: auth → intent → permission → audit.

classify_intent is faked, so these tests use no LLM quota and are deterministic.
"""

import pytest
from sqlalchemy import select

from app.api.v1.endpoints import chat
from app.models.ai_audit_log import AiAuditLog
from app.services.ai.intent import IntentResult
from app.services.ai.permissions import PERMISSIONS, Action
from tests.conftest import auth

URL = "/api/v1/chat/router"


def fake_intent(monkeypatch, intent: str):
    async def _classify(message: str) -> IntentResult:
        return IntentResult(intent=intent, confidence=0.9, reason="test")

    monkeypatch.setattr(chat, "classify_intent", _classify)


async def audit_rows(session_factory) -> list[AiAuditLog]:
    async with session_factory() as db:
        return list((await db.execute(select(AiAuditLog))).scalars())


async def test_requires_login(client):
    resp = await client.post(URL, json={"message": "What is the leave policy?"})
    assert resp.status_code == 401


async def test_rejects_empty_message(client, org):
    resp = await client.post(URL, json={"message": ""}, headers=auth(org.employee_a))
    assert resp.status_code == 422


@pytest.mark.parametrize("intent", ["POLICY_QA", "SQL_QUERY", "HR_ACTION", "UNKNOWN"])
async def test_routes_and_audits_success(client, org, session_factory, monkeypatch, intent):
    fake_intent(monkeypatch, intent)
    resp = await client.post(URL, json={"message": "hello"}, headers=auth(org.employee_a))

    assert resp.status_code == 200
    assert resp.json()["data"]["intent"] == intent
    [row] = await audit_rows(session_factory)
    assert (row.user_id, row.intent, row.action_status) == (org.employee_a.id, intent, "SUCCESS")


async def test_permission_denied_is_403_and_audited(client, org, session_factory, monkeypatch):
    fake_intent(monkeypatch, "POLICY_QA")
    monkeypatch.delitem(PERMISSIONS, Action.ASK_POLICY)  # deny by default kicks in

    resp = await client.post(URL, json={"message": "Leave policy? PAN ABCDE1234F"}, headers=auth(org.employee_a))

    assert resp.status_code == 403
    assert resp.json()["error"] == {
        "code": "AI_PERMISSION_DENIED",
        "message": "You do not have permission to ask HR policy questions.",
    }
    [row] = await audit_rows(session_factory)
    assert row.action_status == "DENIED"
    assert row.message == "Leave policy? PAN [REDACTED_PAN]"


async def test_llm_failure_is_503_without_internals(client, org, session_factory, monkeypatch):
    async def _boom(message: str):
        raise RuntimeError("429 RESOURCE_EXHAUSTED key=AIzaSECRET")

    monkeypatch.setattr(chat, "classify_intent", _boom)
    resp = await client.post(URL, json={"message": "hello"}, headers=auth(org.employee_a))

    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "AI_UNAVAILABLE"
    assert "RESOURCE_EXHAUSTED" not in resp.text and "AIza" not in resp.text
    [row] = await audit_rows(session_factory)
    assert (row.action_status, row.tool_name) == ("ERROR", "intent_classifier")
