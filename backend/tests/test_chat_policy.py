"""POST /api/v1/chat/policy, and the POLICY_QA branch of /chat/router: auth → permission → policy RAG → audit.

answer_question is faked (its own behaviour is tested in test_policy_rag.py), so no LLM quota is used.
"""

from sqlalchemy import select

from app.api.v1.endpoints import chat
from app.models.ai_audit_log import AiAuditLog
from app.services.ai.intent import IntentResult
from app.services.ai.permissions import PERMISSIONS, Action
from app.services.ai.policy_rag import NOT_FOUND, PolicyAnswer, Source
from tests.conftest import auth

URL = "/api/v1/chat/policy"
ANSWERED = PolicyAnswer(
    answer="Eligible employees can work from home up to 2 days a week.",
    found=True,
    sources=[Source(title="Work From Home Policy", category="WORK", filename="wfh_policy.md", section="Allowance")],
)
REFUSED = PolicyAnswer(answer=NOT_FOUND, found=False, sources=[])


def fake_rag(monkeypatch, result: PolicyAnswer | Exception):
    async def _answer(question: str) -> PolicyAnswer:
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(chat, "answer_question", _answer)


async def audit_rows(session_factory) -> list[AiAuditLog]:
    async with session_factory() as db:
        return list((await db.execute(select(AiAuditLog))).scalars())


async def test_requires_login(client):
    resp = await client.post(URL, json={"message": "How many WFH days?"})
    assert resp.status_code == 401


async def test_answer_with_sources_is_audited_as_success(client, org, session_factory, monkeypatch):
    fake_rag(monkeypatch, ANSWERED)
    resp = await client.post(URL, json={"message": "How many WFH days?"}, headers=auth(org.employee_a))

    assert resp.status_code == 200
    assert resp.json()["data"] == {
        "answer": ANSWERED.answer,
        "found": True,
        "sources": [ANSWERED.sources[0].model_dump()],
    }
    [row] = await audit_rows(session_factory)
    assert (row.user_id, row.intent, row.tool_name, row.action_status) == (
        org.employee_a.id, "POLICY_QA", "policy_rag", "SUCCESS",
    )


async def test_refusal_is_200_and_audited_as_no_answer(client, org, session_factory, monkeypatch):
    fake_rag(monkeypatch, REFUSED)
    resp = await client.post(URL, json={"message": "Stock options?"}, headers=auth(org.employee_a))

    assert resp.status_code == 200
    assert resp.json()["data"] == {"answer": NOT_FOUND, "found": False, "sources": []}
    [row] = await audit_rows(session_factory)
    assert row.action_status == "NO_ANSWER"


async def test_permission_denied_is_403_and_rag_not_called(client, org, session_factory, monkeypatch):
    fake_rag(monkeypatch, AssertionError("RAG must not run without permission"))
    monkeypatch.delitem(PERMISSIONS, Action.ASK_POLICY)  # deny by default kicks in

    resp = await client.post(URL, json={"message": "How many WFH days?"}, headers=auth(org.employee_a))

    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "AI_PERMISSION_DENIED"
    [row] = await audit_rows(session_factory)
    assert (row.intent, row.action_status) == ("POLICY_QA", "DENIED")


async def test_rag_failure_is_503_without_internals(client, org, session_factory, monkeypatch):
    fake_rag(monkeypatch, RuntimeError("429 RESOURCE_EXHAUSTED key=AIzaSECRET"))
    resp = await client.post(URL, json={"message": "How many WFH days?"}, headers=auth(org.employee_a))

    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "AI_UNAVAILABLE"
    assert "RESOURCE_EXHAUSTED" not in resp.text and "AIza" not in resp.text
    [row] = await audit_rows(session_factory)
    assert (row.action_status, row.tool_name) == ("ERROR", "policy_rag")


# ---------- the same flow through the router ----------
async def test_router_policy_qa_returns_the_rag_answer(client, org, session_factory, monkeypatch):
    async def _classify(message: str) -> IntentResult:
        return IntentResult(intent="POLICY_QA", confidence=0.9, reason="test")

    monkeypatch.setattr(chat, "classify_intent", _classify)
    fake_rag(monkeypatch, ANSWERED)

    resp = await client.post("/api/v1/chat/router", json={"message": "wfh days?"}, headers=auth(org.employee_a))

    assert resp.status_code == 200
    data = resp.json()["data"]
    assert (data["intent"], data["answer"], data["found"]) == ("POLICY_QA", ANSWERED.answer, True)
    assert data["sources"][0]["title"] == "Work From Home Policy"
    [row] = await audit_rows(session_factory)
    assert (row.intent, row.tool_name, row.action_status) == ("POLICY_QA", "policy_rag", "SUCCESS")
