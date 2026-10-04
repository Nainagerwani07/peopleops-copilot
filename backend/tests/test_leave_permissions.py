"""Leave approval permissions.

Rules (see docs/learning/01_auth_and_rbac.md):
- EMPLOYEE: cannot view pending approvals or approve/reject anything.
- MANAGER: only leave of their direct reports (employees.manager_id == manager), never their own.
  A request outside the team returns 404, the same as a missing id, so its existence is not leaked.
- ADMIN: any request.
"""

import pytest

from tests.conftest import auth

LEAVE = {"leave_type": "CASUAL", "start_date": "2026-11-02", "end_date": "2026-11-03", "reason": "Personal work"}


async def apply_leave(client, user) -> int:
    resp = await client.post("/api/v1/leaves/requests", json=LEAVE, headers=auth(user))
    assert resp.status_code == 200, resp.text
    return resp.json()["data"]["id"]


async def pending_ids(client, user) -> set[int]:
    resp = await client.get("/api/v1/leaves/requests/pending", headers=auth(user))
    assert resp.status_code == 200, resp.text
    return {item["id"] for item in resp.json()["data"]["items"]}


# ---- Pending list -------------------------------------------------------------------------


async def test_manager_sees_only_direct_reports_pending(client, org):
    team_leave = await apply_leave(client, org.employee_a)
    other_team_leave = await apply_leave(client, org.employee_b)
    own_leave = await apply_leave(client, org.manager_a)

    visible = await pending_ids(client, org.manager_a)

    assert team_leave in visible
    assert other_team_leave not in visible
    assert own_leave not in visible


async def test_admin_sees_all_pending(client, org):
    ids = {
        await apply_leave(client, org.employee_a),
        await apply_leave(client, org.employee_b),
        await apply_leave(client, org.manager_a),
    }
    assert ids <= await pending_ids(client, org.admin)


async def test_employee_cannot_view_pending(client, org):
    resp = await client.get("/api/v1/leaves/requests/pending", headers=auth(org.employee_a))
    assert resp.status_code == 403


# ---- Approve / reject ---------------------------------------------------------------------


@pytest.mark.parametrize("action", ["approve", "reject"])
async def test_manager_can_act_on_direct_report(client, org, action):
    leave_id = await apply_leave(client, org.employee_a)
    resp = await client.post(f"/api/v1/leaves/requests/{leave_id}/{action}", headers=auth(org.manager_a))
    assert resp.status_code == 200, resp.text
    assert resp.json()["data"]["status"] == ("APPROVED" if action == "approve" else "REJECTED")
    assert resp.json()["data"]["approver_id"] == org.manager_a.id


@pytest.mark.parametrize("action", ["approve", "reject"])
async def test_manager_cannot_act_on_other_team(client, org, action):
    leave_id = await apply_leave(client, org.employee_b)
    resp = await client.post(f"/api/v1/leaves/requests/{leave_id}/{action}", headers=auth(org.manager_a))
    assert resp.status_code == 404
    assert resp.json()["detail"]["error"]["code"] == "LEAVE_NOT_FOUND"


@pytest.mark.parametrize("action", ["approve", "reject"])
async def test_manager_cannot_act_on_own_leave(client, org, action):
    leave_id = await apply_leave(client, org.manager_a)
    resp = await client.post(f"/api/v1/leaves/requests/{leave_id}/{action}", headers=auth(org.manager_a))
    assert resp.status_code == 403
    assert resp.json()["detail"]["error"]["code"] == "LEAVE_SELF_APPROVAL"


@pytest.mark.parametrize("action", ["approve", "reject"])
async def test_employee_cannot_act(client, org, action):
    leave_id = await apply_leave(client, org.employee_a)
    resp = await client.post(f"/api/v1/leaves/requests/{leave_id}/{action}", headers=auth(org.employee_a))
    assert resp.status_code == 403


async def test_admin_can_approve_managers_leave(client, org):
    leave_id = await apply_leave(client, org.manager_a)
    resp = await client.post(f"/api/v1/leaves/requests/{leave_id}/approve", headers=auth(org.admin))
    assert resp.status_code == 200, resp.text
    assert resp.json()["data"]["status"] == "APPROVED"


# ---- Existing behaviour that must not change --------------------------------------------


async def test_approve_reduces_balance(client, org):
    leave_id = await apply_leave(client, org.employee_a)  # 2 days CASUAL
    await client.post(f"/api/v1/leaves/requests/{leave_id}/approve", headers=auth(org.manager_a))

    resp = await client.get("/api/v1/leaves/balances/me", headers=auth(org.employee_a))
    casual = next(b for b in resp.json()["data"] if b["leave_type"] == "CASUAL")
    assert casual["used"] == 2.0


async def test_cannot_approve_twice(client, org):
    leave_id = await apply_leave(client, org.employee_a)
    await client.post(f"/api/v1/leaves/requests/{leave_id}/approve", headers=auth(org.manager_a))
    resp = await client.post(f"/api/v1/leaves/requests/{leave_id}/approve", headers=auth(org.manager_a))
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"]["code"] == "LEAVE_NOT_PENDING"


async def test_missing_request_returns_404(client, org):
    resp = await client.post("/api/v1/leaves/requests/99999/approve", headers=auth(org.admin))
    assert resp.status_code == 404
