"""AI permission matrix spec (pure Python, no LLM, no DB)."""

import pytest

from app.models.enums import Role
from app.services.ai.permissions import (
    PERMISSIONS,
    Action,
    PermissionDenied,
    can,
    data_scope,
    ensure,
)

E, M, A = Role.EMPLOYEE, Role.MANAGER, Role.ADMIN

# Expected matrix: action -> roles allowed. Everything else must be denied.
EXPECTED = {
    Action.ASK_POLICY: {E, M, A},
    Action.QUERY_HR_DATA: {E, M, A},  # how much data each role sees is data_scope(), not this
    Action.VIEW_SQL: {M, A},
    Action.APPLY_LEAVE: {E, M, A},
    Action.DECIDE_LEAVE: {M, A},
    Action.CREATE_TICKET: {E, M, A},
    Action.MANAGE_TICKET: {M, A},
    Action.CREATE_ANNOUNCEMENT: {M, A},
    Action.ASSIGN_PROJECT: {M, A},
}


@pytest.mark.parametrize("action", list(Action))
@pytest.mark.parametrize("role", list(Role))
def test_matrix(role, action):
    assert can(role, action) == (role in EXPECTED[action])


def test_every_action_is_in_the_matrix():
    # A new Action added without a decision should fail here, not silently in production.
    assert set(PERMISSIONS) == set(Action)


def test_missing_action_is_denied(monkeypatch):
    # Deny by default: if an action is ever missing from PERMISSIONS, nobody gets it.
    monkeypatch.delitem(PERMISSIONS, Action.DECIDE_LEAVE)
    assert can(Role.ADMIN, Action.DECIDE_LEAVE) is False


def test_ensure_raises_clear_refusal():
    with pytest.raises(PermissionDenied) as exc:
        ensure(Role.EMPLOYEE, Action.DECIDE_LEAVE)
    assert str(exc.value) == "You do not have permission to approve or reject leave requests."


def test_ensure_allows_silently():
    assert ensure(Role.MANAGER, Action.DECIDE_LEAVE) is None


@pytest.mark.parametrize(("role", "scope"), [(E, "SELF"), (M, "TEAM"), (A, "ALL")])
def test_data_scope(role, scope):
    assert data_scope(role) == scope
