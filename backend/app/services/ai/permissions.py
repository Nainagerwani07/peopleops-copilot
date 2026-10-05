"""AI permission matrix: which role may use which AI capability.

This is the AI layer's early check, used to refuse politely before calling an LLM or an API.
The backend endpoints still enforce their own rules, so this file is defense in depth,
not the only line of defense.
Deny by default: an action missing from PERMISSIONS is allowed to nobody.
"""

from enum import StrEnum
from typing import Literal

from app.models.enums import Role


class Action(StrEnum):
    # The value is the human description used in refusal messages.
    ASK_POLICY = "ask HR policy questions"
    QUERY_HR_DATA = "query HR data"
    VIEW_SQL = "view the generated SQL"
    APPLY_LEAVE = "apply for leave"
    DECIDE_LEAVE = "approve or reject leave requests"
    CREATE_TICKET = "create tickets"
    MANAGE_TICKET = "assign or update tickets"
    CREATE_ANNOUNCEMENT = "create announcements"
    ASSIGN_PROJECT = "assign employees to projects"


DataScope = Literal["SELF", "TEAM", "ALL"]


class PermissionDenied(Exception):
    pass


PERMISSIONS: dict[Action, frozenset[Role]] = {
    Action.ASK_POLICY: frozenset({Role.EMPLOYEE, Role.MANAGER, Role.ADMIN}),
    Action.QUERY_HR_DATA: frozenset({Role.EMPLOYEE, Role.MANAGER, Role.ADMIN}),
    Action.VIEW_SQL: frozenset({Role.MANAGER, Role.ADMIN}),
    Action.APPLY_LEAVE: frozenset({Role.EMPLOYEE, Role.MANAGER, Role.ADMIN}),
    Action.DECIDE_LEAVE: frozenset({Role.MANAGER, Role.ADMIN}),
    Action.CREATE_TICKET: frozenset({Role.EMPLOYEE, Role.MANAGER, Role.ADMIN}),
    Action.MANAGE_TICKET: frozenset({Role.MANAGER, Role.ADMIN}),
    Action.CREATE_ANNOUNCEMENT: frozenset({Role.MANAGER, Role.ADMIN}),
    Action.ASSIGN_PROJECT: frozenset({Role.MANAGER, Role.ADMIN}),
}


def can(role: Role, action: Action) -> bool:
    return role in PERMISSIONS.get(action, frozenset())


def ensure(role: Role, action: Action) -> None:
    if not can(role, action):
        raise PermissionDenied(f"You do not have permission to {action.value}.")


def data_scope(role: Role) -> DataScope:
    """SELF = only my own records, TEAM = me + my direct reports, ALL = everyone."""
    if role == Role.EMPLOYEE:
        return "SELF"
    elif role == Role.MANAGER:
        return "TEAM"
    elif role == Role.ADMIN:
        return "ALL"
    else:
        raise ValueError(f"Unknown role: {role}")
