# M0 Notes: Auth & RBAC in CB Nest

## Request flow

```
POST /auth/login → verify bcrypt hash → JWT {sub: user_id, role, exp: +15 min} signed with JWT_SECRET_KEY

Any request with "Authorization: Bearer <JWT>"
  → get_current_user()  (app/services/auth.py)
       decode + verify signature/expiry ── fail → 401
       load Employee by sub from the DB ── missing → 401
  → endpoint checks current_user.role ─── not allowed → 403
```

- **The JWT is stateless.** The server keeps no sessions, so restarting it doesn't log anyone out. Only expiry or a changed secret does.
- **The role comes from the database, not the token.** `get_current_user` reloads the employee, so if a user is demoted, their old token's `role` claim stops mattering.
- **The frontend logs you out only on a 401.** If the server is down, you get a network error, not a 401, so you stay logged in and see an error message. That's correct.

## How the AI layer will reuse this

1. AI endpoints use `Depends(get_current_user)`. The role always comes from the DB, never from the chat message ("I'm an admin…").
2. The action agent calls existing APIs while **forwarding the user's own JWT**, so the backend's checks still apply even if the LLM misbehaves.
3. The AI layer has its own permission check (`permissions.py`) for early, friendly refusals. That's defense in depth: the backend remains the source of truth.

## Gap found and fixed (Option B: fix the backend, also check in the AI layer)

Before, `leaves.py` only checked `role in {ADMIN, MANAGER}`:
- any manager could see and approve **any** employee's leave (no team scope)
- a manager could **approve their own** leave

New rules (`_get_actionable_leave` and the pending-list filter):

| Who | Pending list | Approve/reject |
|---|---|---|
| Employee | 403 | 403 |
| Manager | direct reports only (`employees.manager_id = me`) | direct reports only; own leave → 403 `LEAVE_SELF_APPROVAL`; other team → **404** |
| Admin | all | all (admin's own leave is auto-approved on creation) |

**Why other-team requests get 404 and not 403:** a 403 would confirm that request #N exists. A 404 looks the same as a missing ID, so it reveals nothing (the same idea as the assignment's "don't say 'I found the record but can't show it'").

"Team" means **direct reports only**, not the whole reporting chain below a manager. In the seed data all 1,001 employees report to the single manager (id 2), and the manager's own leave goes to the admin.

Proven test-first: `backend/tests/test_leave_permissions.py`. Before the fix, 5 tests failed (exactly the gaps); after it, all 15 pass.

```bash
cd backend && .venv/bin/pytest -q
```

## Real API vs the assignment's examples

The assignment shows `PATCH /leaves/requests/{id}`. The real endpoints are `POST /leaves/requests/{id}/approve` and `POST /leaves/requests/{id}/reject`. Agent tools must call the real ones.
