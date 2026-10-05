---
name: checkpoint
description: End-of-session checkpoint for the PeopleOps Copilot capstone. Runs the tests, adds a short entry to the session log in docs/CAPSTONE_PLAN.md, and proposes a commit. Use when the user says /checkpoint, "wrap up", "end session" or "save progress".
---

# Checkpoint

Save where we are so the next session starts from here instead of from scratch. Keep it short.

## Steps

1. **Tests:** run `cd backend && .venv/bin/pytest -q` and note the pass/fail count. If anything fails, say so plainly. Don't hide it.
2. **What changed:** run `git status --short` and `git log --oneline -5`.
3. **Session log:** in `docs/CAPSTONE_PLAN.md`, under `## Session log`, add one entry at the **top** (newest first), in this format, with **at most 4 lines**:
   ```
   - **YYYY-MM-DD** · <milestone + chunk>. Done: <what was built/learned>. **Next:** <the exact next step>. Open: <unresolved questions, or "none">.
   ```
   If a milestone finished, also tick it in `## Progress log`.
   "Next" must be specific enough to resume cold, for example "M2 chunk 2: write chunker in policy_rag.py", not "continue M2".
4. **Commit:** if there are uncommitted changes, draft a commit message. Show it to the user and **ask before committing or pushing**. `docs/CAPSTONE_PLAN.md` is gitignored, which is expected.
5. **Report** in 3–5 lines: tests, the log entry you added, the commit status, and what to do first next session.

## Rules
- Never commit `backend/.env`, keys or `docs/assignment/`.
- Don't rewrite older log entries. Only add the new one.
