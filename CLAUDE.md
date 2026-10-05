# PeopleOps Copilot (CB Nest + AI)

AI features (Policy RAG, SQL agent, HR action agent, RBAC, audit log) added to the existing CB Nest HRMS.
This is a learning project for an AI Engineering capstone, due 2026-10-21. **Learning comes before finishing.**

## Start of every session
Read the **Session log** at the bottom of `docs/CAPSTONE_PLAN.md` (local only, gitignored) and continue from "Next".
The assignment spec is in `docs/assignment/` (local only).

## Run
- Backend (port **8001**): `cd backend && .venv/bin/uvicorn app.main:app --port 8001 --reload`
- Frontend (port **3001**): `cd frontend && npx next dev -p 3001`
- Tests: `cd backend && .venv/bin/pytest -q`
- Ports 8000 and 3000 belong to another local project. Don't use or stop them.
- Full details are in `docs/learning/00_local_dev_runbook.md`.

## How we work
- One milestone at a time: learn → design → implement → review → test. Move on only when the user says so.
- Work in **small chunks**, a few files and one concern each. After each chunk, list what to **review deeply** (concept code) and what to skim (boilerplate).
- **Minimal code.** No speculative abstractions.
- The user writes the core AI logic (guardrails, prompts, retrieval, permissions). Claude writes the boilerplate and reviews.
- Test-first for security rules: write the failing test, then fix.

## Hard rules (breaking these fails the assignment)
- AI agents never write to the DB. Every change goes through existing backend APIs, using the user's JWT.
- The SQL agent runs one read-only `SELECT`. Never expose passwords, bank, PAN, DOB, salary or photo columns.
- Authorization is enforced in the backend, never only in the frontend.
- Never commit secrets (`backend/.env` holds the API keys). Don't break existing HRMS features.

## Stack
FastAPI + async SQLAlchemy + SQLite (`backend/storage/hrms.db`) + Alembic, and Next.js 15.
LLM: LangChain `init_chat_model`, Gemini by default (free tier, so expect rate limits), OpenAI optional.
Commits use the repo-local identity (personal email). Don't change the global git config.
