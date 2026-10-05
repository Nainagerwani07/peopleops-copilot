"""Shared test fixtures.

Every test gets a fresh, empty SQLite database in a temp folder, so tests never touch
storage/hrms.db and never depend on seed data (which changes on every reseed).
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import date

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.models  # noqa: F401  (registers every table on Base.metadata)
from app.core.security import create_access_token, hash_password
from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.models.employee import Employee
from app.models.enums import Role

# Free-tier quota is tiny: replay identical LLM calls (same prompt + message + model) from disk.
# A changed prompt is a new cache key, so it still hits the real API.
set_llm_cache(SQLiteCache(database_path=".pytest_cache/llm_cache.db"))


@pytest.fixture
async def session_factory(tmp_path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)

    async def _override_get_db():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    yield factory
    app.dependency_overrides.clear()
    await engine.dispose()


@pytest.fixture
async def client(session_factory) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@dataclass
class Org:
    """A small org chart:

        admin
        manager_a ── employee_a
        manager_b ── employee_b
    """

    admin: Employee
    manager_a: Employee
    manager_b: Employee
    employee_a: Employee
    employee_b: Employee


@pytest.fixture
async def org(session_factory) -> Org:
    async with session_factory() as db:

        def make(name: str, role: Role, manager: Employee | None = None) -> Employee:
            return Employee(
                name=name,
                email=f"{name}@test.dev",
                hashed_password=hash_password("password123"),
                role=role,
                manager_id=manager.id if manager else None,
                joining_date=date(2024, 1, 1),
            )

        admin = make("admin", Role.ADMIN)
        manager_a = make("manager_a", Role.MANAGER)
        manager_b = make("manager_b", Role.MANAGER)
        db.add_all([admin, manager_a, manager_b])
        await db.flush()
        employee_a = make("employee_a", Role.EMPLOYEE, manager_a)
        employee_b = make("employee_b", Role.EMPLOYEE, manager_b)
        db.add_all([employee_a, employee_b])
        await db.commit()
        return Org(admin, manager_a, manager_b, employee_a, employee_b)


def auth(user: Employee) -> dict[str, str]:
    """Authorization header for a user, the same token the login endpoint would issue."""
    return {"Authorization": f"Bearer {create_access_token(str(user.id), user.role.value)}"}
