"""Rebuild the policy vector store. Run from backend/:  PYTHONPATH=. .venv/bin/python scripts/index_policies.py"""

import asyncio

from app.core.config import settings
from app.db.session import SessionLocal
from app.services.ai.policy_index import index_policies


async def main() -> None:
    async with SessionLocal() as db:
        count = await index_policies(db)
    print(f"Indexed {count} chunks into {settings.vector_store_dir}")


if __name__ == "__main__":
    asyncio.run(main())
