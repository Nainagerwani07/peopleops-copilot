"""Indexing pipeline: hr_policies rows → text → chunks → vectors → Chroma.

Run it with scripts/index_policies.py after seeding or uploading policies.
"""

from pathlib import Path

import structlog
from pypdf import PdfReader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.hr_policy import HRPolicy
from app.services.ai.chunking import Chunk, chunk_policy
from app.services.ai.embeddings import get_embeddings
from app.services.ai.vector_store import add_chunks, reset_collection

logger = structlog.get_logger()


def load_policy_text(policy: HRPolicy) -> str | None:
    """The policy's text: the content column if set, else its uploaded file (markdown, text or PDF)."""
    if policy.content:
        return policy.content
    if not policy.file_path or not Path(policy.file_path).exists():
        return None
    if policy.mime_type == "application/pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(policy.file_path).pages)
    return Path(policy.file_path).read_text(encoding="utf-8", errors="ignore")


async def index_policies(db: AsyncSession, store_path: str | None = None) -> int:
    """Rebuild the vector store from every policy in the DB. Returns the number of chunks indexed."""
    policies = (await db.execute(select(HRPolicy).order_by(HRPolicy.id))).scalars().all()

    per_policy: list[tuple[HRPolicy, list[Chunk]]] = []
    for policy in policies:
        text = load_policy_text(policy)
        if text is None:
            logger.warning("policy_not_indexed", policy_id=policy.id, title=policy.title, reason="no text or file")
            continue
        per_policy.append((policy, chunk_policy(text, policy.title)))

    all_chunks = [c for _, chunks in per_policy for c in chunks]
    if not all_chunks:
        reset_collection(store_path)
        return 0
    # One batched call for every chunk: the free tier limits requests, not chunks per request.
    vectors = get_embeddings().embed_documents([c.text for c in all_chunks])

    collection = reset_collection(store_path)
    start = 0
    for policy, chunks in per_policy:
        add_chunks(collection, policy, chunks, vectors[start : start + len(chunks)])
        start += len(chunks)
    return len(all_chunks)
