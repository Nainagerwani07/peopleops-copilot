"""Indexing pipeline + vector store spec. Fake embeddings: no API calls, no quota, deterministic."""

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.models.hr_policy import HRPolicy
from app.services.ai import policy_index
from app.services.ai.policy_index import index_policies
from app.services.ai.vector_store import get_collection, search

FAKE = DeterministicFakeEmbedding(size=16)  # same text → same vector, so exact matches score ~1.0

LEAVE = "# Leave Policy\n\n## Sick Leave\nEmployees get 12 days of sick leave.\n\n## Approval\nManagers approve leave.\n"


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    monkeypatch.setattr(policy_index, "get_embeddings", lambda: FAKE)


async def add_policies(session_factory, *policies: HRPolicy) -> None:
    async with session_factory() as db:
        db.add_all(policies)
        await db.commit()


async def run_index(session_factory, store: str) -> int:
    async with session_factory() as db:
        return await index_policies(db, store_path=store)


async def test_indexes_every_chunk_with_its_source(session_factory, tmp_path):
    await add_policies(
        session_factory,
        HRPolicy(id=1, title="Leave Policy", category="LEAVE", content=LEAVE, original_filename="seed_policy_01.md"),
    )
    assert await run_index(session_factory, str(tmp_path / "vs")) == 2

    query = FAKE.embed_query("Leave Policy > Sick Leave\nEmployees get 12 days of sick leave.")
    hits = search(get_collection(str(tmp_path / "vs")), query, k=2)
    best = hits[0]
    assert best.section == "Sick Leave"
    assert best.score == pytest.approx(1.0, abs=1e-3)
    assert (best.policy_title, best.category, best.filename) == ("Leave Policy", "LEAVE", "seed_policy_01.md")
    assert hits[0].score >= hits[1].score  # best first


async def test_reads_uploaded_file_when_content_is_empty(session_factory, tmp_path):
    path = tmp_path / "id_card.md"
    path.write_text("# ID Card Policy\n\nEmployees must carry ID cards.\n")
    await add_policies(
        session_factory,
        HRPolicy(id=1, title="ID Card Policy", category="ID", file_path=str(path), mime_type="text/markdown"),
    )
    assert await run_index(session_factory, str(tmp_path / "vs")) == 1
    hit = search(get_collection(str(tmp_path / "vs")), FAKE.embed_query("x"), k=1)[0]
    assert hit.section is None  # stored as "" in Chroma, read back as None
    assert "Employees must carry ID cards." in hit.text


async def test_policy_with_missing_file_is_skipped_not_fatal(session_factory, tmp_path):
    await add_policies(
        session_factory,
        HRPolicy(id=1, title="Leave Policy", category="LEAVE", content=LEAVE),
        HRPolicy(id=2, title="Ghost Policy", category="GHOST", file_path=str(tmp_path / "missing.md")),
    )
    assert await run_index(session_factory, str(tmp_path / "vs")) == 2


async def test_reindex_is_a_full_rebuild(session_factory, tmp_path):
    store = str(tmp_path / "vs")
    await add_policies(session_factory, HRPolicy(id=1, title="Leave Policy", category="LEAVE", content=LEAVE))
    await run_index(session_factory, store)
    await run_index(session_factory, store)
    assert get_collection(store).count() == 2  # not 4: no duplicate or stale chunks
