"""Chroma vector store for policy chunks. Indexing writes to it; the Policy RAG searches it.

Each record = chunk text + its vector + where it came from (policy title, category, filename, section),
so an answer can cite its sources without another DB query.
"""

from dataclasses import dataclass

import chromadb
from chromadb.api.models.Collection import Collection

from app.core.config import settings
from app.models.hr_policy import HRPolicy
from app.services.ai.chunking import Chunk

COLLECTION_NAME = "policy_chunks"


@dataclass(frozen=True)
class SearchHit:
    text: str
    score: float  # cosine similarity: 1.0 = same meaning, ~0 = unrelated (not a probability)
    policy_title: str
    category: str
    filename: str
    section: str | None


def _client(path: str | None) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=path or settings.vector_store_dir)


def reset_collection(path: str | None = None) -> Collection:
    """Delete and recreate the collection: indexing is a full rebuild, so no stale chunks survive."""
    client = _client(path)
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    return client.create_collection(COLLECTION_NAME, configuration={"hnsw": {"space": "cosine"}})


def get_collection(path: str | None = None) -> Collection:
    return _client(path).get_collection(COLLECTION_NAME)


def add_chunks(collection: Collection, policy: HRPolicy, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    collection.add(
        ids=[f"{policy.id}:{i}" for i in range(len(chunks))],
        embeddings=vectors,
        documents=[c.text for c in chunks],
        # Chroma metadata can't hold None, so "no section" is stored as "".
        metadatas=[
            {
                "policy_id": policy.id,
                "policy_title": policy.title,
                "category": policy.category,
                "filename": policy.original_filename or "",
                "section": c.section or "",
            }
            for c in chunks
        ],
    )


def search(collection: Collection, query_vector: list[float], k: int = 3) -> list[SearchHit]:
    """The k chunks closest in meaning to the question, best first."""
    result = collection.query(query_embeddings=[query_vector], n_results=k)
    return [
        SearchHit(
            text=text,
            score=1 - distance,  # Chroma returns cosine *distance*; we work with similarity like the demo
            policy_title=meta["policy_title"],
            category=meta["category"],
            filename=meta["filename"],
            section=meta["section"] or None,
        )
        for text, distance, meta in zip(result["documents"][0], result["distances"][0], result["metadatas"][0])
    ]
