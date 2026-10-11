"""Learning tool: ask policy questions and watch every step of policy_rag.answer_question().

It runs the REAL answer_question(); it only wraps the embedding model, search() and the chat model to print
what goes in and out of each step. Nothing is re-implemented here, so what you see is what the code does.

Run from backend/:  PYTHONPATH=. .venv/bin/python scripts/trace_policy_rag.py ["question" ...]
No question → interactive: type questions, empty line to quit.
Builds storage/vector_store from data/policies/*.md the first time (one embedding call, ~42 chunks).
Each question costs 1 embedding call + at most 1 LLM call (free tier).
"""

import asyncio
import sys
import time
from pathlib import Path

from app.models.hr_policy import HRPolicy
from app.services.ai import policy_rag
from app.services.ai.chunking import chunk_policy
from app.services.ai.embeddings import get_embeddings
from app.services.ai.llm import get_chat_model
from app.services.ai.vector_store import add_chunks, get_collection, reset_collection, search

POLICY_DIR = Path(__file__).resolve().parents[1] / "data" / "policies"
LINE = "─" * 100


def step(n: int, title: str) -> None:
    print(f"\n{LINE}\n STEP {n}: {title}\n{LINE}")


def indent(text: str, prefix: str = "   │ ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


# ---------- build the index from the .md files (no DB needed) ----------
def ensure_index() -> None:
    try:
        if get_collection().count() > 0:
            return
    except Exception:
        pass  # collection doesn't exist yet
    print("Vector store is empty: indexing data/policies/*.md ...")
    docs = sorted(POLICY_DIR.glob("*.md"))
    per_policy = []
    for i, path in enumerate(docs, start=1):
        md = path.read_text(encoding="utf-8")
        title = md.splitlines()[0].removeprefix("# ")
        policy = HRPolicy(id=i, title=title, category=title.split()[0].upper(), original_filename=path.name)
        per_policy.append((policy, chunk_policy(md, title)))
    all_chunks = [c for _, chunks in per_policy for c in chunks]
    vectors = get_embeddings().embed_documents([c.text for c in all_chunks])
    collection = reset_collection()
    start = 0
    for policy, chunks in per_policy:
        add_chunks(collection, policy, chunks, vectors[start : start + len(chunks)])
        start += len(chunks)
    print(f"Indexed {len(all_chunks)} chunks from {len(docs)} policies.\n")


# ---------- wrappers that print, then call the real thing ----------
class TracingEmbeddings:
    async def aembed_query(self, text: str) -> list[float]:
        step(1, "embed the question (model call: gemini-embedding-001, task type RETRIEVAL_QUERY)")
        t = time.perf_counter()
        vector = await get_embeddings().aembed_query(text)
        print(f"   question: {text!r}")
        print(f"   → a vector of {len(vector)} numbers: [{', '.join(f'{x:+.4f}' for x in vector[:6])}, ...]")
        print(f"   ({(time.perf_counter() - t) * 1000:.0f} ms)")
        return vector


def tracing_search(collection, query_vector, k=3):
    step(2, f"search Chroma: top {k} of {collection.count()} chunks by cosine similarity (maths, no model)")
    hits = search(collection, query_vector, k)
    for rank, h in enumerate(hits, start=1):
        verdict = "✅ passes" if h.score >= policy_rag.MIN_SCORE else "❌ below"
        print(f"   #{rank}  score {h.score:.3f}  {verdict} MIN_SCORE {policy_rag.MIN_SCORE}   {h.policy_title} > {h.section}")
        print(indent(h.text[:300] + ("..." if len(h.text) > 300 else "")))
    step(3, "layer 1: score threshold (code)")
    kept = [h for h in hits if h.score >= policy_rag.MIN_SCORE]
    if kept:
        print(f"   {len(kept)} of {len(hits)} chunks pass → they become excerpts 1..{len(kept)} for the LLM")
    else:
        print("   nothing passes → REFUSE now. The LLM is never called (free, and can't be talked around).")
    return hits


class TracingLLM:
    def __init__(self):
        self.real = get_chat_model()

    def with_structured_output(self, schema):
        self.real = self.real.with_structured_output(schema)
        return self

    async def ainvoke(self, messages):
        step(4, "layer 2: the exact messages sent to the LLM")
        for role, text in messages:
            print(f"   [{role}]")
            print(indent(text.strip()))
        t = time.perf_counter()
        result = await self.real.ainvoke(messages)
        step(5, f"LLM structured output (LLMAnswer), {(time.perf_counter() - t) * 1000:.0f} ms")
        print(f"   found            = {result.found}")
        print(f"   answer           = {result.answer!r}")
        print(f"   used_excerpt_ids = {result.used_excerpt_ids}")
        return result


async def trace(question: str) -> None:
    print(f"\n{'═' * 100}\n QUESTION: {question}\n{'═' * 100}")
    result = await policy_rag.answer_question(question)
    step(6, "citation check + final PolicyAnswer (what /chat/policy will return)")
    print(f"   found   = {result.found}")
    print(f"   answer  = {result.answer}")
    for s in result.sources:
        print(f"   source  = {s.title} > {s.section}   ({s.filename})")
    if not result.sources:
        print("   sources = []")


async def main() -> None:
    ensure_index()
    policy_rag.get_embeddings = TracingEmbeddings
    policy_rag.search = tracing_search
    policy_rag.get_chat_model = TracingLLM

    if len(sys.argv) > 1:
        for question in sys.argv[1:]:
            await trace(question)
        return
    while question := input("\nAsk a policy question (empty line to quit): ").strip():
        await trace(question)


if __name__ == "__main__":
    asyncio.run(main())
