"""Retrieval eval: does search find the right chunk, and which MIN_SCORE separates answerable from unanswerable?

Search only, no LLM. Uses the real vector store and search(), so it measures what policy_rag sees.
  answerable question: "hit@k" = a top-k chunk contains expected_fact;
                       "survives t" = a chunk containing expected_fact scores >= t (layer 1 keeps it).
  unanswerable question: "blocked at t" = the top score is < t (layer 1 refuses without the LLM).

Run from backend/:  PYTHONPATH=. .venv/bin/python scripts/eval_retrieval.py [--held-out]
Question vectors are cached in storage/vector_store/ (gitignored), so re-runs cost no API quota.
"""

import hashlib
import json
import runpy
import sys
from pathlib import Path

from app.core.config import settings
from app.services.ai import policy_rag
from app.services.ai.embeddings import get_embeddings
from app.services.ai.vector_store import get_collection, search
from scripts.trace_policy_rag import ensure_index

DATASET = Path(__file__).resolve().parents[1] / "data" / "eval" / "policy_retrieval.py"
CACHE = Path("storage/vector_store/eval_retrieval_cache.json")
THRESHOLDS = [round(0.50 + 0.01 * i, 2) for i in range(26)]  # 0.50 .. 0.75


def embed_questions(questions: list[str]) -> list[list[float]]:
    """One batched call for all uncached questions. Same task type as policy_rag's embed_query."""
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    key = lambda q: hashlib.sha256(f"{settings.embedding_model}:query:{q}".encode()).hexdigest()  # noqa: E731
    missing = [q for q in dict.fromkeys(questions) if key(q) not in cache]
    if missing:
        vectors = get_embeddings().embed_documents(missing, task_type="RETRIEVAL_QUERY")
        cache.update({key(q): v for q, v in zip(missing, vectors)})
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache))
    return [cache[key(q)] for q in questions]


def run(dataset: list[tuple[str, str | None, str]]) -> list[dict]:
    collection = get_collection()
    vectors = embed_questions([q for q, _, _ in dataset])
    rows = []
    for (question, fact, note), vector in zip(dataset, vectors):
        hits = search(collection, vector, k=policy_rag.TOP_K)
        answer_ranks = [i for i, h in enumerate(hits, start=1) if fact and fact in h.text]
        rows.append({
            "question": question,
            "note": note,
            "answerable": fact is not None,
            "top_score": hits[0].score,
            "top_section": f"{hits[0].policy_title} > {hits[0].section}",
            "answer_rank": answer_ranks[0] if answer_ranks else None,
            "answer_score": hits[answer_ranks[0] - 1].score if answer_ranks else None,
        })
    return rows


def report(rows: list[dict], thresholds: list[float]) -> None:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]

    print(f"\nPER QUESTION (sorted by top score; current MIN_SCORE = {policy_rag.MIN_SCORE})")
    print(f"{'top':>6}  {'kind':<6} {'rank':>4}  {'note':<14} question  →  top chunk")
    for r in sorted(rows, key=lambda r: r["top_score"], reverse=True):
        kind = "ANS" if r["answerable"] else "NONE"
        rank = "-" if not r["answerable"] else (r["answer_rank"] or "miss")
        print(f"{r['top_score']:>6.3f}  {kind:<6} {rank:>4}  {r['note']:<14} {r['question']}  →  {r['top_section']}")

    if answerable:
        hit1 = sum(r["answer_rank"] == 1 for r in answerable)
        hit3 = sum(r["answer_rank"] is not None for r in answerable)
        print(f"\nRETRIEVAL  hit@1 {hit1}/{len(answerable)}   hit@{policy_rag.TOP_K} {hit3}/{len(answerable)}")

    print("\nTHRESHOLD SWEEP  (answer survives = answer chunk found AND scores >= t;  blocked = top score < t)")
    print(f"{'t':>6}  {'answer survives':>16}  {'unanswerable blocked':>21}")
    for t in thresholds:
        survives = sum(r["answer_score"] is not None and r["answer_score"] >= t for r in answerable)
        blocked = sum(r["top_score"] < t for r in unanswerable)
        mark = "  ← current" if t == policy_rag.MIN_SCORE else ""
        print(f"{t:>6.2f}  {survives:>10}/{len(answerable):<5}  {blocked:>15}/{len(unanswerable):<5}{mark}")


def main() -> None:
    ensure_index()
    data = runpy.run_path(str(DATASET))
    if "--held-out" in sys.argv:
        print(f"HELD-OUT set ({len(data['HELD_OUT'])} questions), at the chosen MIN_SCORE only")
        report(run(data["HELD_OUT"]), [policy_rag.MIN_SCORE])
    else:
        print(f"MAIN set ({len(data['MAIN'])} questions)")
        report(run(data["MAIN"]), THRESHOLDS)


if __name__ == "__main__":
    main()
