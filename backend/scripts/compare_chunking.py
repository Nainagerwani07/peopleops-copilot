"""Learning tool: compare chunking strategies on our 8 policies, side by side.

Shows how one document is cut by each strategy, then asks the same questions against each one
and scores whether the retrieved chunks contain the answer.

Run from backend/:  PYTHONPATH=. .venv/bin/python scripts/compare_chunking.py [policy_file.md]
Embeddings are cached in storage/vector_store/ (gitignored), so re-runs cost no API quota.
"""

import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

from app.core.config import settings
from app.services.ai.chunking import chunk_policy
from app.services.ai.embeddings import get_embeddings

POLICY_DIR = Path(__file__).resolve().parents[1] / "data" / "policies"
CACHE = Path("storage/vector_store/compare_chunking_cache.json")


# ---------- 5 strategies: each takes (markdown, title) and returns a list of chunk texts ----------
def whole_document(md: str, title: str) -> list[str]:
    """No chunking: the whole policy is one chunk."""
    return [md]


def fixed_300(md: str, title: str) -> list[str]:
    """Cut every 300 characters, repeating the last 50 (overlap). Ignores sentences and headings."""
    return [md[i : i + 300] for i in range(0, len(md), 250)]


def one_sentence(md: str, title: str) -> list[str]:
    """Every sentence is its own chunk."""
    lines = [line for line in md.splitlines() if line.strip() and not line.startswith("#")]
    return [s for line in lines for s in re.split(r"(?<=[.!?])\s+", line.strip()) if s]


def heading_only(md: str, title: str) -> list[str]:
    """Split at '## ' headings. No size limit, no 'Policy > Section' prefix."""
    return [s.strip() for s in re.split(r"\n(?=## )", md) if s.strip() and not s.strip().startswith("# ")] or [md]


def ours(md: str, title: str) -> list[str]:
    """Our chunk_policy(): headings, then sentence packing up to 800 chars, with the prefix."""
    return [c.text for c in chunk_policy(md, title)]


STRATEGIES = {
    "1 whole document": whole_document,
    "2 fixed 300 chars": fixed_300,
    "3 one sentence": one_sentence,
    "4 heading only": heading_only,
    "5 ours (chunk_policy)": ours,
}

# Each answer fact appears exactly once in the policies: a chunk "has the answer" if it contains it whole.
QUESTIONS = [
    ("Can my manager alone approve a 20,000 rupee claim?", "Claims above INR 5,000 and up to INR 50,000 need approval from both"),
    ("Can I carry over unused vacation to next year?", "carried forward to the next year, up to a maximum of 30 days"),
    ("Do I need a doctor's note if I'm sick for 3 days?", "a medical certificate must be uploaded"),
    ("How fast must I report a lost laptop?", "must be reported to IT within 1 hour"),
    ("What is the dress code on Fridays?", "Fridays are casual"),
    ("How many days a week can I work from home?", "up to 2 days a week"),
    ("When do I get a late mark?", "A clock-in after 9:30 AM is recorded as a late mark"),
    ("What is the notice period once I'm confirmed?", "60 days after confirmation"),
    ("How long is a comp off valid?", "valid for 60 days from the date"),
    ("What's the hotel limit in Mumbai?", "INR 6,000 per night in metro cities"),
    ("When will my approved claim be paid?", "paid with the next monthly salary"),
    ("Can I submit a rejected claim again?", "Rejected claims can be resubmitted once"),
    ("Can I accept an expensive gift from a vendor?", "worth more than INR 3,000"),
    ("What's the deadline for a travel claim?", "Travel claims must be submitted within 15 days of return"),
    ("Who handles sexual harassment complaints?", "Internal Committee"),
]


# ---------- embeddings with a disk cache ----------
def embed(texts: list[str], kind: str) -> list[list[float]]:
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    key = lambda t: hashlib.sha256(f"{settings.embedding_model}:{kind}:{t}".encode()).hexdigest()  # noqa: E731  (model in key: switching models must not reuse old vectors)
    missing = [t for t in dict.fromkeys(texts) if key(t) not in cache]
    # Free tier: 100 embedded texts per minute (each text counts, even inside one batch call).
    # On a 429, wait a minute and retry; batches are saved, so a stopped run resumes where it left off.
    for start in range(0, len(missing), 50):
        batch = missing[start : start + 50]
        while True:
            try:
                emb = get_embeddings()
                vectors = emb.embed_documents(batch) if kind == "doc" else [emb.embed_query(t) for t in batch]
                break
            except Exception as exc:
                if "RESOURCE_EXHAUSTED" not in str(exc):
                    raise
                print(f"  ...free-tier limit hit at text {start} of {len(missing)}: waiting 65s")
                time.sleep(65)
        cache.update({key(t): v for t, v in zip(batch, vectors)})
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache))
    return [cache[key(t)] for t in texts]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def short(text: str, width: int = 90) -> str:
    one_line = text.replace("\n", " ⏎ ")
    return one_line if len(one_line) <= width else one_line[: width - 3] + "..."


def main() -> None:
    docs = {p.name: p.read_text(encoding="utf-8") for p in sorted(POLICY_DIR.glob("*.md"))}
    title_of = {name: md.splitlines()[0].removeprefix("# ") for name, md in docs.items()}
    show = sys.argv[1] if len(sys.argv) > 1 else "reimbursement_policy.md"

    # ---- Part 1: how ONE document is cut ----
    print(f"\n{'=' * 100}\nPART 1: how '{show}' ({len(docs[show])} chars) is cut by each strategy\n{'=' * 100}")
    for name, strategy in STRATEGIES.items():
        chunks = strategy(docs[show], title_of[show])
        print(f"\n--- {name}: {len(chunks)} chunks ---")
        for i, c in enumerate(chunks):
            print(f"  [{i:>2}] {len(c):>4} chars | {short(c)}")

    # ---- Part 2: same questions against each strategy ----
    query_vectors = embed([q for q, _ in QUESTIONS], "query")
    rows, details = [], {}
    for name, strategy in STRATEGIES.items():
        chunks = [c for doc, md in docs.items() for c in strategy(md, title_of[doc])]
        vectors = embed(chunks, "doc")
        hit1 = hit3 = 0
        context_chars = 0
        details[name] = []
        for (question, fact), qv in zip(QUESTIONS, query_vectors):
            ranked = sorted(zip((cosine(qv, v) for v in vectors), chunks), reverse=True)[:3]
            has = [fact in c for _, c in ranked]
            hit1 += has[0]
            hit3 += any(has)
            context_chars += sum(len(c) for _, c in ranked)
            details[name].append((question, ranked[0][0], has, ranked[0][1]))
        n = len(QUESTIONS)
        rows.append((name, len(chunks), hit1, hit3, context_chars // n))

    print(f"\n{'=' * 100}\nPART 2: {len(QUESTIONS)} questions, all 8 policies\n{'=' * 100}")
    print("  hit@1 = the top chunk contains the answer   hit@3 = one of the top 3 does")
    print("  LLM context = characters the LLM must read per question (top 3 chunks): more = slower, costlier, noisier\n")
    print(f"  {'strategy':<24}{'chunks':>8}{'hit@1':>10}{'hit@3':>10}{'LLM context':>14}")
    for name, count, h1, h3, ctx in rows:
        print(f"  {name:<24}{count:>8}{h1:>7}/{len(QUESTIONS)}{h3:>7}/{len(QUESTIONS)}{ctx:>11} ch")

    print(f"\n{'=' * 100}\nPART 3: per question (✅ = that rank contains the answer; top chunk shown)\n{'=' * 100}")
    for i, (question, _) in enumerate(QUESTIONS):
        print(f"\nQ: {question}")
        for name in STRATEGIES:
            q, score, has, top = details[name][i]
            marks = "".join("✅" if h else "·" for h in has)
            print(f"   {name:<24} {marks}  {score:.3f}  {short(top, 60)}")


if __name__ == "__main__":
    main()
