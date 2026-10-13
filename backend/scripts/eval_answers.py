"""Answer eval: run the full policy_rag.answer_question() and check what the user would finally get.

The retrieval eval (eval_retrieval.py) only showed what reaches the LLM. This one checks layer 2:
  unanswerable question: refused (at which layer?) or LEAKED (an answer was given)?
  answerable question:   answered and cited the section that holds the answer, or refused / wrong source?

Run from backend/:  PYTHONPATH=. .venv/bin/python scripts/eval_answers.py [--held-out]
Costs 1 embedding call per question + 1 LLM call per question that passes layer 1 (rate-limited to 15/min).
"""

import asyncio
import runpy
import sys
import warnings
from collections import Counter
from pathlib import Path

from app.services.ai import policy_rag
from app.services.ai.chunking import chunk_policy
from scripts.trace_policy_rag import ensure_index

DATASET = Path(__file__).resolve().parents[1] / "data" / "eval" / "policy_retrieval.py"
POLICY_DIR = Path(__file__).resolve().parents[1] / "data" / "policies"
warnings.filterwarnings("ignore", message=".*sampling parameter")  # flash-lite ignores temperature; once is enough


def fact_sections() -> list[tuple[str, str | None, str]]:
    """(policy title, section, chunk text) for every chunk, to find which section holds an expected fact."""
    rows = []
    for path in sorted(POLICY_DIR.glob("*.md")):
        md = path.read_text(encoding="utf-8")
        rows += [(c.policy_title, c.section, c.text) for c in chunk_policy(md, md.splitlines()[0].removeprefix("# "))]
    return rows


# Spy on the LLM step: answer_question() still runs unchanged, we only record what the LLM returned.
llm_outputs: dict[str, policy_rag.LLMAnswer] = {}
_real_generate = policy_rag.generate_answer


async def _recording_generate(question, hits):
    llm_outputs[question] = await _real_generate(question, hits)
    return llm_outputs[question]


policy_rag.generate_answer = _recording_generate


def outcome(question: str, fact: str | None, result: policy_rag.PolicyAnswer, sections) -> str:
    if not result.found:
        if question not in llm_outputs:
            layer = "layer 1 (score)"
        elif not llm_outputs[question].found:
            layer = "layer 2 (LLM found=false)"
        else:
            layer = "layer 2 (no valid citation)"
        return f"{'ok' if fact is None else 'FALSE REFUSAL'}: refused at {layer}"
    if fact is None:
        return "LEAKED: answered an uncovered question"
    expected = {(title, section) for title, section, text in sections if fact in text}
    cited = {(s.title, s.section) for s in result.sources}
    return "ok: answered, cites the right section" if expected & cited else "WRONG SOURCE: answered, cites other sections"


async def main() -> None:
    ensure_index()
    data = runpy.run_path(str(DATASET))
    dataset = data["HELD_OUT" if "--held-out" in sys.argv else "MAIN"]
    sections = fact_sections()
    print(f"{len(dataset)} questions, MIN_SCORE = {policy_rag.MIN_SCORE}, model calls are rate-limited: be patient\n")

    counts = Counter()
    for question, fact, note in dataset:
        result = await policy_rag.answer_question(question)
        verdict = outcome(question, fact, result, sections)
        counts[verdict.split(":")[0]] += 1
        print(f"{'ANS ' if fact else 'NONE'}  {note:<14} {question}\n      → {verdict}", flush=True)
        if not verdict.startswith("ok") or (fact and result.found):
            print(f"      answer: {result.answer}")
            print(f"      sources: {[f'{s.title} > {s.section}' for s in result.sources]}")

    print(f"\nSUMMARY  {dict(counts)}")


if __name__ == "__main__":
    asyncio.run(main())
