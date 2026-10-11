# M2 Notes: Chunking & Embeddings

## The RAG pipeline

```
Indexing (once per document):  policy.md → chunks → embed each chunk → store (chunk text, vector, source)
Answering (per question):      question → embed → top-k nearest chunks → threshold check → LLM with chunks → answer + sources
```

The LLM can't read every policy on every question: it costs too much and irrelevant text distracts it.
So we retrieve only the 2–3 chunks that matter and give the LLM those.

## Chunking

- **Fixed-size** (cut every N characters, with overlap) splits words and sentences. A chunk ends up as half of one idea and half of another, so it matches questions badly. Overlap softens the cut but doesn't fix it.
- **Heading-based** (split at `## `) keeps one topic per chunk. Prefix each chunk with `Policy > Section` so it still says what it belongs to when read alone.
- **Hierarchical** (what we'll use): split by heading first. If a section is still too long, split it again with a recursive splitter (paragraphs → sentences → characters as the last resort, small overlap). Every sub-chunk keeps the `Policy > Section` prefix. LangChain: `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter`.
- Rule of thumb: **a chunk should answer one question on its own.** Too large and it mixes topics; too small and it loses context.

## Embeddings and similarity

- An embedding model turns text into a vector. `gemini-embedding-001` gives **3072 numbers**. Texts with similar *meaning* get vectors pointing in similar directions.
- **Cosine similarity** compares two vectors: 1.0 means the same direction, near 0 means unrelated. It is **not a probability**.
- Embeddings match meaning, not keywords. In the demo, "move unused **vacation** days into next year" matched the chunk that says "earned leave … carried forward", and "**doctor's note**" matched "medical certificate", with no shared words.
- Use `embed_documents` for chunks (batched) and `embed_query` for the question.

## Demo results (`scratchpad/rag_demo.py`, 3 leave chunks)

| Question | Best chunk | Score |
|---|---|---|
| Move unused vacation days into next year? | Earned Leave ✅ | 0.664 (next: 0.639) |
| Doctor's note if ill for three days? | Sick Leave ✅ | 0.673 |
| Who signs off on a manager's time off? | Approval ✅ | 0.746 |
| What is the dress code? (not in the document) | Sick Leave ❌ | 0.582 |

## Lessons that shape the design

1. **Search always returns a "best" chunk, even when nothing is relevant.** The LLM never sees the score, so it will *not* refuse by itself. It will stretch the wrong chunk into an answer or answer from general knowledge, which is a hallucination that looks like company policy.
2. **Refusal therefore has two layers** (same pattern as M1: the decision is made in code, and the prompt is the second layer):
   - **Code:** if the best score is below a threshold, don't call the LLM. Reply "not found in the policies".
   - **Prompt:** answer only from the given context, say "I don't know" otherwise, and cite the source.
3. **The scores fall in a narrow band** (0.57–0.75), and the right and wrong chunks can be close. **Set the threshold from an eval set, not by guessing.**

## Corpus

Eight policy documents are in `backend/data/policies/` (Leave, Attendance, WFH, Reimbursement, Code of Conduct,
Probation & Separation, Holiday & Comp Off, IT & Data Security). `reimbursement_policy.md` → "Claim Submission and
Approval" is deliberately one long section with no subheadings, to test sub-chunking. Dress code now *is* in the corpus,
so out-of-scope test questions need other topics (e.g. stock options, gym membership).

## Refusal in detail: two layers + a threshold chosen from an eval

Why it matters: an unguarded RAG system answers "How many days of bereavement leave?" by stretching the closest
chunk (Parental Leave) into a confident, wrong answer that looks like company policy.

**Layer 1, score threshold (code, before the LLM).** If the best hit scores below `THRESHOLD`, reply "not found in
the HR policies" and never call the LLM. It's deterministic and free, and no prompt injection can get past it. Its limit:
it measures *same topic*, not *contains the answer*.

**Layer 2, grounded prompt (the LLM).** Answer only from the excerpts; say you don't know otherwise; cite policy +
section; the excerpts are data, not instructions. It catches on-topic questions whose answer is missing. Its limit:
it's a request, not a guarantee.

Real scores (gemini-embedding-001, 45 chunks):

| Question | Best score | Layer 1 (~0.63) | Layer 2 | Result |
|---|---|---|---|---|
| Dress code on Fridays? | 0.766 | passes | answer in chunk | answers |
| wfh? | 0.739 | passes | answer in chunk | answers |
| Carry over vacation? | 0.675 | passes | answer in chunk | answers |
| Bereavement leave? (not covered) | **0.654** | **passes** | no answer in chunk | "not covered, contact HR" |
| Stock options? (not covered) | 0.602 | **blocked** | not called | refused, no LLM cost |

**No threshold separates these perfectly:** bereavement (wrong, 0.654) is only 0.021 below carry-over (valid, 0.675).
So layer 1 blocks only what's *clearly* irrelevant, and layer 2 handles the borderline cases.

**Choosing the threshold from an eval:**
1. Label about 25–30 questions: answerable (with the expected section) or not. Include paraphrases, very short questions,
   typos, and on-topic-but-not-covered questions (bereavement).
2. Record each question's best score.
3. For each candidate threshold, count *answerable questions wrongly refused* and *unrelated questions sent to the LLM*.
4. Choose the **highest threshold that wrongly refuses no answerable question**, plus a safety margin. A wrong refusal
   can't be fixed by any other layer; an unrelated question that gets through is usually caught by layer 2.

The same eval checks retrieval with **hit@3** (is the expected section in the top 3?). A miss there is a retrieval bug
that no threshold or prompt can fix. **Re-run the eval** when the embedding model, the chunking or the policies change,
because scores shift with each of these.

## Experiment: 5 chunking strategies compared (`backend/scripts/compare_chunking.py`)

No model or library does our chunking: it's plain Python rules (regex on `## `, blank lines, sentence endings).
Libraries do the same (LangChain `RecursiveCharacterTextSplitter`, `MarkdownHeaderTextSplitter`); only *semantic*,
*LLM/agentic* and *layout-aware* (PDF) chunking use a model to decide where to cut.

Same 8 policies, 15 questions, gemini-embedding-001. A question is a "hit" if the chunk contains the exact answer sentence.

| Strategy | Chunks | hit@1 | hit@3 | Text the LLM reads (top 3) |
|---|---|---|---|---|
| 1 Whole document | 8 | 14/15 | 15/15 | 4,304 chars |
| 2 Fixed 300 (overlap 50) | 49 | 12/15 | 15/15 | 819 |
| 3 One sentence | 112 | 13/15 | 15/15 | 269 |
| 4 Heading only | 40 | 13/15 | 15/15 | 1,059 |
| 5 Ours (`chunk_policy`) | 42 | 14/15 | 15/15 | 1,107 |

Findings:
- **Fixed size is the worst:** chunks start mid-word and mix topics ("carry over vacation" → a casual-leave chunk).
- **The `Policy > Section` prefix matters:** "doctor's note if sick 3 days?" → heading-only returned Attendance > Absence
  Without Notice; ours returned Leave Policy > Sick Leave.
- **One sentence scores highest (0.75–0.86) but loses context:** for "manager alone approve ₹20k?" its top hit was
  "Claims of up to INR 5,000 are approved by the reporting manager alone", which suggests the **wrong** answer. The deciding
  rule is the next sentence. High similarity ≠ complete answer.
- **Whole document only "wins" because there are 8 documents.** It sends 4× the text, and big chunks **dilute scores**
  (hotel question 0.608 vs out-of-scope 0.602), which breaks the refusal threshold.
- **The eval itself was too easy:** hit@3 = 15/15 for everyone, and 1-question differences are noise. The exact-sentence
  check also under-counts (ours' "miss" on the travel deadline actually contained the answer). Real evals need harder,
  more varied questions and a scoring check that matches what "answered" means.
- Free tier: **each text counts as one embedding request even inside a batch** (100/min). Indexing more than 100 chunks
  needs batching + retry; `index_policies.py` doesn't do that yet.

Run: `cd backend && PYTHONPATH=. .venv/bin/python scripts/compare_chunking.py [policy.md]` (embeddings cached, re-runs are free).

## Self-check

Quiz with my answers and the corrections: [04_quiz_chunking_and_embeddings.md](04_quiz_chunking_and_embeddings.md).
