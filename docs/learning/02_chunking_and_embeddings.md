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
