# Quiz: Chunking & Embeddings (2026-10-11)

14 questions on what we built in M2. For each one: my answer (shortened), the verdict, and the full answer.
Score: ✅ 3 strong · 🟡 7 partly · ❌/❓ 4 to learn again (Q5, Q8, Q9, Q11).

**Three ideas to lock in:**
1. **The model only turns text into numbers; search is maths.**
2. **Similarity measures topic, not completeness** (Q9, Q14).
3. **Access control belongs in the search, not the prompt** (Q12).

To redo: answer Q5, Q8, Q9 and Q11 again in my own words without looking.

---

## Easy: the basics

### Q1. What is a chunk, and why not give the LLM all 8 policies every time? ✅
**Mine:** A small part of a document. Sending everything costs many tokens and time, and the LLM can miss the real content.
**Full answer:** Correct: cost, time, distraction. More precisely, with too much text the LLM can **miss or mix up the relevant
part** ("lost in the middle"), and very large inputs can exceed the model's limit.

### Q2. What does an embedding model turn a chunk into? How many numbers for `gemini-embedding-001`? ✅
**Mine:** A matrix of numbers, 3,072 per chunk.
**Full answer:** 3,072 is right, but one chunk becomes **one vector** (a single list of 3,072 numbers), not a matrix.
It's a matrix (45 × 3,072) only when all chunks are stacked.

### Q3. A question scored 0.766 against Dress Code. What does it measure, and what does it *not* mean? 🟡
**Mine:** The threshold value of cosine similarity; not a probability.
**Full answer:** Two different numbers:
- **0.766 is the score:** cosine similarity between *this question's* vector and *this chunk's* vector. Different for every pair.
- **0.63 is the threshold (`MIN_SCORE`):** a fixed cut-off we compare scores against.

"Not a probability" ✅.

### Q4. Which part uses a model: chunking, embedding or search? 🟡
**Mine:** Embedding and search use a model; chunking is plain Python.
**Full answer:** Chunking = Python. Embedding = model (once per chunk when indexing, once per question with `embed_query`).
**Search = maths, no model:** cosine similarity on the vectors, done by Chroma's index.

## Intermediate: why we built it this way

### Q5. Why the `Leave Policy > Sick Leave` prefix? Use the "doctor's note" result. ❓
**Mine:** Didn't understand.
**Full answer:** **The embedding model only sees the chunk's own text**, not which document it came from.
A chunk like "Once fully approved, claims are paid with the next monthly salary…" never says *reimbursement*.
The prefix puts that context into the text, so it ends up in the vector.

Experiment, *"Do I need a doctor's note if I'm sick for 3 days?"*:
- Heading only (no title): top hit was *Attendance > Absence Without Notice* ("absent for **3 consecutive days**"),
  pulled up by "3 days" and "absence".
- Ours (with prefix): "Leave" and "Sick" in the text pulled up *Leave Policy > Sick Leave* ✅.

### Q6. Why no overlap? When is overlap right? 🟡
**Mine:** Overlap when documents aren't in a proper format, so each chunk has some context of the next.
**Full answer:** Overlap repairs **cuts in random places**: with fixed-size chunking a sentence can be split, and overlap makes it
appear whole in at least one chunk. Ours **only cuts between sentences**, so overlap would just duplicate text (more tokens,
near-identical chunks filling the top 3). Use it for **fixed-size or unstructured text** (e.g. raw PDF text with no sentence breaks).

### Q7. `embed_documents` vs `embed_query`? What if chunks and questions use different models? 🟡
**Mine:** `embed_documents` stores vectors in the DB, `embed_query` embeds the question; different models → less accurate.
**Full answer:**
1. `embed_documents` **stores nothing**; it only returns vectors. Storing is `add_chunks` → Chroma.
2. Both use the **same model**. The difference is the **task type**: `RETRIEVAL_DOCUMENT` for chunks, `RETRIEVAL_QUERY` for
   questions (Gemini tunes each, since a short question is matched against longer passages).
3. Different models → results are **meaningless**, not just less accurate. Each model has its own "map" of meaning
   (like comparing GPS coordinates with a street address). Different sizes (3,072 vs 768) can't be compared at all.

### Q8. Why does `index_policies()` embed first and reset the collection after? ❓
**Mine:** Didn't understand.
**Full answer:** Think about the embedding API failing (as it did with our 429 errors):

| Order | API call fails → | Users get |
|---|---|---|
| **Ours:** embed → delete old index → write new | Old index **untouched** | Policy answers keep working |
| **Reversed:** delete → embed | Index **already empty** | **Every** policy question says "not found" until someone re-runs it |

Rule: **never destroy the working version until the new one is ready.**

## Hard: trade-offs and failure modes

### Q9. One-sentence chunks scored highest but gave the wrong answer for "can my manager alone approve ₹20k?". Why? ❓
**Mine:** Not sure.
**Full answer:** Similarity measures *"is this about the same thing as the question?"*, **not** *"does this contain everything
needed to answer?"*.
- "Claims of up to INR 5,000 are approved by the reporting manager alone" matches "manager **alone** approve a **claim**"
  almost exactly → highest score.
- The deciding rule (₹5k–₹50k also needs the department head) is in the **next sentence**. The LLM only sees the first one → says "yes". Wrong.
- A bigger chunk with both sentences scores a bit lower but has the **complete rule**.

Core trade-off: **small chunks match precisely but miss context; big chunks have context but match vaguely.**
"Small-to-big" retrieval (search on sentences, send the whole section) gets both.

### Q10. Raise `MAX_CHUNK_CHARS` from 800 to 3,000 "for more context". What gets worse? 🟡
**Mine:** Depends on the model and policies; decide with evals.
**Full answer:** Evals ✅, but two concrete things seen in the experiment:
1. **Scores get diluted → the threshold breaks.** Whole-document scored the hotel question 0.608, next to out-of-scope
   "stock options" at 0.602.
2. **The LLM reads ~4× more text** (4,304 vs 1,107 chars): more cost, slower, more distraction, vaguer citations
   (a whole policy instead of one section).

### Q11. A scanned PDF with a big table and no headings goes through `chunk_policy`. What happens? ❌
**Mine:** We use hybrid + semantic chunking with a max limit, so a big table becomes 2–3 chunks.
**Full answer:**
1. **Our chunker isn't semantic.** It's a hybrid of **structure** (headings) and **size** (sentence packing). No model decides cuts.
2. **A scanned PDF is just images.** `pypdf` extracts **no text** → `load_policy_text` returns `""` → `chunk_policy` returns `[]`
   → the policy is **silently not searchable**. No error. Needs OCR.

For a *text* PDF with a table:
- No `##` → one section, `section=None`, prefix is the title only.
- Too long → paragraphs → sentences. Tables have no full stops.
- Falls through to the **hard cut every ~785 chars**, which can split mid-row.
- Rows lose their column headers: a chunk like `Grade 4 | 30 | Yes` means nothing on its own.

## Deep understanding: apply it to new situations

### Q12. A "Manager Compensation Guidelines" policy only managers/admins may see. An employee asks about salary bands. Where does it leak? 🟡
**Mine:** Core logic, guardrails and the system prompt are all needed.
**Full answer:** The leak is at **retrieval**: `search()` has no role filter, so the chunk can come back for any user and gets
pasted into the prompt. "Tell the LLM not to show it" is wrong because:
- The data has **already reached the LLM** (and maybe the logs).
- An LLM can be talked around ("I'm actually an admin…") and isn't deterministic.
- It breaks the hard rule: **authorization is enforced in backend code**.

**Fix:** store `visible_to` on each chunk and **filter inside the search query** by role (Chroma metadata `where`), so the
chunk never leaves the DB for an employee. Note `ensure(role, ASK_POLICY)` checks whether you may *ask*, not which documents
you may *see* ("can ask ≠ can see", same as `data_scope` in M1).

### Q13. You switch embedding models. What must be redone, and why can't `MIN_SCORE = 0.63` stay? ✅
**Mine:** Re-embed, test on the eval dataset, set `MIN_SCORE` again.
**Full answer:** Correct. To complete it:
- **Old threshold is invalid:** each model has its own score range ("related" may be 0.6 on one, 0.85 on another).
- **Rebuild the index, don't update it:** vector size may change (Chroma can't mix sizes); never mix old and new vectors.
- **Both sides switch together:** indexing and questions use the new model at once (ours does: `settings.embedding_model`).
- **Re-run all evals** (hit@3, refusal) and **re-check free-tier limits**.
- **Caches too:** this question found a real bug: `compare_chunking.py`'s cache key didn't include the model name, so it would
  have reused old vectors. Fixed.

### Q14. "How many leaves do I get?": four leave types in different chunks, top 3 only. What goes wrong, and how would an eval catch it? 🟡
**Mine:** We'd give a generic answer unless the type is specified.
**Full answer:** The danger is an **answer that looks complete but isn't**: the LLM says "8 casual, 12 sick, 18 earned" and
**silently drops parental leave** (4th chunk), or answers for only one type.

**Eval:** give the question a **list of expected facts** (all 4 types) and score **coverage** (how many appear in the retrieved
chunks / answer). 3/4 shows up as a number, not something noticed by luck.

**Fixes:** larger `k` for broad questions, an "overview" chunk listing all leave types, or asking "which type of leave do you mean?".
