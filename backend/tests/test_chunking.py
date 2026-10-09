"""Policy chunker spec (pure Python, no LLM, no DB)."""

import re
from pathlib import Path

import pytest

from app.services.ai.chunking import MAX_CHUNK_CHARS, chunk_policy

POLICY_DIR = Path(__file__).resolve().parents[1] / "data" / "policies"
POLICY_FILES = sorted(POLICY_DIR.glob("*.md"))


def read(name: str) -> str:
    return (POLICY_DIR / name).read_text(encoding="utf-8")


def sentences(markdown: str) -> list[str]:
    """Every sentence of the body text (heading lines excluded)."""
    lines = [line for line in markdown.splitlines() if line.strip() and not line.startswith("#")]
    return [s for line in lines for s in re.split(r"(?<=\.)\s+", line.strip()) if s]


def test_one_chunk_per_section_in_order():
    chunks = chunk_policy(read("leave_policy.md"), title="Leave Policy")
    sections = [c.section for c in chunks if c.section is not None]
    assert sections == [
        "Leave Year",
        "Earned Leave",
        "Casual Leave",
        "Sick Leave",
        "Parental Leave",
        "Approval",
        "Leave Encashment",
    ]


def test_every_chunk_says_where_it_comes_from():
    for chunk in chunk_policy(read("leave_policy.md"), title="Leave Policy"):
        assert chunk.policy_title == "Leave Policy"
        prefix = f"Leave Policy > {chunk.section}" if chunk.section else "Leave Policy"
        assert chunk.text.startswith(prefix)


def test_intro_before_first_section_is_kept():
    chunks = chunk_policy(read("leave_policy.md"), title="Leave Policy")
    assert any("Interns and contractors" in c.text for c in chunks)


def test_long_section_is_split_without_cutting_sentences():
    chunks = chunk_policy(read("reimbursement_policy.md"), title="Reimbursement Policy")
    claims = [c for c in chunks if c.section == "Claim Submission and Approval"]
    assert len(claims) >= 2
    # The approval-limit rule must stay one whole sentence, or retrieval returns half a rule.
    rule = "Claims above INR 5,000 and up to INR 50,000 need approval from both the reporting manager and the department head."
    assert any(rule in c.text for c in claims)


def test_one_line_policy_without_sections():
    md = "# ID Card Policy\n\nEmployees must carry ID cards inside office premises.\n"
    chunks = chunk_policy(md, title="ID Card Policy")
    assert len(chunks) == 1
    assert chunks[0].section is None
    assert chunks[0].text.startswith("ID Card Policy")
    assert "Employees must carry ID cards inside office premises." in chunks[0].text


@pytest.mark.parametrize("md", ["", "   \n", "# Empty Policy\n"])
def test_document_without_body_gives_no_chunks(md):
    assert chunk_policy(md, title="Empty Policy") == []


@pytest.mark.parametrize("path", POLICY_FILES, ids=lambda p: p.name)
def test_real_policies_lose_no_sentence_and_respect_max_size(path):
    md = path.read_text(encoding="utf-8")
    chunks = chunk_policy(md, title=md.splitlines()[0].removeprefix("# "))
    for sentence in sentences(md):
        assert any(sentence in c.text for c in chunks), f"lost or cut: {sentence!r}"
    for chunk in chunks:
        assert len(chunk.text) <= MAX_CHUNK_CHARS, f"{chunk.section}: {len(chunk.text)} chars"


def test_sentence_longer_than_limit_is_hard_cut_not_an_error():
    # Text extracted from PDFs (tables, lists) can have no full stops; indexing must not crash on it.
    md = "## Table\n" + "x" * 2000
    chunks = chunk_policy(md, title="T")
    assert len(chunks) >= 3
    assert all(len(c.text) <= MAX_CHUNK_CHARS for c in chunks)
    assert "".join(c.text.removeprefix("T > Table\n") for c in chunks) == "x" * 2000
