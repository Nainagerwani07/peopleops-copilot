"""Split a policy document (markdown) into retrieval-friendly chunks for the Policy RAG.

Strategy (docs/learning/02_chunking_and_embeddings.md): split by '## ' heading first, so each chunk is one topic.
A section longer than MAX_CHUNK_CHARS is split again at paragraph/sentence boundaries.
If a single sentence is too long, characters are the last-resort fallback.
Every chunk's text starts with "<policy title> > <section>" so it still says what it belongs to when read alone.

No overlap is used because normal splits preserve complete sentences.
"""

from dataclasses import dataclass
import re

MAX_CHUNK_CHARS = 800


@dataclass(frozen=True)
class Chunk:
    text: str  # What gets embedded and shown to the LLM
    policy_title: str
    section: str | None  # None for text outside any '## ' section


def split_sections(markdown: str) -> list[tuple[str | None, str]]:
    """Split Markdown into sections using ## headings."""

    sections = []
    current_section = None
    current_lines = []

    for line in markdown.splitlines():

        if line.startswith("# "):
            continue

        if line.startswith("## "):
            # Save the previous section
            content = "\n".join(current_lines).strip()

            if content:
                sections.append((current_section, content))

            # Start a new section
            current_section = line[3:].strip()
            current_lines = []

        else:
            current_lines.append(line)

    # Save the final section
    content = "\n".join(current_lines).strip()

    if content:
        sections.append((current_section, content))

    return sections


def split_long_section(content: str, max_chars: int) -> list[str]:
    """Split by paragraphs, then sentences, then characters if needed."""

    paragraphs = re.split(r"\n\s*\n", content.strip())

    pieces = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(paragraph) <= max_chars:
            pieces.append((paragraph, True))
            continue

        sentences = re.split(r"(?<=[.!?])\s+", paragraph)

        first_piece = True

        for sentence in sentences:
            for start in range(0, len(sentence), max_chars):
                part = sentence[start:start + max_chars]

                pieces.append((part, first_piece))
                first_piece = False

    chunks = []
    current_chunk = ""

    for piece, starts_paragraph in pieces:
        if not current_chunk:
            separator = ""
        elif starts_paragraph:
            separator = "\n\n"
        else:
            separator = " "

        candidate = current_chunk + separator + piece

        if len(candidate) <= max_chars:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk)

            current_chunk = piece

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def chunk_policy(markdown: str, title: str) -> list[Chunk]:
    """Split a Markdown policy into retrieval-friendly chunks."""

    chunks = []

    # Step 1: Split document by headings
    sections = split_sections(markdown)

    for section, content in sections:

        # Step 2: Build context prefix
        if section:
            prefix = f"{title} > {section}"
        else:
            prefix = title

        # Step 3: Account for prefix and newline
        available_chars = MAX_CHUNK_CHARS - len(prefix) - 1

        if available_chars <= 0:
            raise ValueError("Policy title and section are too long")

        # Step 4: Split oversized content
        if len(content) <= available_chars:
            parts = [content]
        else:
            parts = split_long_section(content, available_chars)

        # Step 5: Create structured Chunk objects
        for part in parts:
            chunks.append(
                Chunk(
                    text=f"{prefix}\n{part}",
                    policy_title=title,
                    section=section,
                )
            )

    return chunks
