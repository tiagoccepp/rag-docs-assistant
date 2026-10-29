"""Structure-aware document ingestion and chunking.

Strategy:
  1. Split Markdown on headings (## / ###) to preserve section boundaries.
  2. Within each section, split into token windows of `chunk_size` with overlap.
  3. Attach metadata: source path, section title, position, token count.

Token counting uses tiktoken (cl100k_base) when available, falling back
to a whitespace heuristic so tests run fully offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def count_tokens(text: str) -> int:
    """Count tokens with tiktoken if installed, else whitespace heuristic."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Offline fallback: ~1.3 tokens per word approximation
        words = text.split()
        return max(1, int(len(words) * 1.3)) if words else 0


_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)


@dataclass
class Section:
    title: str
    level: int
    content: str


@dataclass
class Chunk:
    text: str
    source: str
    section: str
    position: int
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_markdown_structure(markdown: str) -> list[Section]:
    """Split Markdown into sections on headings.

    Text before the first heading becomes an "Introduction" section.
    Returns at least one section even for heading-less documents.
    """
    matches = list(_HEADING_RE.finditer(markdown))
    if not matches:
        return [Section(title="Document", level=0, content=markdown.strip())]

    sections: list[Section] = []
    # Preamble before first heading
    if matches[0].start() > 0:
        preamble = markdown[: matches[0].start()].strip()
        if preamble:
            sections.append(Section(title="Introduction", level=0, content=preamble))

    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        content = markdown[start:end].strip()
        sections.append(Section(title=title, level=level, content=content))
    return sections


def split_tokens_with_overlap(
    text: str, chunk_size: int = 512, overlap: int = 50
) -> list[str]:
    """Split text into token windows with overlap.

    Uses word-level windows scaled to approximate the token budget.
    Guarantees progress even for pathological inputs.
    """
    words = text.split()
    if not words:
        return []
    # Approximate words per chunk from token budget
    words_per_chunk = max(10, int(chunk_size / 1.3))
    overlap_words = max(0, min(int(overlap / 1.3), words_per_chunk - 1))
    step = max(1, words_per_chunk - overlap_words)

    windows: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + words_per_chunk]
        if not window:
            break
        windows.append(" ".join(window))
        if start + words_per_chunk >= len(words):
            break
    return windows


def chunk_document(
    markdown: str,
    source: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[Chunk]:
    """Chunk a Markdown document with structure awareness.

    Each chunk keeps its section title so retrieval can cite precisely.
    Section headers are prepended to chunk text for retrieval context.
    """
    sections = parse_markdown_structure(markdown)
    chunks: list[Chunk] = []
    position = 0
    for section in sections:
        body = section.content.strip()
        if not body and section.title:
            continue
        windows = split_tokens_with_overlap(body, chunk_size, overlap) or [body]
        for window in windows:
            text = f"# {section.title}\n\n{window}" if section.title else window
            chunks.append(
                Chunk(
                    text=text,
                    source=source,
                    section=section.title,
                    position=position,
                    token_count=count_tokens(text),
                    metadata={"level": section.level},
                )
            )
            position += 1
    return chunks


def ingest_file(path: str | Path, chunk_size: int = 512, overlap: int = 50) -> list[Chunk]:
    """Read a Markdown file from disk and chunk it."""
    p = Path(path)
    markdown = p.read_text(encoding="utf-8")
    return chunk_document(markdown, source=str(p), chunk_size=chunk_size, overlap=overlap)


def ingest_directory(
    directory: str | Path, chunk_size: int = 512, overlap: int = 50
) -> list[Chunk]:
    """Ingest all `.md` files in a directory (non-recursive)."""
    d = Path(directory)
    all_chunks: list[Chunk] = []
    for md_file in sorted(d.glob("*.md")):
        all_chunks.extend(ingest_file(md_file, chunk_size, overlap))
    return all_chunks
