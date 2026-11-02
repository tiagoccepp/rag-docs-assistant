"""Tests for structure-aware chunking (src/ingest.py). Fully offline."""

from src.ingest import (
    chunk_document,
    count_tokens,
    ingest_directory,
    parse_markdown_structure,
    split_tokens_with_overlap,
)

SAMPLE_MD = """# Title

Intro paragraph here.

## Section One

Content of section one with several words to chunk across windows.

## Section Two

Content of section two. It has multiple sentences. Each sentence adds tokens.
"""


def test_parse_markdown_structure_finds_sections():
    sections = parse_markdown_structure(SAMPLE_MD)
    titles = [s.title for s in sections]
    assert "Section One" in titles
    assert "Section Two" in titles


def test_parse_markdown_structure_headingless_doc():
    sections = parse_markdown_structure("Just plain text, no headings.")
    assert len(sections) == 1
    assert sections[0].content == "Just plain text, no headings."


def test_split_tokens_with_overlap_respects_budget():
    text = " ".join(f"word{i}" for i in range(500))
    windows = split_tokens_with_overlap(text, chunk_size=100, overlap=10)
    assert len(windows) > 1
    # Overlap: last words of first window appear in second window
    assert windows[0].split()[-1] in windows[1].split()


def test_split_empty_text_returns_empty():
    assert split_tokens_with_overlap("") == []


def test_chunk_document_attaches_section_metadata():
    chunks = chunk_document(SAMPLE_MD, source="test.md", chunk_size=100, overlap=10)
    assert len(chunks) >= 2
    assert all(c.source == "test.md" for c in chunks)
    sections = {c.section for c in chunks}
    assert "Section One" in sections
    assert "Section Two" in sections


def test_chunk_document_positions_are_sequential():
    chunks = chunk_document(SAMPLE_MD, source="t.md", chunk_size=60, overlap=5)
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_count_tokens_fallback_positive():
    assert count_tokens("hello world foo bar") > 0
    assert count_tokens("") == 0


def test_ingest_directory_loads_sample_docs(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\nSome content here.", encoding="utf-8")
    (tmp_path / "b.md").write_text("# B\n\nOther content here.", encoding="utf-8")
    (tmp_path / "skip.txt").write_text("not markdown", encoding="utf-8")
    chunks = ingest_directory(tmp_path)
    assert len(chunks) == 2
    assert {c.section for c in chunks} == {"A", "B"}
