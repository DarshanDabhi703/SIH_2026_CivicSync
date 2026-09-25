"""
chunker.py
==========
Conservative text cleaning and semantic chunking for legal PDF content.

Rules:
  - Clean: normalize whitespace, collapse newlines, detect/remove repeated
    headers/footers. Do NOT rewrite, summarize, or alter legal text.
  - Chunk: target 600-800 tokens (≈ 2400-3200 chars at 4 chars/token),
    overlap 80-120 tokens (≈ 320-480 chars). Prefer paragraph/section
    boundaries over hard character splits.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import TypedDict

from src.pdf_parser import PageRecord


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class ChunkMetadata(TypedDict):
    domain: str


class Chunk(TypedDict):
    chunk_id: str
    source_file: str
    page_start: int
    page_end: int
    text: str
    metadata: ChunkMetadata


# ---------------------------------------------------------------------------
# Constants (tunable)
# ---------------------------------------------------------------------------

# Target chunk size in characters (≈ 4 chars per token)
CHUNK_TARGET_CHARS = 2800          # ≈ 700 tokens
CHUNK_MIN_CHARS = 200              # avoid tiny trailing chunks
CHUNK_OVERLAP_CHARS = 400          # ≈ 100 tokens overlap


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _detect_repeated_lines(pages: list[PageRecord], min_freq: float = 0.4) -> set[str]:
    """
    Identify header/footer lines that appear on >= min_freq fraction of pages.
    Only considers short lines (< 120 chars) as likely headers/footers.
    """
    total = len([p for p in pages if p["text"].strip()])
    if total < 3:
        return set()  # too few pages to reliably detect repeats

    line_counts: Counter[str] = Counter()
    for page in pages:
        seen_on_page: set[str] = set()
        for line in page["text"].splitlines():
            stripped = line.strip()
            if stripped and len(stripped) < 120 and stripped not in seen_on_page:
                line_counts[stripped] += 1
                seen_on_page.add(stripped)

    threshold = total * min_freq
    return {line for line, count in line_counts.items() if count >= threshold}


def clean_text(text: str, repeated_lines: set[str] | None = None) -> str:
    """
    Conservatively clean extracted PDF text.

    Steps applied (in order):
      1. Unicode normalization (NFKC)
      2. Remove soft hyphens and zero-width characters
      3. Repair mid-word line-break hyphenation  (word-\\nbreak → wordbreak)
      4. Collapse runs of spaces/tabs into one space
      5. Collapse runs of 3+ blank lines into 2 blank lines
      6. Strip repeated header/footer lines if provided
      7. Strip leading/trailing whitespace

    Nothing is rewritten, summarized, or invented.
    """
    if not text:
        return ""

    # 1. Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove zero-width / soft-hyphen characters
    text = re.sub(r"[\u00ad\u200b\u200c\u200d\ufeff]", "", text)

    # 3. Repair hyphenated line breaks  (legal-\nterm → legal-term)
    text = re.sub(r"-\n(\w)", r"-\1", text)

    # 4. Collapse horizontal whitespace (spaces/tabs only, preserve newlines)
    text = re.sub(r"[ \t]+", " ", text)

    # 5. Collapse 3+ consecutive blank lines → 2 blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 6. Remove known repeated header/footer lines
    if repeated_lines:
        lines = text.splitlines()
        cleaned_lines = [
            line for line in lines
            if line.strip() not in repeated_lines
        ]
        text = "\n".join(cleaned_lines)

    return text.strip()


# ---------------------------------------------------------------------------
# Paragraph / section splitting
# ---------------------------------------------------------------------------

# Patterns that suggest a good chunk boundary (section headers, numbered
# provisions, articles, rules, clauses, etc.)
_SECTION_BOUNDARY_RE = re.compile(
    r"""
    (?:^|\n)                     # start of string or line
    (?:
        \d+\.\s+[A-Z]            # "1. Capital letter" (numbered section)
      | [A-Z]{2,}[^\n]{0,80}\n  # ALL-CAPS heading line
      | (?:Section|Rule|Article|Chapter|Part|Schedule|Clause|Annex)\s+
        [\dIVXivxa-z]+           # "Section 4", "Article II", etc.
      | \n\n                     # blank line paragraph boundary
    )
    """,
    re.VERBOSE | re.MULTILINE,
)


def _split_at_boundaries(text: str) -> list[str]:
    """
    Split text into paragraphs/sections using structural boundaries.
    Returns non-empty strings.
    """
    # Primary split: blank lines (most reliable universal boundary)
    paragraphs = re.split(r"\n\n+", text)
    return [p.strip() for p in paragraphs if p.strip()]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def create_chunks(
    pages: list[PageRecord],
    domain: str,
    source_file: str,
) -> list[Chunk]:
    """
    Convert a list of page records into semantically complete chunks.

    Algorithm:
      - Clean text from all pages
      - Detect repeated headers/footers across all pages
      - Re-clean with header/footer removal applied
      - Split into paragraphs (paragraph = natural boundary)
      - Accumulate paragraphs until approaching CHUNK_TARGET_CHARS
      - When limit reached, finalize chunk and start next with overlap
      - Track which pages contributed to each chunk

    Parameters
    ----------
    pages : list[PageRecord]
        Output from pdf_parser.parse_pdf().
    domain : str
        Domain label inferred from filename (e.g. "traffic").
    source_file : str
        PDF basename (e.g. "traffic.pdf").

    Returns
    -------
    list[Chunk]
    """
    if not pages:
        return []

    # --- Step 1: detect repeated headers/footers across all pages ----------
    repeated = _detect_repeated_lines(pages)

    # --- Step 2: clean and annotate each paragraph with page info ----------
    # Each item: (paragraph_text, page_number)
    annotated_paragraphs: list[tuple[str, int]] = []
    for record in pages:
        cleaned = clean_text(record["text"], repeated_lines=repeated)
        if not cleaned:
            continue
        for para in _split_at_boundaries(cleaned):
            if para:
                annotated_paragraphs.append((para, record["page"]))

    if not annotated_paragraphs:
        return []

    # --- Step 3: accumulate paragraphs into chunks -------------------------
    chunks: list[Chunk] = []
    chunk_index = 0

    current_parts: list[str] = []
    current_pages: list[int] = []
    current_len = 0

    def _flush(parts: list[str], pages_seen: list[int]) -> None:
        nonlocal chunk_index
        text = "\n\n".join(parts).strip()
        if not text or len(text) < CHUNK_MIN_CHARS:
            return
        chunk_index += 1
        domain_slug = re.sub(r"\W+", "_", domain.lower()).strip("_")
        source_slug = Path(source_file).stem.lower()
        source_slug = re.sub(r"\W+", "_", source_slug).strip("_")
        chunk_id = f"{domain_slug}_{chunk_index:04d}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                source_file=source_file,
                page_start=min(pages_seen),
                page_end=max(pages_seen),
                text=text,
                metadata=ChunkMetadata(domain=domain),
            )
        )

    for para_text, page_num in annotated_paragraphs:
        para_len = len(para_text)

        # If adding this paragraph would exceed target and we have content,
        # flush the current chunk first
        if current_len + para_len > CHUNK_TARGET_CHARS and current_parts:
            _flush(current_parts, current_pages)

            # Build overlap: keep trailing paragraphs up to CHUNK_OVERLAP_CHARS
            overlap_parts: list[str] = []
            overlap_pages: list[int] = []
            overlap_len = 0
            for op, opg in zip(
                reversed(current_parts), reversed(current_pages)
            ):
                if overlap_len + len(op) > CHUNK_OVERLAP_CHARS:
                    break
                overlap_parts.insert(0, op)
                overlap_pages.insert(0, opg)
                overlap_len += len(op)

            current_parts = overlap_parts
            current_pages = overlap_pages
            current_len = overlap_len

        # If a single paragraph is larger than the target, split it by
        # sentences to avoid a giant chunk
        if para_len > CHUNK_TARGET_CHARS:
            sentences = re.split(r"(?<=[.!?])\s+", para_text)
            for sent in sentences:
                if current_len + len(sent) > CHUNK_TARGET_CHARS and current_parts:
                    _flush(current_parts, current_pages)
                    current_parts = []
                    current_pages = []
                    current_len = 0
                current_parts.append(sent)
                current_pages.append(page_num)
                current_len += len(sent)
        else:
            current_parts.append(para_text)
            current_pages.append(page_num)
            current_len += para_len

    # Flush any remaining content
    if current_parts:
        _flush(current_parts, current_pages)

    return chunks
