"""
context_builder.py
==================
CivicSync Phase 3 — Context Builder

Formats retrieved chunks from Supabase into a structured context string
with full metadata for consumption by the LLM prompt.
"""

from __future__ import annotations

from typing import Any, Dict, List


def build_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Build a structured context string from a list of retrieved chunk dicts.

    Parameters
    ----------
    chunks : list[dict]
        Retrieved chunk dictionaries with keys:
        chunk_id, content, source_file, page_start, page_end, domain, similarity.

    Returns
    -------
    str
        Formatted context string containing all source metadata and legal content.
    """
    if not chunks:
        return "No relevant legal evidence retrieved."

    formatted_sources = []
    for rank, chunk in enumerate(chunks, 1):
        chunk_id = chunk.get("chunk_id", "N/A")
        domain = chunk.get("domain", "N/A")
        source_file = chunk.get("source_file", "N/A")
        page_start = chunk.get("page_start", "?")
        page_end = chunk.get("page_end", "?")
        similarity = chunk.get("similarity", 0.0)
        content = (chunk.get("content") or "").strip()

        page_str = (
            str(page_start)
            if page_start == page_end
            else f"{page_start}–{page_end}"
        )

        source_block = (
            f"[SOURCE {rank}]\n"
            f"Domain: {domain}\n"
            f"Source File: {source_file}\n"
            f"Page: {page_str}\n"
            f"Similarity: {similarity:.4f}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Content:\n{content}"
        )
        formatted_sources.append(source_block)

    return "\n\n".join(formatted_sources)
