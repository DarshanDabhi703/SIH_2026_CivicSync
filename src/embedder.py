"""
embedder.py
===========
CivicSync Phase 2A - Embedding generation

Loads intfloat/multilingual-e5-base and encodes document chunks
using the  "passage: <text>"  prefix required by E5 models.

Public API
----------
    load_model(model_name)          -> SentenceTransformer
    embed_chunks(chunks, model, batch_size) -> list[dict]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL  = "intfloat/multilingual-e5-base"
EMBEDDING_DIM  = 768
DEFAULT_BATCH  = 32          # safe for CPU; GPU users can raise to 64+


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """Load and return the SentenceTransformer model."""
    model = SentenceTransformer(model_name)
    return model


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_chunks(
    chunks: List[dict],
    model: SentenceTransformer,
    batch_size: int = DEFAULT_BATCH,
    model_name: str = DEFAULT_MODEL,
) -> List[dict]:
    """
    Embed a list of chunk dicts.

    Each chunk must contain at least:
        chunk_id, source_file, page_start, page_end, text, metadata.domain

    Returns a new list of dicts with all original fields plus:
        model_name, dimensions, embedding
    """
    if not chunks:
        return []

    # Build texts with E5 passage prefix
    texts = [f"passage: {chunk['text']}" for chunk in chunks]

    # Batch encode — returns numpy array shape (N, 768)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Validate shape
    if vectors.shape != (len(chunks), EMBEDDING_DIM):
        raise ValueError(
            f"Unexpected embedding shape {vectors.shape}; "
            f"expected ({len(chunks)}, {EMBEDDING_DIM})"
        )

    # Build output records
    records: List[dict] = []
    for chunk, vector in zip(chunks, vectors):
        embedding_list = vector.tolist()

        # Per-record validation
        if len(embedding_list) != EMBEDDING_DIM:
            raise ValueError(
                f"chunk_id={chunk['chunk_id']} produced embedding of "
                f"length {len(embedding_list)}, expected {EMBEDDING_DIM}"
            )
        if any(v is None for v in embedding_list):
            raise ValueError(f"chunk_id={chunk['chunk_id']} contains null values")

        domain = chunk.get("metadata", {}).get("domain", "unknown")

        record = {
            "chunk_id":    chunk["chunk_id"],
            "source_file": chunk["source_file"],
            "page_start":  chunk["page_start"],
            "page_end":    chunk["page_end"],
            "domain":      domain,
            "text":        chunk["text"],
            "model_name":  model_name,
            "dimensions":  EMBEDDING_DIM,
            "embedding":   embedding_list,
        }
        records.append(record)

    return records


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_chunks_jsonl(path: Path) -> List[dict]:
    """Read all chunks from a JSONL file."""
    chunks = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def write_embeddings_jsonl(records: List[dict], path: Path) -> None:
    """Write embedding records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_embeddings(records: List[dict], expected_count: int) -> None:
    """
    Run all validation checks on a list of embedding records.
    Raises ValueError on any failure.
    """
    # 1. Count
    if len(records) != expected_count:
        raise ValueError(
            f"Embedding count mismatch: got {len(records)}, "
            f"expected {expected_count}"
        )

    for rec in records:
        emb = rec.get("embedding")

        # 2. Not null
        if emb is None:
            raise ValueError(f"chunk_id={rec['chunk_id']} has null embedding")

        # 3. Correct dimension
        if len(emb) != EMBEDDING_DIM:
            raise ValueError(
                f"chunk_id={rec['chunk_id']} has dimension {len(emb)}, "
                f"expected {EMBEDDING_DIM}"
            )

        # 4. All numeric
        if any(not isinstance(v, (int, float)) for v in emb):
            raise ValueError(
                f"chunk_id={rec['chunk_id']} contains non-numeric values"
            )

        # 5. chunk_id preserved
        if not rec.get("chunk_id"):
            raise ValueError("Record missing chunk_id")

        # 6. Original text preserved
        if not rec.get("text"):
            raise ValueError(f"chunk_id={rec['chunk_id']} missing text field")
