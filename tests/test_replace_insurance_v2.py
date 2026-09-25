"""
test_replace_insurance_v2.py
=============================
Unit tests for CivicSync Phase 5B-3 — Insurance Replacement Pipeline.
Tests local validation logic without modifying Supabase database.
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest

from src.replace_insurance_v2 import (
    validate_local_inputs,
    read_jsonl,
    CHUNKS_FILE,
    EMBEDDINGS_FILE,
)


def _sample_valid_chunks(count: int = 5) -> list[dict]:
    return [
        {
            "chunk_id": f"INS-V2-{i:06d}",
            "domain": "insurance",
            "source_file": "doc.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": f"Regulatory clause text {i}",
        }
        for i in range(1, count + 1)
    ]


def _sample_valid_embeddings(count: int = 5) -> list[dict]:
    return [
        {
            "chunk_id": f"INS-V2-{i:06d}",
            "embedding": [0.1] * 768,
        }
        for i in range(1, count + 1)
    ]


def test_validate_local_inputs_valid():
    chunks = _sample_valid_chunks(5)
    embs = _sample_valid_embeddings(5)
    valid, msg = validate_local_inputs(chunks, embs, expected_count=5)
    assert valid is True
    assert "passed" in msg


def test_validate_local_inputs_count_mismatch():
    chunks = _sample_valid_chunks(5)
    embs = _sample_valid_embeddings(4)
    valid, msg = validate_local_inputs(chunks, embs, expected_count=5)
    assert valid is False
    assert "Embedding count mismatch" in msg


def test_validate_local_inputs_wrong_dimension():
    chunks = _sample_valid_chunks(1)
    embs = [{"chunk_id": "INS-V2-000001", "embedding": [0.1] * 512}]
    valid, msg = validate_local_inputs(chunks, embs, expected_count=1)
    assert valid is False
    assert "dimension is 512" in msg


def test_validate_local_inputs_wrong_domain():
    chunks = _sample_valid_chunks(1)
    chunks[0]["domain"] = "labour"
    embs = _sample_valid_embeddings(1)
    valid, msg = validate_local_inputs(chunks, embs, expected_count=1)
    assert valid is False
    assert "not 'insurance'" in msg


def test_validate_local_inputs_invalid_chunk_id():
    chunks = _sample_valid_chunks(1)
    chunks[0]["chunk_id"] = "OLD-001"
    embs = _sample_valid_embeddings(1)
    embs[0]["chunk_id"] = "OLD-001"
    valid, msg = validate_local_inputs(chunks, embs, expected_count=1)
    assert valid is False
    assert "Invalid chunk_id format" in msg


def test_validate_local_inputs_nan_embedding():
    chunks = _sample_valid_chunks(1)
    embs = _sample_valid_embeddings(1)
    embs[0]["embedding"][0] = float("nan")
    valid, msg = validate_local_inputs(chunks, embs, expected_count=1)
    assert valid is False
    assert "non-numeric/NaN" in msg


def test_actual_v2_files_local_validation():
    """Verify that the actual generated Phase 5B V2 files pass local validation."""
    assert CHUNKS_FILE.exists()
    assert EMBEDDINGS_FILE.exists()

    chunks = read_jsonl(CHUNKS_FILE)
    embs = read_jsonl(EMBEDDINGS_FILE)

    valid, msg = validate_local_inputs(chunks, embs, expected_count=647, expected_dim=768)
    assert valid is True, f"Local validation failed on actual files: {msg}"
