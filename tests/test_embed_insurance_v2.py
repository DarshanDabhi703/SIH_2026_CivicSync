"""
test_embed_insurance_v2.py
===========================
Unit & Integration tests for CivicSync Phase 5B-2 — Insurance V2 Embeddings.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import pytest

from src.embed_insurance_v2 import (
    generate_insurance_v2_embeddings,
    validate_insurance_v2_embeddings,
    INPUT_FILE,
    OUTPUT_FILE,
)
from src.embedder import EMBEDDING_DIM, DEFAULT_MODEL


def test_validation_function():
    sample_records = [
        {
            "chunk_id": "INS-V2-000001",
            "source_file": "doc1.pdf",
            "page_start": 1,
            "page_end": 1,
            "domain": "insurance",
            "text": "Sample text",
            "model_name": DEFAULT_MODEL,
            "dimensions": 768,
            "embedding": [1.0 / math.sqrt(768)] * 768,  # L2 norm == 1.0
        }
    ]
    res = validate_insurance_v2_embeddings(sample_records, expected_count=1)
    assert res["all_passed"] is True
    assert res["count_match"] is True
    assert res["dimension"] is True
    assert res["numeric_vectors"] is True
    assert res["no_nulls"] is True
    assert res["metadata"] is True
    assert res["normalized"] is True


def test_insurance_v2_embedding_generation():
    """Run pipeline and verify generated embeddings against all requirements."""
    records, status = generate_insurance_v2_embeddings()

    # 1. Output exists
    assert OUTPUT_FILE.exists()

    # 2. Count == input count (647)
    with open(INPUT_FILE, encoding="utf-8") as fh:
        input_count = len(fh.readlines())

    assert len(records) == input_count == 647
    assert status["all_passed"] is True
    assert status["count_match"] is True

    # 3. Line-by-line checks on output file
    seen_ids = set()
    with open(OUTPUT_FILE, encoding="utf-8") as fh:
        lines = fh.readlines()

    assert len(lines) == 647

    for line in lines:
        rec = json.loads(line)

        # Chunk ID preserved and starts with INS-V2-
        cid = rec["chunk_id"]
        assert cid.startswith("INS-V2-")
        assert cid not in seen_ids
        seen_ids.add(cid)

        # Domain == insurance
        assert rec["domain"] == "insurance"

        # Source metadata preserved
        assert rec["source_file"] != ""
        assert isinstance(rec["page_start"], int)
        assert isinstance(rec["page_end"], int)
        assert rec["page_start"] <= rec["page_end"]
        assert rec["text"].strip() != ""

        # Model and dimensions
        assert rec["model_name"] == DEFAULT_MODEL
        assert rec["dimensions"] == EMBEDDING_DIM

        # Embeddings check
        emb = rec["embedding"]
        assert isinstance(emb, list)
        assert len(emb) == EMBEDDING_DIM

        # No nulls and numeric
        for v in emb:
            assert v is not None
            assert isinstance(v, (int, float))
            assert not math.isnan(v)

        # L2 Normalized (sum of squares ~ 1.0)
        norm_sq = sum(v * v for v in emb)
        assert abs(norm_sq - 1.0) < 1e-3
