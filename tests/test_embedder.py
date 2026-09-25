"""
test_embedder.py
================
CivicSync Phase 2A — Embedding Tests

Tests:
    1. Model loading
    2. Embedding dimension == 768
    3. One chunk produces one embedding
    4. All six input files can be processed
    5. Embedding count matches chunk count per domain
    6. Every vector has dimension 768
    7. All embeddings contain only numeric values
    8. chunk_id is preserved in output
    9. Original text is preserved in output
   10. Validation raises on wrong count
   11. Validation raises on wrong dimension
   12. E5 passage prefix is used (not query:)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embedder import (
    load_model,
    embed_chunks,
    read_chunks_jsonl,
    validate_embeddings,
    EMBEDDING_DIM,
    DEFAULT_MODEL,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"

DOMAIN_PAIRS = [
    ("traffic",       "traffic_chunks.jsonl",       "traffic_embeddings.jsonl"),
    ("labour",        "labour_chunks.jsonl",         "labour_embeddings.jsonl"),
    ("women_safety",  "women_safety_chunks.jsonl",   "women_safety_embeddings.jsonl"),
    ("consumer",      "consumer_chunks.jsonl",       "consumer_embeddings.jsonl"),
    ("insurance",     "insurance_chunks.jsonl",      "insurance_embeddings.jsonl"),
    ("land_property", "land_property_chunks.jsonl",  "land_property_embeddings.jsonl"),
]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    """Load the embedding model once for all tests in this module."""
    return load_model(DEFAULT_MODEL)


@pytest.fixture(scope="module")
def sample_chunk():
    """Return a single valid chunk dict for unit tests."""
    return {
        "chunk_id":    "test_0001",
        "source_file": "test.pdf",
        "page_start":  1,
        "page_end":    1,
        "text":        "Driving without a valid driving licence is an offence under the Motor Vehicles Act.",
        "metadata":    {"domain": "traffic"},
    }


# ---------------------------------------------------------------------------
# Test 1: Model loading
# ---------------------------------------------------------------------------

def test_model_loads():
    """Model should load without error."""
    m = load_model(DEFAULT_MODEL)
    assert m is not None


# ---------------------------------------------------------------------------
# Test 2: Embedding dimension == 768
# ---------------------------------------------------------------------------

def test_embedding_dimension(model, sample_chunk):
    """A single chunk should produce a 768-dim embedding."""
    records = embed_chunks([sample_chunk], model)
    assert len(records) == 1
    assert len(records[0]["embedding"]) == EMBEDDING_DIM == 768


# ---------------------------------------------------------------------------
# Test 3: One chunk → one embedding
# ---------------------------------------------------------------------------

def test_one_chunk_one_embedding(model, sample_chunk):
    """embed_chunks with one chunk returns exactly one record."""
    records = embed_chunks([sample_chunk], model)
    assert len(records) == 1


# ---------------------------------------------------------------------------
# Tests 4 & 5: All six files exist and embedding count matches chunk count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("domain, chunk_file, _embed_file", DOMAIN_PAIRS)
def test_chunk_file_exists(domain, chunk_file, _embed_file):
    """Each processed JSONL file must exist."""
    path = PROCESSED_DIR / chunk_file
    assert path.exists(), f"Missing: {path}"


@pytest.mark.parametrize("domain, chunk_file, embed_file", DOMAIN_PAIRS)
def test_embedding_count_matches_chunk_count(domain, chunk_file, embed_file):
    """
    After the pipeline runs, the embedding file must exist and
    contain the same number of records as the chunk file.
    """
    chunk_path = PROCESSED_DIR / chunk_file
    embed_path = EMBEDDINGS_DIR / embed_file

    if not chunk_path.exists():
        pytest.skip(f"Chunk file missing: {chunk_file}")
    if not embed_path.exists():
        pytest.skip(f"Embedding file not yet generated: {embed_file}")

    chunks  = read_chunks_jsonl(chunk_path)
    records = read_chunks_jsonl(embed_path)

    assert len(records) == len(chunks), (
        f"{domain}: {len(chunks)} chunks but {len(records)} embeddings"
    )


# ---------------------------------------------------------------------------
# Test 6: Every vector has dimension 768
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("domain, _chunk_file, embed_file", DOMAIN_PAIRS)
def test_every_vector_dimension_768(domain, _chunk_file, embed_file):
    """Every embedding record in each output file must have dim 768."""
    embed_path = EMBEDDINGS_DIR / embed_file
    if not embed_path.exists():
        pytest.skip(f"Embedding file not yet generated: {embed_file}")

    records = read_chunks_jsonl(embed_path)
    for rec in records:
        assert len(rec["embedding"]) == 768, (
            f"{domain}/{rec['chunk_id']}: dim={len(rec['embedding'])}"
        )


# ---------------------------------------------------------------------------
# Test 7: All numeric values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("domain, _chunk_file, embed_file", DOMAIN_PAIRS)
def test_all_values_numeric(domain, _chunk_file, embed_file):
    """Every value in every embedding must be a float."""
    embed_path = EMBEDDINGS_DIR / embed_file
    if not embed_path.exists():
        pytest.skip(f"Embedding file not yet generated: {embed_file}")

    records = read_chunks_jsonl(embed_path)
    for rec in records:
        assert all(isinstance(v, (int, float)) for v in rec["embedding"]), (
            f"{domain}/{rec['chunk_id']} contains non-numeric values"
        )


# ---------------------------------------------------------------------------
# Test 8: chunk_id preserved
# ---------------------------------------------------------------------------

def test_chunk_id_preserved(model, sample_chunk):
    """chunk_id from original chunk must appear in output record."""
    records = embed_chunks([sample_chunk], model)
    assert records[0]["chunk_id"] == sample_chunk["chunk_id"]


# ---------------------------------------------------------------------------
# Test 9: Text preserved
# ---------------------------------------------------------------------------

def test_text_preserved(model, sample_chunk):
    """Original text must appear verbatim in output record."""
    records = embed_chunks([sample_chunk], model)
    assert records[0]["text"] == sample_chunk["text"]


# ---------------------------------------------------------------------------
# Test 10: validate_embeddings raises on wrong count
# ---------------------------------------------------------------------------

def test_validation_fails_on_wrong_count(model, sample_chunk):
    """validate_embeddings should raise ValueError if counts don't match."""
    records = embed_chunks([sample_chunk], model)
    with pytest.raises(ValueError, match="mismatch"):
        validate_embeddings(records, expected_count=99)


# ---------------------------------------------------------------------------
# Test 11: validate_embeddings raises on bad dimension
# ---------------------------------------------------------------------------

def test_validation_fails_on_wrong_dimension():
    """validate_embeddings should raise if an embedding has wrong dimension."""
    bad_record = {
        "chunk_id":  "bad_001",
        "text":      "some text",
        "embedding": [0.1, 0.2],  # only 2 dims — wrong
    }
    with pytest.raises(ValueError):
        validate_embeddings([bad_record], expected_count=1)


# ---------------------------------------------------------------------------
# Test 12: Passage prefix is applied (not "query:")
# ---------------------------------------------------------------------------

def test_passage_prefix_not_query(model):
    """
    embed_chunks must prepend 'passage: ' to each text.
    We verify this indirectly: encoding with passage prefix vs. query prefix
    should produce different vectors for a real sentence.
    """
    import numpy as np

    text = "What are the penalties for drunk driving in India?"
    chunk = {
        "chunk_id":    "prefix_test",
        "source_file": "test.pdf",
        "page_start":  1,
        "page_end":    1,
        "text":        text,
        "metadata":    {"domain": "traffic"},
    }

    passage_records = embed_chunks([chunk], model)
    passage_vec = passage_records[0]["embedding"]

    # Encode with query prefix directly for comparison
    query_vec = model.encode(
        f"query: {text}", normalize_embeddings=True
    ).tolist()

    # They should NOT be identical
    diff = sum(abs(a - b) for a, b in zip(passage_vec, query_vec))
    assert diff > 0.01, (
        "passage: and query: prefixes produced identical vectors — "
        "prefix may not be applied correctly"
    )


# ---------------------------------------------------------------------------
# Test 13: Empty input returns empty output
# ---------------------------------------------------------------------------

def test_empty_input_returns_empty(model):
    """embed_chunks([]) should return []."""
    result = embed_chunks([], model)
    assert result == []


# ---------------------------------------------------------------------------
# Test 14: metadata fields are populated in output
# ---------------------------------------------------------------------------

def test_output_metadata_fields(model, sample_chunk):
    """Output record should include model_name and dimensions fields."""
    records = embed_chunks([sample_chunk], model)
    rec = records[0]
    assert rec["model_name"] == DEFAULT_MODEL
    assert rec["dimensions"] == 768
    assert rec["domain"] == "traffic"
    assert "page_start" in rec
    assert "page_end" in rec
    assert "source_file" in rec
