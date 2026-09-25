"""
test_retriever.py
=================
CivicSync Phase 2C — Retrieval Tests

Tests:
    1. Query embedding dimension == 768
    2. Query uses "query: " prefix (E5 format constraint)
    3. top_k parameter is respected
    4. domain filter parameter works
    5. metadata fields are present and valid
    6. content field is not null
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import retrieve, get_retrieval_model


# ---------------------------------------------------------------------------
# Test 1: Query embedding dimension == 768
# ---------------------------------------------------------------------------

def test_query_embedding_dimension():
    """Query embedding must be exactly 768 dimensions."""
    model = get_retrieval_model()
    # E5 prefix
    emb = model.encode("query: What is the penalty for overspeeding?", normalize_embeddings=True)
    assert len(emb) == 768


# ---------------------------------------------------------------------------
# Test 2: Query uses "query:" prefix
# ---------------------------------------------------------------------------

def test_query_prefix_not_passage():
    """Query embeddings should differ from passage embeddings for the same text."""
    model = get_retrieval_model()
    text = "What is the penalty for overspeeding?"
    
    query_emb = model.encode(f"query: {text}", normalize_embeddings=True).tolist()
    passage_emb = model.encode(f"passage: {text}", normalize_embeddings=True).tolist()
    
    # Cosine distance check
    diff = sum(abs(a - b) for a, b in zip(query_emb, passage_emb))
    assert diff > 0.01


# ---------------------------------------------------------------------------
# Test 3: top_k parameter is respected (Mocked)
# ---------------------------------------------------------------------------

def test_top_k_respected():
    """RPC call parameter match_count should match top_k."""
    mock_client = MagicMock()
    mock_client.rpc.return_value.execute.return_value.data = [{"chunk_id": 1}]
    
    results = retrieve("overspeeding", top_k=3, domain=None, client=mock_client)
    
    mock_client.rpc.assert_called_once_with(
        "match_chunks",
        {
            "query_embedding": pytest.approx([0.0]*768, abs=2.0), # check parameter exists
            "match_count": 3,
            "filter_domain": None
        }
    )
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Test 4: domain filter parameter works (Mocked)
# ---------------------------------------------------------------------------

def test_domain_filter_works():
    """RPC call parameter filter_domain should match domain."""
    mock_client = MagicMock()
    mock_client.rpc.return_value.execute.return_value.data = []
    
    retrieve("overspeeding", top_k=5, domain="traffic", client=mock_client)
    
    mock_client.rpc.assert_called_once_with(
        "match_chunks",
        {
            "query_embedding": pytest.approx([0.0]*768, abs=2.0),
            "match_count": 5,
            "filter_domain": "traffic"
        }
    )


# ---------------------------------------------------------------------------
# Test 5 & 6: Results metadata completeness (Mocked)
# ---------------------------------------------------------------------------

def test_metadata_and_content_present():
    """Retrieved results must contain required metadata keys and non-null content."""
    mock_client = MagicMock()
    mock_data = [
        {
            "chunk_id": 42,
            "content": "Driving without a valid licence is an offence.",
            "source_file": "india_traffic_rules_formatted.pdf",
            "page_start": 3,
            "page_end": 3,
            "domain": "traffic",
            "similarity": 0.85
        }
    ]
    mock_client.rpc.return_value.execute.return_value.data = mock_data
    
    results = retrieve("licence", top_k=1, domain="traffic", client=mock_client)
    assert len(results) == 1
    res = results[0]
    
    assert res["chunk_id"] == 42
    assert res["content"] == "Driving without a valid licence is an offence."
    assert res["source_file"] == "india_traffic_rules_formatted.pdf"
    assert res["page_start"] == 3
    assert res["page_end"] == 3
    assert res["domain"] == "traffic"
    assert res["similarity"] == 0.85
