"""
test_qualification.py
=====================
Unit tests for CivicSync Phase 5A — Refined Legal Qualification Engine.
Mocks Ollama LLM so all tests run offline.
"""

from __future__ import annotations

import json
from unittest.mock import patch
import pytest

from src.qualification import (
    qualify_case,
    _resolve_sources,
    _determine_status,
    _parse_json_from_llm,
    _score_chunk,
    STATUS_SUPPORTED,
    STATUS_PARTIALLY_SUPPORTED,
    STATUS_INSUFFICIENT,
)


# ---------------------------------------------------------------------------
# Sample test chunks
# ---------------------------------------------------------------------------
CHUNK_TRAFFIC = {
    "chunk_id": 3,
    "domain": "traffic",
    "source_file": "india_traffic_rules_formatted.pdf",
    "page_start": 3,
    "page_end": 3,
    "similarity": 0.50,  # Low similarity score to prove similarity does not control status
    "content": "Driving without an effective driving licence — NO_VALID_LICENCE. "
               "Motor Vehicles Act, 1988 Sections 3 and 181. "
               "A driver must hold a valid licence authorising the relevant vehicle class.",
}

CHUNK_LABOUR = {
    "chunk_id": 13,
    "domain": "labour",
    "source_file": "Indian_Labour_Rights_ML_Dataset_v5 (1).pdf",
    "page_start": 4,
    "page_end": 5,
    "similarity": 0.40,
    "content": "Step 1: Issue a formal written demand to the employer specifying unpaid amounts. "
               "Step 2: File a complaint before the Labour Commissioner or Authority "
               "under the Payment of Wages Act.",
}

CHUNK_METADATA_ONLY = {
    "chunk_id": 109,
    "domain": "women_safety",
    "source_file": "Women Safety and Rights.pdf",
    "page_start": 12,
    "page_end": 14,
    "similarity": 0.85,  # High similarity but pure structural metadata!
    "content": "Page 12\n0.98Official government service source 2026-08-18\n"
               "Page 13\ncategory\nlegal_awareness\npolice_procedure\ndomestic_violence\n"
               "workplace_harassment\nchild_protection\ndowry\ntrafficking",
}

SITUATION_TRAFFIC = {
    "original_query": "Traffic police stopped me without a licence.",
    "domain": "traffic",
    "situation": "stopped_by_traffic_police",
    "issue": "driving_licence",
    "intent": "understand_consequences",
    "confidence": 0.95,
}


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

def test_1_one_valid_chunk_is_sufficient_for_supported():
    """Requirement 1: One valid chunk with a core claim produces supported status."""
    items = [{"text": "Driver must hold a valid licence.", "evidence": [{"chunk_id": 3}]}]
    status = _determine_status(items, [], [], [], [CHUNK_TRAFFIC])
    assert status == STATUS_SUPPORTED


def test_2_two_valid_chunks_work():
    """Requirement 2: Two valid chunks with core claims work as supported."""
    items = [
        {"text": "Claim 1", "evidence": [{"chunk_id": 3}]},
        {"text": "Claim 2", "evidence": [{"chunk_id": 13}]},
    ]
    status = _determine_status(items, [], [], [], [CHUNK_TRAFFIC, CHUNK_LABOUR])
    assert status == STATUS_SUPPORTED


def test_3_invalid_chunk_reference_is_rejected():
    """Requirement 3: Invalid chunk reference is rejected during resolving."""
    items = [{"text": "Invalid ref claim", "chunk_refs": [99]}]  # Chunk ref 99 doesn't exist
    resolved = _resolve_sources(items, [CHUNK_TRAFFIC])
    assert resolved == []  # Discarded


def test_4_unsupported_claim_is_rejected():
    """Requirement 4: Unsupported claim without valid chunk_refs is rejected."""
    items = [
        {"text": "Valid claim", "chunk_refs": [1]},
        {"text": "Unsupported claim", "chunk_refs": []},
    ]
    resolved = _resolve_sources(items, [CHUNK_TRAFFIC])
    assert len(resolved) == 1
    assert resolved[0]["text"] == "Valid claim"


def test_5_similarity_score_does_not_control_status():
    """Requirement 5: Low similarity (0.40) does NOT force INSUFFICIENT status if evidence exists."""
    items = [{"text": "Issue formal written demand", "evidence": [{"chunk_id": 13}]}]
    # CHUNK_LABOUR has similarity = 0.40
    status = _determine_status(items, [], [], [], [CHUNK_LABOUR])
    assert status == STATUS_SUPPORTED


def test_8_source_traceability_preserved():
    """Requirement 8: Resolved items preserve full evidence metadata structure."""
    items = [{"text": "Must hold valid licence", "chunk_refs": [1]}]
    resolved = _resolve_sources(items, [CHUNK_TRAFFIC])
    assert len(resolved) == 1
    item = resolved[0]
    assert "evidence" in item
    assert len(item["evidence"]) == 1
    ev = item["evidence"][0]
    assert ev["chunk_id"] == 3
    assert ev["source_file"] == "india_traffic_rules_formatted.pdf"
    assert ev["page_start"] == 3
    assert ev["similarity"] == 0.50


def test_9_dataset_metadata_is_not_treated_as_substantive_evidence():
    """Requirement 9: Metadata-dominant chunk is identified by _score_chunk."""
    score = _score_chunk(CHUNK_METADATA_ONLY["content"])
    assert score["is_metadata_dominant"] is True
    assert score["is_substantive"] is False


def test_10_partial_evidence_produces_partially_supported():
    """Requirement 10: Having actions/authorities but no core applicable/rights info yields partially_supported."""
    actions = [{"text": "File complaint", "evidence": [{"chunk_id": 13}]}]
    status = _determine_status([], [], actions, [], [CHUNK_LABOUR])
    assert status == STATUS_PARTIALLY_SUPPORTED


def test_11_no_evidence_produces_insufficient_evidence():
    """Requirement 11: No valid evidence yields insufficient_evidence status."""
    status = _determine_status([], [], [], [], [CHUNK_TRAFFIC])
    assert status == STATUS_INSUFFICIENT


@patch("src.qualification.generate")
def test_qualify_case_full_flow(mock_generate):
    """End-to-end qualify_case with mocked LLM output returning chunk_refs array."""
    mock_generate.return_value = json.dumps({
        "applicable_information": [
            {"text": "A driver must hold a valid licence.", "chunk_refs": [1]}
        ],
        "rights_or_protections": [],
        "possible_actions": [
            {"text": "Carry valid licence while driving.", "chunk_refs": [1]}
        ],
        "authorities_or_channels": [],
        "documents_or_evidence": [],
        "conditions": [],
        "missing_information": ["State of issuance."],
    })

    result = qualify_case(SITUATION_TRAFFIC, [CHUNK_TRAFFIC])

    assert result["status"] == STATUS_SUPPORTED
    assert len(result["applicable_information"]) == 1
    assert result["applicable_information"][0]["evidence"][0]["chunk_id"] == 3
    assert len(result["possible_actions"]) == 1
    assert result["possible_actions"][0]["evidence"][0]["chunk_id"] == 3
    assert "sources" in result
    assert len(result["sources"]) == 1
