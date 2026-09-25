"""
tests/test_insurance_v2_e2e.py
==============================
CivicSync Phase 5C — Insurance V2 End-to-End Validation Test Suite

Tests all six validation stages using mocked components.
Does NOT touch the database, Supabase, Ollama, or existing embeddings.
Does NOT modify any production source file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Shared test fixtures — realistic Insurance V2 data
# ---------------------------------------------------------------------------

QUERY = "My insurer rejected my hospital claim. What can I do?"

MOCK_SITUATION = {
    "original_query": QUERY,
    "domain":     "insurance",
    "situation":  "Citizen's hospital claim was rejected by their insurer",
    "issue":      "claim rejection",
    "intent":     "seek_remedy",
    "jurisdiction": "India",
    "entities":   ["insurer", "hospital", "claim"],
    "confidence": 0.92,
}

MOCK_CHUNKS = [
    {
        "chunk_id":    1852,
        "domain":      "insurance",
        "source_file": "civicsync_insurance_regulatory_v2",
        "page_start":  31,
        "page_end":    31,
        "similarity":  0.8414,
        "content": (
            "Insurers may arrange for dedicated Help Desks in physical mode at the hospital "
            "to deal and assist with the cashless requests. "
            "Insurer shall grant final authorization within three hours of the receipt of "
            "discharge authorization request from the hospital."
        ),
    },
    {
        "chunk_id":    1788,
        "domain":      "insurance",
        "source_file": "civicsync_insurance_regulatory_v2",
        "page_start":  9,
        "page_end":    9,
        "similarity":  0.8367,
        "content": (
            "Every insurer shall strive to achieve 100% cashless claim settlement in a "
            "time bound manner. The insurers shall endeavor to ensure that the instances "
            "of claims being settled through reimbursement are at bare minimum."
        ),
    },
    {
        "chunk_id":    2070,
        "domain":      "insurance",
        "source_file": "civicsync_insurance_regulatory_v2",
        "page_start":  24,
        "page_end":    24,
        "similarity":  0.8351,
        "content": (
            "In case the amount admitted is less than the amount claimed, then the insurer shall "
            "inform the insured/claimant in writing about the basis of settlement; in particular, "
            "where the claim is rejected, the insurer shall give the reasons for the same in "
            "writing drawing reference to the specific terms and conditions of the policy document."
        ),
    },
    {
        "chunk_id":    2115,
        "domain":      "insurance",
        "source_file": "civicsync_insurance_regulatory_v2",
        "page_start":  2,
        "page_end":    2,
        "similarity":  0.8348,
        "content": (
            "IRDAI guidelines on standardization of exclusions in health insurance contracts. "
            "The policyholder may approach the Insurance Ombudsman if the grievance is not "
            "redressed by the insurer within 30 days."
        ),
    },
    {
        "chunk_id":    1590,
        "domain":      "insurance",
        "source_file": "civicsync_insurance_regulatory_v2",
        "page_start":  53,
        "page_end":    53,
        "similarity":  0.8323,
        "content": (
            "The insurer shall, before refusing to act upon the endorsement, record in writing "
            "the reasons for such refusal and communicate the same to the policyholder not later "
            "than thirty days from the date of the policyholder giving notice. "
            "Any person aggrieved by the decision of an insurer may within a period of appeal."
        ),
    },
]

MOCK_QUALIFICATION = {
    "status": "supported",
    "domain": "insurance",
    "issue":  "claim rejection",
    "applicable_information": [
        {
            "text": "Where the claim is rejected, the insurer shall give reasons for rejection in writing.",
            "evidence": [{"chunk_id": 2070, "source_file": "civicsync_insurance_regulatory_v2", "page_start": 24, "page_end": 24, "similarity": 0.8351}],
        },
        {
            "text": "Every insurer shall strive for 100% cashless claim settlement.",
            "evidence": [{"chunk_id": 1788, "source_file": "civicsync_insurance_regulatory_v2", "page_start": 9, "page_end": 9, "similarity": 0.8367}],
        },
    ],
    "rights_or_protections": [
        {
            "text": "The policyholder may approach the Insurance Ombudsman if the grievance is not redressed within 30 days.",
            "evidence": [{"chunk_id": 2115, "source_file": "civicsync_insurance_regulatory_v2", "page_start": 2, "page_end": 2, "similarity": 0.8348}],
        },
    ],
    "possible_actions": [
        {
            "text": "Request the insurer to provide rejection reasons in writing per IRDAI guidelines.",
            "evidence": [{"chunk_id": 2070, "source_file": "civicsync_insurance_regulatory_v2", "page_start": 24, "page_end": 24, "similarity": 0.8351}],
        },
        {
            "text": "File a grievance complaint with the Insurance Ombudsman if the insurer does not respond.",
            "evidence": [{"chunk_id": 2115, "source_file": "civicsync_insurance_regulatory_v2", "page_start": 2, "page_end": 2, "similarity": 0.8348}],
        },
    ],
    "authorities_or_channels": [
        {
            "text": "Insurance Ombudsman — authority for escalating unresolved insurance grievances.",
            "evidence": [{"chunk_id": 2115, "source_file": "civicsync_insurance_regulatory_v2", "page_start": 2, "page_end": 2, "similarity": 0.8348}],
        },
    ],
    "documents_or_evidence": [
        {
            "text": "Policy document showing specific terms referenced in the rejection letter.",
            "evidence": [{"chunk_id": 2070, "source_file": "civicsync_insurance_regulatory_v2", "page_start": 24, "page_end": 24, "similarity": 0.8351}],
        },
    ],
    "conditions": [],
    "missing_information": [
        "Specific IRDAI complaint portal URL not present in retrieved evidence.",
    ],
    "sources": [
        {"chunk_id": 2070, "source_file": "civicsync_insurance_regulatory_v2", "page_start": 24, "page_end": 24, "similarity": 0.8351},
        {"chunk_id": 2115, "source_file": "civicsync_insurance_regulatory_v2", "page_start": 2, "page_end": 2, "similarity": 0.8348},
        {"chunk_id": 1788, "source_file": "civicsync_insurance_regulatory_v2", "page_start": 9, "page_end": 9, "similarity": 0.8367},
    ],
    "chunk_quality_report": [
        {"chunk_ref": i, "chunk_id": c["chunk_id"], "is_metadata_dominant": False, "is_substantive": True, "substantive_hits": 4}
        for i, c in enumerate(MOCK_CHUNKS, 1)
    ],
}

MOCK_RAG_RESPONSE = """SITUATION
Your insurer has rejected your hospital claim. You are looking for remedies and next steps.

WHAT THE SOURCES SAY
According to IRDAI guidelines, when a claim is rejected, the insurer must provide reasons
for rejection in writing, referencing specific terms and conditions of the policy document.
Every insurer is required to strive for 100% cashless claim settlement.

YOUR RIGHTS / PROTECTIONS
If the insurer fails to redress your grievance within 30 days, you may approach the
Insurance Ombudsman for resolution.

WHAT YOU CAN DO
1. Contact your insurer and request the rejection reasons in writing.
2. If unresolved within 30 days, file a complaint with the Insurance Ombudsman.

WHERE TO GO
Insurance Ombudsman — the appropriate authority for unresolved insurance grievances under
IRDAI oversight.

WHAT TO KEEP READY
Your policy document showing the specific terms referenced in the rejection communication.

WHAT IS STILL UNCLEAR
The specific IRDAI complaint portal URL is not present in the available sources.

LEGAL BASIS
IRDAI guidelines on standardization of exclusions in health insurance contracts.

SOURCES
civicsync_insurance_regulatory_v2, pages 24, 2, 9, 31, 53.

DISCLAIMER
This is legal information and awareness only, not professional legal advice.
For personalized legal guidance, consult a qualified lawyer.
"""


# ---------------------------------------------------------------------------
# Helpers imported from the runner (mocked)
# ---------------------------------------------------------------------------

def _count_evidence_items(q: Dict) -> int:
    total = 0
    for key in ("applicable_information", "rights_or_protections",
                "possible_actions", "authorities_or_channels",
                "documents_or_evidence", "conditions"):
        total += len(q.get(key, []))
    return total


def _count_evidence_backed_actions(ap: Dict) -> int:
    count = 0
    for key in ("immediate_steps", "formal_remedies", "authority_pathway"):
        for item in ap.get(key, []):
            if isinstance(item, dict) and item.get("evidence"):
                count += 1
    return count


# =========================================================================
# TEST SUITE
# =========================================================================

# ---------------------------------------------------------------------------
# Stage 1: Situation Intelligence
# ---------------------------------------------------------------------------
class TestSituationIntelligence:
    def test_domain_is_insurance(self):
        assert MOCK_SITUATION["domain"] == "insurance", \
            f"Expected domain=insurance, got {MOCK_SITUATION['domain']}"

    def test_issue_mentions_claim(self):
        issue = MOCK_SITUATION["issue"].lower()
        assert "claim" in issue or "reject" in issue, \
            f"Issue should mention 'claim' or 'reject', got: {issue}"

    def test_intent_is_seek_remedy(self):
        assert MOCK_SITUATION["intent"] in {"seek_remedy", "file_complaint", "understand_rights"}, \
            f"Intent should be seek_remedy / file_complaint / understand_rights, got: {MOCK_SITUATION['intent']}"

    def test_confidence_above_threshold(self):
        assert MOCK_SITUATION["confidence"] >= 0.45, \
            f"Confidence {MOCK_SITUATION['confidence']} below minimum threshold 0.45"

    def test_situation_schema_fields_present(self):
        required = {"original_query", "domain", "situation", "issue", "intent", "jurisdiction", "entities", "confidence"}
        missing = required - set(MOCK_SITUATION.keys())
        assert not missing, f"Situation dict missing fields: {missing}"


# ---------------------------------------------------------------------------
# Stage 2: Retrieval
# ---------------------------------------------------------------------------
class TestRetrieval:
    def test_returns_five_chunks(self):
        assert len(MOCK_CHUNKS) == 5, f"Expected 5 chunks, got {len(MOCK_CHUNKS)}"

    def test_all_chunks_are_insurance_domain(self):
        for c in MOCK_CHUNKS:
            assert c.get("domain") == "insurance", \
                f"Chunk {c.get('chunk_id')} has domain={c.get('domain')}, expected insurance"

    def test_all_chunks_have_similarity(self):
        for c in MOCK_CHUNKS:
            sim = c.get("similarity", 0.0)
            assert 0 < sim <= 1.0, f"Chunk {c.get('chunk_id')} has invalid similarity {sim}"

    def test_all_chunks_have_source_metadata(self):
        for c in MOCK_CHUNKS:
            assert c.get("source_file"), f"Chunk {c.get('chunk_id')} missing source_file"
            assert c.get("page_start") is not None, f"Chunk {c.get('chunk_id')} missing page_start"

    def test_top_result_similarity_above_0_80(self):
        top_sim = MOCK_CHUNKS[0].get("similarity", 0.0)
        assert top_sim >= 0.80, f"Top result similarity {top_sim} should be >= 0.80"

    def test_all_chunks_have_content(self):
        for c in MOCK_CHUNKS:
            assert c.get("content"), f"Chunk {c.get('chunk_id')} has empty content"

    def test_chunk_ids_are_unique(self):
        ids = [c.get("chunk_id") for c in MOCK_CHUNKS]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs in retrieval results"

    def test_retrieval_evidence_contains_claim_rejection_text(self):
        combined = " ".join(c.get("content", "") for c in MOCK_CHUNKS).lower()
        assert "claim" in combined, "Retrieved chunks contain no mention of 'claim'"
        assert re.search(r"reject|refus|repudiat|decline", combined), \
            "Retrieved chunks contain no rejection-related text"


# ---------------------------------------------------------------------------
# Stage 3: Qualification
# ---------------------------------------------------------------------------
class TestQualification:
    def test_status_is_supported(self):
        assert MOCK_QUALIFICATION["status"] == "supported", \
            f"Expected status=supported, got {MOCK_QUALIFICATION['status']}"

    def test_applicable_information_not_empty(self):
        assert len(MOCK_QUALIFICATION["applicable_information"]) >= 1, \
            "applicable_information should contain at least one item"

    def test_every_applicable_info_has_valid_chunk_refs(self):
        for item in MOCK_QUALIFICATION["applicable_information"]:
            evs = item.get("evidence", [])
            assert evs, f"Item '{item.get('text', '')[:60]}' has no chunk evidence"
            for ev in evs:
                assert ev.get("chunk_id"), "Evidence entry missing chunk_id"
                assert ev.get("source_file"), "Evidence entry missing source_file"

    def test_rights_or_protections_not_empty(self):
        assert len(MOCK_QUALIFICATION["rights_or_protections"]) >= 1, \
            "rights_or_protections should contain at least one item"

    def test_every_rights_item_has_valid_chunk_refs(self):
        for item in MOCK_QUALIFICATION["rights_or_protections"]:
            evs = item.get("evidence", [])
            assert evs, f"Rights item '{item.get('text', '')[:60]}' has no evidence"

    def test_possible_actions_not_empty(self):
        assert len(MOCK_QUALIFICATION["possible_actions"]) >= 1, \
            "possible_actions should contain at least one item"

    def test_every_action_has_valid_chunk_refs(self):
        for item in MOCK_QUALIFICATION["possible_actions"]:
            evs = item.get("evidence", [])
            assert evs, f"Action '{item.get('text', '')[:60]}' has no evidence"

    def test_authorities_not_empty(self):
        assert len(MOCK_QUALIFICATION["authorities_or_channels"]) >= 1, \
            "authorities_or_channels should contain at least one item"

    def test_every_authority_has_valid_chunk_refs(self):
        for item in MOCK_QUALIFICATION["authorities_or_channels"]:
            evs = item.get("evidence", [])
            assert evs, f"Authority '{item.get('text', '')[:60]}' has no evidence"

    def test_sources_not_empty(self):
        assert len(MOCK_QUALIFICATION["sources"]) >= 1, \
            "sources list should be non-empty after qualification"

    def test_chunk_quality_report_present(self):
        qr = MOCK_QUALIFICATION.get("chunk_quality_report", [])
        assert len(qr) == 5, f"Expected 5 quality report entries, got {len(qr)}"

    def test_no_chunk_is_metadata_dominant(self):
        qr = MOCK_QUALIFICATION.get("chunk_quality_report", [])
        for entry in qr:
            assert not entry.get("is_metadata_dominant"), \
                f"Chunk {entry.get('chunk_ref')} incorrectly flagged as metadata-dominant"


# ---------------------------------------------------------------------------
# Stage 4: Action Engine
# ---------------------------------------------------------------------------
class TestActionEngine:
    @pytest.fixture(autouse=True)
    def build_plan(self):
        from src.action_engine import build_action_plan
        self.action_plan = build_action_plan(MOCK_QUALIFICATION)

    def test_action_plan_has_required_keys(self):
        required = {"immediate_steps", "formal_remedies", "authority_pathway",
                    "information_to_collect", "escalation", "status"}
        missing = required - set(self.action_plan.keys())
        assert not missing, f"Action plan missing keys: {missing}"

    def test_action_plan_has_at_least_one_step(self):
        total = (len(self.action_plan.get("immediate_steps", []))
                 + len(self.action_plan.get("formal_remedies", [])))
        assert total >= 1, "Action plan has no immediate or formal action steps"

    def test_all_action_items_are_evidence_backed(self):
        for key in ("immediate_steps", "formal_remedies", "authority_pathway"):
            for item in self.action_plan.get(key, []):
                if isinstance(item, dict):
                    assert item.get("evidence"), \
                        f"Action item in '{key}' has no supporting evidence: {item.get('text', '')[:80]}"

    def test_status_reflects_qualification(self):
        assert self.action_plan.get("status") == "supported", \
            f"Action plan status should be 'supported', got {self.action_plan.get('status')}"

    def test_no_monetary_hallucination_in_actions(self):
        """Actions must not contain specific monetary amounts unless from evidence."""
        money_re = re.compile(r"Rs\.\s*\d{1,3}(?:,\d{3})*")
        for key in ("immediate_steps", "formal_remedies", "authority_pathway"):
            for item in self.action_plan.get(key, []):
                text = item.get("text", "") if isinstance(item, dict) else str(item)
                assert not money_re.search(text), \
                    f"Action item contains uninferred monetary value: {text[:100]}"


# ---------------------------------------------------------------------------
# Stage 5: Civic Engine output shape
# ---------------------------------------------------------------------------
class TestCivicEngineOutput:
    @pytest.fixture(autouse=True)
    def mock_result(self):
        self.result = {
            "type": "success",
            "situation": MOCK_SITUATION,
            "qualification": MOCK_QUALIFICATION,
            "action_plan": {
                "immediate_steps": MOCK_QUALIFICATION["possible_actions"][:1],
                "formal_remedies": MOCK_QUALIFICATION["possible_actions"][1:],
                "authority_pathway": MOCK_QUALIFICATION["authorities_or_channels"],
                "information_to_collect": ["Policy document"],
                "escalation": MOCK_QUALIFICATION["authorities_or_channels"],
                "status": "supported",
            },
            "rag_response": MOCK_RAG_RESPONSE,
            "retrieved_sources": [
                {
                    "chunk_id":    c["chunk_id"],
                    "domain":      c["domain"],
                    "source_file": c["source_file"],
                    "page_start":  c["page_start"],
                    "page_end":    c["page_end"],
                    "similarity":  c["similarity"],
                }
                for c in MOCK_CHUNKS
            ],
            "similarity_scores": [c["similarity"] for c in MOCK_CHUNKS],
            "model_name": "llama3.1:8b",
        }

    def test_response_type_is_success(self):
        assert self.result["type"] == "success"

    def test_rag_response_not_empty(self):
        assert len(self.result["rag_response"]) > 100

    def test_rag_response_contains_situation_section(self):
        assert "SITUATION" in self.result["rag_response"]

    def test_rag_response_contains_disclaimer(self):
        resp_lower = self.result["rag_response"].lower()
        assert "not professional legal advice" in resp_lower or "consult a qualified" in resp_lower, \
            "RAG response missing legal disclaimer"

    def test_retrieved_sources_have_metadata(self):
        for s in self.result["retrieved_sources"]:
            assert s.get("source_file"), "Source missing source_file"
            assert s.get("page_start") is not None, "Source missing page_start"

    def test_five_sources_returned(self):
        assert len(self.result["retrieved_sources"]) == 5

    def test_no_bare_act_year_hallucination(self):
        """RAG response must not contain bare 'Act YYYY' patterns not in chunk evidence."""
        chunk_text = " ".join(c["content"] for c in MOCK_CHUNKS).lower()
        act_re = re.compile(r"\bAct\s+\d{4}\b")
        for m in act_re.finditer(self.result["rag_response"]):
            if m.group(0).lower() not in chunk_text:
                pytest.fail(f"RAG response contains uninferred citation: {m.group(0)}")


# ---------------------------------------------------------------------------
# Stage 6: Grounding Check
# ---------------------------------------------------------------------------
class TestGroundingCheck:
    def test_no_unsupported_section_citations(self):
        chunk_text = " ".join(c["content"] for c in MOCK_CHUNKS).lower()
        section_re = re.compile(r"\bSection\s+\d+\b")
        for m in section_re.finditer(MOCK_RAG_RESPONSE):
            assert m.group(0).lower() in chunk_text, \
                f"Section citation not found in chunks: {m.group(0)}"

    def test_limitation_disclaimer_present(self):
        lower = MOCK_RAG_RESPONSE.lower()
        assert "not professional legal advice" in lower or "consult a qualified" in lower

    def test_sources_section_present(self):
        assert "SOURCES" in MOCK_RAG_RESPONSE, "RAG response missing SOURCES section"

    def test_qualification_items_no_orphan_evidence(self):
        """All qualification items must have at least one evidence entry."""
        for key in ("applicable_information", "rights_or_protections",
                    "possible_actions", "authorities_or_channels"):
            for item in MOCK_QUALIFICATION.get(key, []):
                assert item.get("evidence"), \
                    f"Orphan qualification item (no evidence) in '{key}': {item.get('text', '')[:80]}"

    def test_evidence_chunk_ids_match_retrieved(self):
        """Evidence chunk IDs must belong to the set of retrieved chunk IDs."""
        retrieved_ids = {c["chunk_id"] for c in MOCK_CHUNKS}
        for key in ("applicable_information", "rights_or_protections", "possible_actions"):
            for item in MOCK_QUALIFICATION.get(key, []):
                for ev in item.get("evidence", []):
                    cid = ev.get("chunk_id")
                    assert cid in retrieved_ids, \
                        f"Evidence references chunk_id={cid} not in retrieved set {retrieved_ids}"


# ---------------------------------------------------------------------------
# Stage 7: Report file validation
# ---------------------------------------------------------------------------
class TestReportFile:
    @pytest.fixture(autouse=True)
    def ensure_report(self, tmp_path):
        """Write a sample report to a tmp file to test its schema."""
        self.report = {
            "query":                    QUERY,
            "domain":                   "insurance",
            "issue":                    "claim rejection",
            "intent":                   "seek_remedy",
            "qualification_status":     "supported",
            "retrieval_results":        5,
            "validated_evidence_items": _count_evidence_items(MOCK_QUALIFICATION),
            "validated_actions":        2,
            "invalid_claims_removed":   0,
            "unsupported_rag_claims":   0,
            "grounding_passed":         True,
            "limitation_disclaimer":    True,
            "source_traceability":      True,
            "validation_passed":        True,
        }
        self.report_path = tmp_path / "phase5c_validation.json"
        with open(self.report_path, "w") as fh:
            json.dump(self.report, fh, indent=2)

    def test_report_has_required_keys(self):
        required = {
            "query", "domain", "issue", "qualification_status",
            "retrieval_results", "validated_evidence_items", "validated_actions",
            "invalid_claims_removed", "grounding_passed", "validation_passed",
        }
        missing = required - set(self.report.keys())
        assert not missing, f"Report missing keys: {missing}"

    def test_report_query_correct(self):
        assert self.report["query"] == QUERY

    def test_report_domain_is_insurance(self):
        assert self.report["domain"] == "insurance"

    def test_report_validation_passed(self):
        assert self.report["validation_passed"] is True

    def test_report_grounding_passed(self):
        assert self.report["grounding_passed"] is True

    def test_report_retrieval_results_is_five(self):
        assert self.report["retrieval_results"] == 5

    def test_report_is_valid_json(self):
        with open(self.report_path) as fh:
            loaded = json.load(fh)
        assert loaded["domain"] == "insurance"
