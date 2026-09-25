"""
tests/test_api.py
=================
CivicSync Phase 6A — FastAPI Unit Tests

All tests are fully offline / mocked.
No database, no Ollama, no Supabase connections required.

Test coverage:
  1.  GET /api/health
  2.  POST /api/query -- valid query
  3.  POST /api/query -- empty message
  4.  POST /api/query -- message > 2000 chars
  5.  POST /api/query -- language field (default + explicit)
  6.  POST /api/query -- Civic Engine failure (HTTP 500 safe response)
  7.  Response structure validation
  8.  Source traceability preserved
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Import the app — schemas are pure Pydantic, no ML side-effects at import
# ---------------------------------------------------------------------------
from backend.main import app

client = TestClient(app, raise_server_exceptions=False)

# ---------------------------------------------------------------------------
# Shared mock fixture — realistic civic engine output for a labour query
# ---------------------------------------------------------------------------
MOCK_ENGINE_SUCCESS = {
    "type": "success",
    "situation": {
        "original_query": "My employer has not paid my salary for two months.",
        "domain":     "labour",
        "situation":  "Employer has withheld salary for two months",
        "issue":      "unpaid wages",
        "intent":     "seek_remedy",
        "jurisdiction": "India",
        "entities":   ["employer", "salary"],
        "confidence": 0.93,
    },
    "qualification": {
        "status": "supported",
        "domain": "labour",
        "issue":  "unpaid wages",
        "applicable_information": [
            {
                "text": "Every employer must pay wages before the 7th of the following month.",
                "evidence": [
                    {"chunk_id": 101, "source_file": "civicsync_labour_laws", "page_start": 12, "page_end": 12, "similarity": 0.87}
                ],
            }
        ],
        "rights_or_protections": [
            {
                "text": "A worker is entitled to receive wages on time under the Payment of Wages Act.",
                "evidence": [
                    {"chunk_id": 102, "source_file": "civicsync_labour_laws", "page_start": 14, "page_end": 14, "similarity": 0.85}
                ],
            }
        ],
        "possible_actions": [
            {
                "text": "File a complaint with the Labour Commissioner.",
                "evidence": [
                    {"chunk_id": 103, "source_file": "civicsync_labour_laws", "page_start": 20, "page_end": 20, "similarity": 0.82}
                ],
            }
        ],
        "authorities_or_channels": [
            {
                "text": "Labour Commissioner",
                "evidence": [
                    {"chunk_id": 103, "source_file": "civicsync_labour_laws", "page_start": 20, "page_end": 20, "similarity": 0.82}
                ],
            }
        ],
        "documents_or_evidence": [],
        "conditions": [],
        "missing_information": ["Exact salary amount not specified."],
        "sources": [
            {"chunk_id": 101, "source_file": "civicsync_labour_laws", "page_start": 12, "page_end": 12, "similarity": 0.87}
        ],
        "chunk_quality_report": [],
    },
    "action_plan": {
        "immediate_steps": [
            {
                "text": "Send a written notice to the employer demanding payment within 7 days.",
                "evidence": [{"chunk_id": 101, "source_file": "civicsync_labour_laws", "page_start": 12, "page_end": 12, "similarity": 0.87}],
            }
        ],
        "formal_remedies": [
            {
                "text": "File a complaint with the Labour Commissioner.",
                "evidence": [{"chunk_id": 103, "source_file": "civicsync_labour_laws", "page_start": 20, "page_end": 20, "similarity": 0.82}],
            }
        ],
        "authority_pathway": [
            {
                "text": "Labour Commissioner",
                "evidence": [{"chunk_id": 103, "source_file": "civicsync_labour_laws", "page_start": 20, "page_end": 20, "similarity": 0.82}],
            }
        ],
        "escalation": [],
        "information_to_collect": ["Salary slips", "Employment contract"],
        "status": "supported",
    },
    "rag_response": (
        "SITUATION\nYour employer has not paid your salary for two months.\n\n"
        "WHAT THE SOURCES SAY\nUnder the Payment of Wages Act, employers must pay wages before the 7th of the following month.\n\n"
        "YOUR RIGHTS / PROTECTIONS\nYou are entitled to receive wages on time.\n\n"
        "WHAT YOU CAN DO\nFile a complaint with the Labour Commissioner.\n\n"
        "WHERE TO GO\nLabour Commissioner\n\n"
        "WHAT TO KEEP READY\nSalary slips and employment contract.\n\n"
        "WHAT IS STILL UNCLEAR\nExact salary amount not specified.\n\n"
        "LEGAL BASIS\nPayment of Wages Act.\n\n"
        "SOURCES\ncivicsync_labour_laws, page 12, 14, 20.\n\n"
        "DISCLAIMER\nThis is legal information and awareness only, not professional legal advice. "
        "For personalized guidance, consult a qualified lawyer."
    ),
    "retrieved_sources": [
        {"chunk_id": 101, "domain": "labour", "source_file": "civicsync_labour_laws", "page_start": 12, "page_end": 12, "similarity": 0.87},
        {"chunk_id": 102, "domain": "labour", "source_file": "civicsync_labour_laws", "page_start": 14, "page_end": 14, "similarity": 0.85},
        {"chunk_id": 103, "domain": "labour", "source_file": "civicsync_labour_laws", "page_start": 20, "page_end": 20, "similarity": 0.82},
    ],
    "similarity_scores": [0.87, 0.85, 0.82],
    "model_name": "llama3.1:8b",
}

MOCK_ENGINE_CLARIFICATION = {
    "type": "clarification_required",
    "message": (
        "The situation description is ambiguous or falls outside CivicSync's supported legal domains. "
        "Please provide more specific details about your situation."
    ),
    "situation": {
        "original_query": "what?",
        "domain": "unknown",
        "situation": "unknown",
        "issue": "unknown",
        "intent": "unknown",
        "jurisdiction": "India",
        "entities": [],
        "confidence": 0.20,
    },
    "qualification":     None,
    "action_plan":       None,
    "retrieved_sources": [],
    "similarity_scores": [],
    "rag_response":      None,
    "model_name":        "llama3.1:8b",
}


# =========================================================================
# Test 1 — Health endpoint
# =========================================================================
class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/api/health")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    def test_health_response_body(self):
        response = client.get("/api/health")
        body = response.json()
        assert body["status"] == "ok"
        assert body["service"] == "civicsync-api"

    def test_health_content_type_json(self):
        response = client.get("/api/health")
        assert "application/json" in response.headers["content-type"]


# =========================================================================
# Test 2 — Valid query
# =========================================================================
class TestValidQuery:
    @patch("backend.main.process_query", return_value=MOCK_ENGINE_SUCCESS)
    def test_valid_query_returns_200(self, mock_pq):
        response = client.post(
            "/api/query",
            json={"message": "My employer has not paid my salary for two months."}
        )
        assert response.status_code == 200

    @patch("backend.main.process_query", return_value=MOCK_ENGINE_SUCCESS)
    def test_valid_query_calls_process_query(self, mock_pq):
        client.post(
            "/api/query",
            json={"message": "My employer has not paid my salary for two months."}
        )
        mock_pq.assert_called_once()

    @patch("backend.main.process_query", return_value=MOCK_ENGINE_SUCCESS)
    def test_valid_query_message_passed_to_engine(self, mock_pq):
        msg = "My employer has not paid my salary for two months."
        client.post("/api/query", json={"message": msg})
        call_kwargs = mock_pq.call_args
        assert call_kwargs.kwargs.get("user_query") == msg or \
               (call_kwargs.args and call_kwargs.args[0] == msg)

    @patch("backend.main.process_query", return_value=MOCK_ENGINE_SUCCESS)
    def test_valid_query_status_success(self, mock_pq):
        response = client.post(
            "/api/query",
            json={"message": "My employer has not paid my salary for two months."}
        )
        body = response.json()
        assert body["status"] == "success"


# =========================================================================
# Test 3 — Empty message validation
# =========================================================================
class TestEmptyMessage:
    def test_empty_string_returns_422(self):
        response = client.post("/api/query", json={"message": ""})
        assert response.status_code == 422

    def test_whitespace_only_returns_422(self):
        response = client.post("/api/query", json={"message": "   "})
        assert response.status_code == 422

    def test_missing_message_field_returns_422(self):
        response = client.post("/api/query", json={"language": "auto"})
        assert response.status_code == 422


# =========================================================================
# Test 4 — Message length validation
# =========================================================================
class TestMessageLength:
    def test_2000_char_message_accepted(self):
        """Exactly 2000 chars should be accepted."""
        msg = "a" * 2000
        with patch("backend.main.process_query", return_value=MOCK_ENGINE_SUCCESS):
            response = client.post("/api/query", json={"message": msg})
        assert response.status_code == 200

    def test_2001_char_message_rejected(self):
        """2001 chars should be rejected with 422."""
        msg = "a" * 2001
        response = client.post("/api/query", json={"message": msg})
        assert response.status_code == 422

    def test_1_char_message_accepted(self):
        """Single non-whitespace character must be accepted."""
        with patch("backend.main.process_query", return_value=MOCK_ENGINE_SUCCESS):
            response = client.post("/api/query", json={"message": "x"})
        assert response.status_code == 200


# =========================================================================
# Test 5 — Language field
# =========================================================================
class TestLanguageField:
    @patch("backend.main.process_query", return_value=MOCK_ENGINE_SUCCESS)
    def test_default_language_is_auto(self, mock_pq):
        """Omitting language field should default to 'auto'."""
        response = client.post(
            "/api/query",
            json={"message": "My employer has not paid my salary."}
        )
        assert response.status_code == 200

    @patch("backend.main.process_query", return_value=MOCK_ENGINE_SUCCESS)
    def test_explicit_language_accepted(self, mock_pq):
        response = client.post(
            "/api/query",
            json={"message": "My employer has not paid my salary.", "language": "en"}
        )
        assert response.status_code == 200

    @patch("backend.main.process_query", return_value=MOCK_ENGINE_SUCCESS)
    def test_language_auto_accepted(self, mock_pq):
        response = client.post(
            "/api/query",
            json={"message": "My employer has not paid my salary.", "language": "auto"}
        )
        assert response.status_code == 200


# =========================================================================
# Test 6 — Civic Engine failure -> safe HTTP 500
# =========================================================================
class TestEngineFailure:
    def test_engine_exception_returns_500(self):
        with patch("backend.main.process_query", side_effect=RuntimeError("Supabase timed out")):
            response = client.post(
                "/api/query",
                json={"message": "My employer has not paid my salary."}
            )
        assert response.status_code == 500

    def test_engine_exception_hides_internal_detail(self):
        """Stack traces and credential details must not appear in the response."""
        with patch("backend.main.process_query", side_effect=RuntimeError("SUPABASE_SECRET_KEY=abc123")):
            response = client.post(
                "/api/query",
                json={"message": "My employer has not paid my salary."}
            )
        body = response.text
        assert "SUPABASE_SECRET_KEY" not in body
        assert "abc123" not in body
        assert "Traceback" not in body

    def test_engine_exception_response_has_status_error(self):
        with patch("backend.main.process_query", side_effect=Exception("broken")):
            response = client.post(
                "/api/query",
                json={"message": "My employer has not paid my salary."}
            )
        body = response.json()
        assert body.get("status") == "error"


# =========================================================================
# Test 7 — Response structure validation
# =========================================================================
class TestResponseStructure:
    @pytest.fixture(autouse=True)
    def _setup(self):
        with patch("backend.main.process_query", return_value=MOCK_ENGINE_SUCCESS):
            self.response = client.post(
                "/api/query",
                json={"message": "My employer has not paid my salary for two months."}
            )
            self.body = self.response.json()

    def test_has_status_field(self):
        assert "status" in self.body

    def test_has_situation_field(self):
        assert "situation" in self.body
        assert self.body["situation"] is not None

    def test_has_qualification_field(self):
        assert "qualification" in self.body
        assert self.body["qualification"] is not None

    def test_has_actions_field(self):
        assert "actions" in self.body

    def test_has_answer_field(self):
        assert "answer" in self.body
        assert self.body["answer"] is not None and len(self.body["answer"]) > 0

    def test_has_sources_field_as_list(self):
        assert "sources" in self.body
        assert isinstance(self.body["sources"], list)

    def test_has_limitations_field_as_list(self):
        assert "limitations" in self.body
        assert isinstance(self.body["limitations"], list)

    def test_situation_has_domain(self):
        assert self.body["situation"]["domain"] == "labour"

    def test_situation_has_issue(self):
        assert "wage" in self.body["situation"]["issue"].lower() or \
               "salary" in self.body["situation"]["issue"].lower() or \
               "unpaid" in self.body["situation"]["issue"].lower()

    def test_qualification_status_is_supported(self):
        assert self.body["qualification"]["status"] == "supported"

    def test_answer_contains_text(self):
        assert len(self.body["answer"]) > 50

    def test_clarification_required_maps_correctly(self):
        """When engine returns clarification_required, status must reflect that."""
        with patch("backend.main.process_query", return_value=MOCK_ENGINE_CLARIFICATION):
            resp = client.post("/api/query", json={"message": "what?"})
        body = resp.json()
        assert body["status"] == "clarification_required"


# =========================================================================
# Test 8 — Source traceability preserved
# =========================================================================
class TestSourceTraceability:
    @pytest.fixture(autouse=True)
    def _setup(self):
        with patch("backend.main.process_query", return_value=MOCK_ENGINE_SUCCESS):
            response = client.post(
                "/api/query",
                json={"message": "My employer has not paid my salary for two months."}
            )
            self.body = response.json()
            self.sources = self.body.get("sources", [])

    def test_sources_list_not_empty(self):
        assert len(self.sources) > 0, "Response must include retrieved sources"

    def test_every_source_has_source_file(self):
        for s in self.sources:
            assert s.get("source_file"), f"Source missing source_file: {s}"

    def test_every_source_has_page_start(self):
        for s in self.sources:
            assert s.get("page_start") is not None, f"Source missing page_start: {s}"

    def test_every_source_has_similarity(self):
        for s in self.sources:
            assert s.get("similarity") is not None, f"Source missing similarity: {s}"
            assert 0 < s["similarity"] <= 1.0

    def test_every_source_has_chunk_id(self):
        for s in self.sources:
            assert s.get("chunk_id") is not None, f"Source missing chunk_id: {s}"

    def test_every_source_has_domain(self):
        for s in self.sources:
            assert s.get("domain") == "labour"

    def test_source_count_matches_engine_output(self):
        assert len(self.sources) == len(MOCK_ENGINE_SUCCESS["retrieved_sources"])

    def test_source_page_numbers_preserved(self):
        expected_pages = {s["page_start"] for s in MOCK_ENGINE_SUCCESS["retrieved_sources"]}
        actual_pages   = {s["page_start"] for s in self.sources}
        assert expected_pages == actual_pages, \
            f"Page numbers lost in serialization. Expected {expected_pages}, got {actual_pages}"


# =========================================================================
# Test 9 — Conversation Flow
# =========================================================================
class TestConversationFlow:
    @patch("backend.main.process_query", return_value=MOCK_ENGINE_SUCCESS)
    def test_new_conversation_generates_id(self, mock_pq):
        response = client.post(
            "/api/query",
            json={"message": "My employer has not paid my salary."}
        )
        assert response.status_code == 200
        body = response.json()
        assert "conversation_id" in body
        assert body["conversation_id"] is not None
        assert len(body["conversation_id"]) > 0

    @patch("backend.main.generate", return_value="Standalone reformulated query")
    @patch("backend.main.process_query", return_value=MOCK_ENGINE_SUCCESS)
    def test_follow_up_message_reformulates_query(self, mock_pq, mock_gen):
        # First query to establish history
        r1 = client.post(
            "/api/query",
            json={"message": "My employer has not paid my salary.", "conversation_id": "test_conv_123"}
        )
        assert r1.status_code == 200

        # Reset mock calls before follow-up
        mock_pq.reset_mock()
        mock_gen.reset_mock()

        # Follow-up query
        r2 = client.post(
            "/api/query",
            json={"message": "Gujarat", "conversation_id": "test_conv_123"}
        )
        assert r2.status_code == 200
        
        # Verify query reformulation was called
        mock_gen.assert_called_once()
        # Verify process_query was called with reformulated query
        mock_pq.assert_called_once_with(user_query="Standalone reformulated query", top_k=5)

    @patch("backend.main.process_query", return_value=MOCK_ENGINE_SUCCESS)
    def test_conversation_id_persistence(self, mock_pq):
        conv_id = "persistent_id_456"
        response = client.post(
            "/api/query",
            json={"message": "My employer has not paid my salary.", "conversation_id": conv_id}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["conversation_id"] == conv_id

    @patch("backend.main.process_query", return_value=MOCK_ENGINE_SUCCESS)
    def test_english_language(self, mock_pq):
        response = client.post(
            "/api/query",
            json={"message": "My employer has not paid my salary.", "language": "en"}
        )
        assert response.status_code == 200
        mock_pq.assert_called_once()

    @patch("backend.main.process_query", return_value=MOCK_ENGINE_SUCCESS)
    def test_hindi_language(self, mock_pq):
        response = client.post(
            "/api/query",
            json={"message": "My employer has not paid my salary.", "language": "hi"}
        )
        assert response.status_code == 200
        mock_pq.assert_called_once()

    def test_insufficient_evidence_friendly_answer(self):
        mock_insufficient = {
            "type": "success",
            "situation": {"domain": "insurance", "issue": "claim rejection"},
            "qualification": {"status": "insufficient_evidence"},
            "action_plan": {},
            "rag_response": "The available CivicSync sources do not contain enough information to address this.",
            "retrieved_sources": [],
        }
        with patch("backend.main.process_query", return_value=mock_insufficient):
            response = client.post(
                "/api/query",
                json={"message": "I want to claim for cosmetic surgery."}
            )
        assert response.status_code == 200
        body = response.json()
        assert "don't have enough reliable information" in body["answer"]

    @patch("backend.main.process_query", return_value=MOCK_ENGINE_SUCCESS)
    def test_backward_compatibility(self, mock_pq):
        # No conversation_id supplied at all
        response = client.post(
            "/api/query",
            json={"message": "My employer has not paid my salary for two months."}
        )
        assert response.status_code == 200
        body = response.json()
        assert "conversation_id" in body

