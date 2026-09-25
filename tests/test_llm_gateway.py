"""
tests/test_llm_gateway.py
==========================
Unit tests for CivicSync Phase 6E — LLM Gateway

Mocks all external HTTP APIs (Groq, Gemini) and local Ollama client so tests
run quickly offline and without actual API keys.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from src.llm_gateway import generate, LLMGatewayError


# ---------------------------------------------------------------------------
# Provider Selection & Routing Tests
# ---------------------------------------------------------------------------

@patch("src.llm_gateway._generate_groq")
def test_groq_provider_selected(mock_groq):
    """If LLM_PROVIDER=groq and GROQ_API_KEY exists, Groq API must be invoked."""
    mock_groq.return_value = "Mocked Groq Response"
    env_mock = {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "gsk_test_12345"}
    
    with patch.dict(os.environ, env_mock):
        res = generate("Ask a question about labour laws")
        assert res == "Mocked Groq Response"
        mock_groq.assert_called_once()


@patch("src.llm_gateway._generate_gemini")
def test_gemini_provider_selected(mock_gemini):
    """If LLM_PROVIDER=gemini and GEMINI_API_KEY exists, Gemini API must be invoked."""
    mock_gemini.return_value = "Mocked Gemini Response"
    env_mock = {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "gemini_test_12345"}
    
    with patch.dict(os.environ, env_mock):
        res = generate("Ask a question about traffic rules")
        assert res == "Mocked Gemini Response"
        mock_gemini.assert_called_once()


@patch("src.llm_gateway._ollama_generate_fallback")
def test_ollama_provider_explicitly_selected(mock_ollama):
    """If LLM_PROVIDER=ollama, local Ollama fallback must be invoked directly."""
    mock_ollama.return_value = "Mocked Ollama Response"
    env_mock = {"LLM_PROVIDER": "ollama"}
    
    with patch.dict(os.environ, env_mock):
        res = generate("Ask a question about property laws")
        assert res == "Mocked Ollama Response"
        mock_ollama.assert_called_once()


# ---------------------------------------------------------------------------
# Fallback & Missing Keys Tests
# ---------------------------------------------------------------------------

@patch("src.llm_gateway._ollama_generate_fallback")
def test_missing_groq_key_falls_back_to_ollama(mock_ollama):
    """If LLM_PROVIDER=groq but GROQ_API_KEY is missing, it must fall back to local Ollama."""
    mock_ollama.return_value = "Ollama Fallback Success"
    
    with patch.dict(os.environ, {"LLM_PROVIDER": "groq"}):
        if "GROQ_API_KEY" in os.environ:
            del os.environ["GROQ_API_KEY"]
            
        res = generate("Explain unpaid salary laws")
        assert res == "Ollama Fallback Success"
        mock_ollama.assert_called_once()


@patch("src.llm_gateway._generate_groq")
@patch("src.llm_gateway._ollama_generate_fallback")
def test_groq_api_failure_falls_back_to_ollama(mock_ollama, mock_groq):
    """If Groq API call throws an error, it must fall back to local Ollama."""
    mock_groq.side_effect = Exception("Groq Cloud API Rate Limit exceeded")
    mock_ollama.return_value = "Ollama Fallback Success"
    env_mock = {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "gsk_test_12345"}
    
    with patch.dict(os.environ, env_mock):
        res = generate("Explain unpaid salary laws")
        assert res == "Ollama Fallback Success"
        mock_groq.assert_called_once()
        mock_ollama.assert_called_once()


@patch("src.llm_gateway.ollama_generate")
def test_entire_gateway_fails_if_fallback_fails(mock_ollama):
    """If local Ollama fallback also throws an exception, LLMGatewayError must be raised."""
    mock_ollama.side_effect = Exception("Ollama daemon is not running")
    env_mock = {"LLM_PROVIDER": "ollama"}
    
    with patch.dict(os.environ, env_mock):
        with pytest.raises(LLMGatewayError) as exc_info:
            generate("Test query")
        assert "All LLM generation paths failed" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Grounding / Safety Validation Tests
# ---------------------------------------------------------------------------

@patch("backend.main.process_query")
def test_insufficient_evidence_response_is_concise_and_safe(mock_pq):
    """If evidence is insufficient, response must be safe and friendly without exposing internals."""
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    
    mock_pq.return_value = {
        "type": "success",
        "situation": {"domain": "labour", "issue": "wage dispute"},
        "qualification": {"status": "insufficient_evidence"},
        "action_plan": {},
        "rag_response": "The available CivicSync sources do not contain enough information to address this.",
        "retrieved_sources": [],
    }

    response = client.post("/api/query", json={"message": "Do I get paid for casual leave?"})
    assert response.status_code == 200
    
    body = response.json()
    assert "don't have enough reliable information" in body["answer"]
    # Check that internal retrieval jargon is absent
    for term in ["chunk", "embedding", "retrieved document", "dataset", "provided text"]:
        assert term not in body["answer"].lower()
