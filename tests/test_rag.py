"""
test_rag.py
===========
Unit tests for CivicSync Phase 3 RAG Pipeline.
Mocks both Supabase retriever and Ollama LLM client so tests run quickly offline without dependencies.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest

from src.llm_client import (
    generate,
    OllamaClientError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    DEFAULT_MODEL_NAME,
)
from src.context_builder import build_context
from src.rag import build_rag_prompt, answer_question, SYSTEM_PROMPT


def test_build_context_formatting():
    chunks = [
        {
            "chunk_id": 10,
            "domain": "traffic",
            "source_file": "traffic_rules.pdf",
            "page_start": 3,
            "page_end": 3,
            "similarity": 0.8284,
            "content": "Driving without a valid licence is prohibited under Section 181.",
        },
        {
            "chunk_id": 11,
            "domain": "traffic",
            "source_file": "traffic_rules.pdf",
            "page_start": 4,
            "page_end": 5,
            "similarity": 0.7950,
            "content": "Penalty for underage driving.",
        },
    ]

    context = build_context(chunks)

    assert "[SOURCE 1]" in context
    assert "[SOURCE 2]" in context
    assert "Domain: traffic" in context
    assert "Source File: traffic_rules.pdf" in context
    assert "Page: 3" in context
    assert "Page: 4–5" in context
    assert "Similarity: 0.8284" in context
    assert "Chunk ID: 10" in context
    assert "Driving without a valid licence" in context


def test_build_context_empty():
    context = build_context([])
    assert "No relevant legal evidence retrieved." in context


def test_build_rag_prompt_anti_hallucination_rules():
    question = "What happens if I drive without a licence?"
    context = "[SOURCE 1]\nContent: Section 181 prohibits driving without licence."

    prompt = build_rag_prompt(question, context)

    assert "Answer ONLY using the supplied retrieved legal evidence." in prompt
    assert "Do NOT invent laws" in prompt
    assert "REQUIRED RESPONSE STRUCTURE:" in prompt
    assert "SITUATION" in prompt
    assert "LEGAL INFORMATION" in prompt
    assert "YOUR RIGHTS / PROTECTIONS" in prompt
    assert "WHAT YOU CAN DO" in prompt
    assert "LEGAL BASIS" in prompt
    assert "SOURCES" in prompt
    assert "LIMITATION" in prompt
    assert question in prompt
    assert context in prompt


@patch("urllib.request.urlopen")
def test_llm_client_generate_success(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({"response": "Mocked LLM Answer"}).encode("utf-8")
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    answer = generate("Test prompt", model="custom-llama:8b")

    assert answer == "Mocked LLM Answer"
    mock_urlopen.assert_called_once()


@patch("urllib.request.urlopen")
def test_llm_client_connection_error(mock_urlopen):
    import urllib.error
    mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

    with pytest.raises(OllamaConnectionError):
        generate("Test prompt")


@patch("urllib.request.urlopen")
def test_llm_client_model_not_found(mock_urlopen):
    import urllib.error
    error = urllib.error.HTTPError("http://localhost:11434/api/generate", 404, "Not Found", {}, None)
    mock_urlopen.side_effect = error

    with pytest.raises(OllamaModelNotFoundError):
        generate("Test prompt", model="nonexistent-model")


@patch("src.rag.retrieve")
@patch("src.rag.generate")
def test_answer_question_pipeline(mock_generate, mock_retrieve):
    mock_chunks = [
        {
            "chunk_id": 1,
            "domain": "labour",
            "source_file": "labour_rights.pdf",
            "page_start": 4,
            "page_end": 5,
            "similarity": 0.835,
            "content": "Failure to pay wages on time is actionable under Payment of Wages Act.",
        }
    ]
    mock_retrieve.return_value = mock_chunks
    mock_generate.return_value = (
        "SITUATION\nYou asked about unpaid wages.\n\n"
        "LEGAL INFORMATION\nEmployer must pay wages on time.\n\n"
        "YOUR RIGHTS / PROTECTIONS\nRight to timely payment.\n\n"
        "WHAT YOU CAN DO\nFile a complaint under Payment of Wages Act.\n\n"
        "LEGAL BASIS\nPayment of Wages Act, 1936\n\n"
        "SOURCES\nlabour_rights.pdf, page 4-5\n\n"
        "LIMITATION\nNone."
    )

    result = answer_question("What can I do if my employer has not paid my wages?", top_k=5, domain="labour")

    assert result["question"] == "What can I do if my employer has not paid my wages?"
    assert "SITUATION" in result["answer"]
    assert "Payment of Wages Act" in result["answer"]
    assert len(result["retrieved_sources"]) == 1
    assert result["retrieved_sources"][0]["chunk_id"] == 1
    assert result["similarity_scores"] == [0.835]
    assert result["model_name"] == DEFAULT_MODEL_NAME

    mock_retrieve.assert_called_once_with(
        question="What can I do if my employer has not paid my wages?",
        top_k=5,
        domain="labour",
        client=None,
    )
    mock_generate.assert_called_once()


@patch("src.rag.retrieve")
@patch("src.rag.generate")
def test_unsupported_question_handling(mock_generate, mock_retrieve):
    mock_retrieve.return_value = []
    mock_generate.return_value = (
        "SITUATION\nYou asked about income tax appeal procedures.\n\n"
        "LEGAL INFORMATION\nNo relevant sources found.\n\n"
        "LIMITATION\nThe available CivicSync sources do not contain enough information to answer this question."
    )

    result = answer_question("What is the exact procedure for filing an income tax appeal?")

    assert "do not contain enough information" in result["answer"].lower()
    assert result["retrieved_sources"] == []
    assert result["similarity_scores"] == []
