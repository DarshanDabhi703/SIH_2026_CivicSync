"""
test_situation.py
==================
Unit tests for Situation Extraction component (Phase 4).
Mocks Ollama generate function so unit tests execute offline without external dependencies.
"""

from __future__ import annotations

import json
from unittest.mock import patch
import pytest

from src.situation import (
    extract_situation,
    SituationSchema,
    _parse_json_from_llm,
    VALID_DOMAINS,
    VALID_INTENTS,
)


def test_schema_cleaning_and_validation():
    schema = SituationSchema(
        original_query="Test query",
        domain="TRAFFIC",
        intent="SEEK_REMEDY",
        confidence=1.5,  # Out of range
    ).clean_and_validate()

    assert schema.domain == "traffic"
    assert schema.intent == "seek_remedy"
    assert schema.confidence == 1.0
    assert schema.jurisdiction == "India"


def test_unsupported_domain_fallback():
    schema = SituationSchema(
        original_query="I need advice on cryptocurrency tax",
        domain="finance_crypto",
        intent="understand_rights",
        confidence=0.9,
    ).clean_and_validate()

    assert schema.domain == "unknown"
    assert schema.confidence <= 0.40


def test_unsupported_intent_fallback():
    schema = SituationSchema(
        original_query="Test query",
        domain="labour",
        intent="invalid_custom_intent",
        confidence=0.8,
    ).clean_and_validate()

    assert schema.intent == "unknown"


def test_json_parsing_variations():
    # Plain JSON string
    raw1 = '{"domain": "consumer", "situation": "defective_product", "confidence": 0.9}'
    assert _parse_json_from_llm(raw1)["domain"] == "consumer"

    # Markdown fenced JSON block
    raw2 = '```json\n{\n  "domain": "labour",\n  "intent": "seek_remedy"\n}\n```'
    assert _parse_json_from_llm(raw2)["domain"] == "labour"

    # Text before and after JSON
    raw3 = 'Here is the output:\n{"domain": "traffic", "issue": "speeding"}\nHope this helps.'
    assert _parse_json_from_llm(raw3)["domain"] == "traffic"


@patch("src.situation.generate")
def test_extract_traffic_domain(mock_generate):
    mock_generate.return_value = json.dumps({
        "domain": "traffic",
        "situation": "stopped_by_traffic_police",
        "issue": "driving_licence",
        "intent": "understand_consequences",
        "jurisdiction": "India",
        "entities": ["traffic police", "driving licence"],
        "confidence": 0.95,
    })

    result = extract_situation("Traffic police stopped me because I didn't have my driving licence.")

    assert result["domain"] == "traffic"
    assert result["situation"] == "stopped_by_traffic_police"
    assert result["issue"] == "driving_licence"
    assert result["intent"] == "understand_consequences"
    assert result["jurisdiction"] == "India"
    assert "driving licence" in result["entities"]
    assert result["confidence"] == 0.95


@patch("src.situation.generate")
def test_extract_labour_domain(mock_generate):
    mock_generate.return_value = json.dumps({
        "domain": "labour",
        "situation": "unpaid_wages",
        "issue": "salary_payment",
        "intent": "seek_remedy",
        "jurisdiction": "India",
        "entities": ["employer", "salary"],
        "confidence": 0.92,
    })

    result = extract_situation("My company has not paid my salary for two months.")

    assert result["domain"] == "labour"
    assert result["intent"] == "seek_remedy"


@patch("src.situation.generate")
def test_extract_consumer_domain(mock_generate):
    mock_generate.return_value = json.dumps({
        "domain": "consumer",
        "situation": "defective_product",
        "issue": "product_replacement",
        "intent": "seek_remedy",
        "jurisdiction": "India",
        "entities": ["phone", "seller"],
        "confidence": 0.90,
    })

    result = extract_situation("I bought a phone and the seller refuses to replace it.")

    assert result["domain"] == "consumer"
    assert result["issue"] == "product_replacement"


@patch("src.situation.generate")
def test_extract_land_property_domain(mock_generate):
    mock_generate.return_value = json.dumps({
        "domain": "land_property",
        "situation": "security_deposit_dispute",
        "issue": "tenant_security_deposit",
        "intent": "seek_remedy",
        "jurisdiction": "India",
        "entities": ["landlord", "security deposit"],
        "confidence": 0.88,
    })

    result = extract_situation("My landlord is refusing to return my security deposit.")

    assert result["domain"] == "land_property"


@patch("src.situation.generate")
def test_extract_insurance_domain(mock_generate):
    mock_generate.return_value = json.dumps({
        "domain": "insurance",
        "situation": "insurance_claim_rejected",
        "issue": "health_insurance_claim",
        "intent": "seek_remedy",
        "jurisdiction": "India",
        "entities": ["insurer", "claim"],
        "confidence": 0.91,
    })

    result = extract_situation("My insurer rejected my hospital claim.")

    assert result["domain"] == "insurance"


@patch("src.situation.generate")
def test_extract_women_safety_domain(mock_generate):
    mock_generate.return_value = json.dumps({
        "domain": "women_safety",
        "situation": "workplace_harassment",
        "issue": "workplace_harassment",
        "intent": "understand_rights",
        "jurisdiction": "India",
        "entities": ["harassment", "workplace"],
        "confidence": 0.94,
    })

    result = extract_situation("Someone is harassing me at my workplace.")

    assert result["domain"] == "women_safety"
    assert result["intent"] == "understand_rights"


@patch("src.situation.generate")
def test_extract_ambiguous_question_unknown_domain(mock_generate):
    mock_generate.return_value = json.dumps({
        "domain": "unknown",
        "situation": "general_trouble",
        "issue": "unspecified",
        "intent": "seek_remedy",
        "jurisdiction": "India",
        "entities": [],
        "confidence": 0.30,
    })

    result = extract_situation("What can I do if someone is troubling me?")

    assert result["domain"] == "unknown"
    assert result["confidence"] <= 0.40


@patch("src.situation.generate")
def test_malformed_llm_json_fallback(mock_generate):
    # LLM returns completely broken text
    mock_generate.return_value = "I am sorry, I cannot output JSON for this."

    result = extract_situation("Random broken query")

    assert result["domain"] == "unknown"
    assert result["confidence"] == 0.0
    assert result["original_query"] == "Random broken query"
