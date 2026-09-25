from unittest.mock import patch

from src.civic_engine import process_query


@patch("src.civic_engine.generate")
@patch("src.civic_engine.build_action_plan")
@patch("src.civic_engine.qualify_case")
@patch("src.civic_engine.retrieve")
@patch("src.civic_engine.extract_situation")
def test_process_query_success_flow(
    mock_extract,
    mock_retrieve,
    mock_qualify,
    mock_action,
    mock_generate,
):
    mock_extract.return_value = {
        "original_query": "Traffic police stopped me without licence",
        "domain": "traffic",
        "situation": "stopped_by_traffic_police",
        "issue": "driving_licence",
        "intent": "understand_consequences",
        "jurisdiction": "India",
        "entities": ["traffic police"],
        "confidence": 0.95,
    }

    mock_retrieve.return_value = [
        {
            "chunk_id": 3,
            "content": "A driver must hold a valid driving licence.",
            "domain": "traffic",
            "source_file": "traffic_rules.pdf",
            "page_start": 3,
            "page_end": 3,
            "similarity": 0.828,
        }
    ]

    mock_qualify.return_value = {
        "status": "supported",
        "domain": "traffic",
        "issue": "driving_licence",
        "applicable_information": [],
        "rights_or_protections": [],
        "possible_actions": [],
        "authorities_or_channels": [],
        "documents_or_evidence": [],
        "conditions": [],
        "missing_information": [],
    }

    mock_action.return_value = {
        "immediate_steps": [],
        "formal_remedies": [],
        "authority_pathway": [],
        "escalation": [],
        "information_to_collect": [],
    }

    mock_generate.return_value = (
        "SITUATION\n"
        "You were stopped by traffic police.\n\n"
        "WHAT THE SOURCES SAY\n"
        "A valid driving licence is required."
    )

    result = process_query(
        "Traffic police stopped me without licence",
        model="test-model",
    )

    assert result["type"] == "success"
    assert result["situation"]["domain"] == "traffic"
    assert result["qualification"]["status"] == "supported"
    assert result["rag_response"] is not None
    assert len(result["retrieved_sources"]) == 1

    mock_extract.assert_called_once()
    mock_retrieve.assert_called_once_with(
        question="Traffic police stopped me without licence",
        top_k=5,
        domain="traffic",
        client=None,
    )

    mock_qualify.assert_called_once()
    mock_action.assert_called_once_with(mock_qualify.return_value)
    mock_generate.assert_called_once()


@patch("src.civic_engine.extract_situation")
def test_process_query_clarification_on_unknown_domain(mock_extract):
    mock_extract.return_value = {
        "original_query": "What can I do?",
        "domain": "unknown",
        "situation": "unclear",
        "issue": "unknown",
        "intent": "unknown",
        "jurisdiction": "India",
        "entities": [],
        "confidence": 0.30,
    }

    result = process_query("What can I do?")

    assert result["type"] == "clarification_required"
    assert result["situation"]["domain"] == "unknown"
    assert result["qualification"] is None
    assert result["action_plan"] is None
    assert result["retrieved_sources"] == []
    assert result["rag_response"] is None


@patch("src.civic_engine.extract_situation")
def test_process_query_clarification_on_low_confidence(mock_extract):
    mock_extract.return_value = {
        "original_query": "Vague question about property",
        "domain": "land_property",
        "situation": "vague",
        "issue": "unknown",
        "intent": "unknown",
        "jurisdiction": "India",
        "entities": [],
        "confidence": 0.40,
    }

    result = process_query("Vague question about property")

    assert result["type"] == "clarification_required"
    assert result["situation"]["confidence"] == 0.40
    assert result["qualification"] is None
    assert result["action_plan"] is None