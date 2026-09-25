"""
test_action_engine.py
=====================
Unit tests for CivicSync Phase 5A — Action Engine.
Verifies that actions and authorities strictly require valid evidence.
"""

from __future__ import annotations

import pytest
from src.action_engine import build_action_plan


def _make_qualification_with_evidence():
    return {
        "status": "supported",
        "domain": "labour",
        "issue": "unpaid_wages",
        "applicable_information": [
            {
                "text": "Salary must be paid on time under the law.",
                "evidence": [{"chunk_id": 13, "source_file": "labour.pdf", "page_start": 4}],
            }
        ],
        "rights_or_protections": [],
        "possible_actions": [
            {
                "text": "Step 1: Issue a formal written demand to the employer.",
                "evidence": [{"chunk_id": 13, "source_file": "labour.pdf", "page_start": 4}],
            },
            {
                "text": "Unsourced action that should be ignored",
                "evidence": [],  # Empty evidence -> MUST BE EXCLUDED
            },
        ],
        "authorities_or_channels": [
            {
                "text": "Labour Commissioner",
                "evidence": [{"chunk_id": 13, "source_file": "labour.pdf", "page_start": 4}],
            },
            {
                "text": "Unsourced Authority",
                "evidence": [],  # Empty evidence -> MUST BE EXCLUDED
            },
        ],
        "documents_or_evidence": [
            {
                "text": "Employment contract and pay slips.",
                "evidence": [{"chunk_id": 13, "source_file": "labour.pdf", "page_start": 4}],
            }
        ],
        "conditions": [],
        "missing_information": ["Type of contract."],
    }


def test_6_action_requires_evidence():
    """Requirement 6: Actions without evidence are excluded from the action plan."""
    qual = _make_qualification_with_evidence()
    plan = build_action_plan(qual)

    all_actions = plan["immediate_steps"] + plan["formal_remedies"]
    action_texts = [a["text"] for a in all_actions]

    assert any("formal written demand" in t for t in action_texts)
    assert not any("Unsourced action" in t for t in action_texts)


def test_7_authority_requires_evidence():
    """Requirement 7: Authorities without evidence are excluded from the action plan."""
    qual = _make_qualification_with_evidence()
    plan = build_action_plan(qual)

    authority_texts = [a["text"] for a in plan["authority_pathway"]]
    assert any("Labour Commissioner" in t for t in authority_texts)
    assert not any("Unsourced Authority" in t for t in authority_texts)


def test_information_to_collect_combines_documents_and_missing():
    qual = _make_qualification_with_evidence()
    plan = build_action_plan(qual)

    info = plan["information_to_collect"]
    assert "Employment contract and pay slips." in info
    assert "Type of contract." in info


def test_action_plan_with_no_valid_actions():
    qual = {
        "status": "insufficient_evidence",
        "domain": "insurance",
        "issue": "claim_rejected",
        "possible_actions": [],
        "authorities_or_channels": [],
        "documents_or_evidence": [],
        "missing_information": ["No evidence found"],
    }
    plan = build_action_plan(qual)
    assert plan["immediate_steps"] == []
    assert plan["formal_remedies"] == []
    assert plan["authority_pathway"] == []
    assert plan["information_to_collect"] == ["No evidence found"]
