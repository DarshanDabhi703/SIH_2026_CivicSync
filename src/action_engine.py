"""
action_engine.py
================
CivicSync Phase 5A — Action Engine

Builds a practical action plan from qualification output.
Operates ONLY on qualification items that already have valid supporting evidence.
If an item has no valid evidence, it is excluded.
"""

from __future__ import annotations

from typing import Any, Dict, List


def build_action_plan(qualification: Dict[str, Any]) -> Dict[str, Any]:
    """
    Organize qualification output into a structured action plan.

    Parameters
    ----------
    qualification : dict
        Output from qualify_case().

    Returns
    -------
    dict
        Action plan with keys:
        immediate_steps, formal_remedies, authority_pathway,
        information_to_collect, escalation, status.

    IMPORTANT: All items must have valid supporting evidence from qualification.
    """
    possible_actions    = qualification.get("possible_actions", [])
    authorities         = qualification.get("authorities_or_channels", [])
    documents           = qualification.get("documents_or_evidence", [])
    missing_info        = qualification.get("missing_information", [])
    status              = qualification.get("status", "insufficient_evidence")

    # Problem 5: Require valid evidence attached to item
    valid_actions = [
        a for a in possible_actions
        if isinstance(a, dict) and (a.get("evidence") or a.get("chunk_id") or a.get("source_file"))
    ]
    valid_authorities = [
        a for a in authorities
        if isinstance(a, dict) and (a.get("evidence") or a.get("chunk_id") or a.get("source_file"))
    ]
    valid_documents = [
        d for d in documents
        if isinstance(d, dict) and (d.get("evidence") or d.get("chunk_id") or d.get("source_file"))
    ]

    # -------------------------------------------------------------------
    # Classify valid_actions into immediate vs. formal
    # -------------------------------------------------------------------
    IMMEDIATE_KEYWORDS = {
        "immediately", "first", "emergency", "call", "contact",
        "notify", "inform", "alert", "report", "step 1",
    }
    FORMAL_KEYWORDS = {
        "file", "complaint", "formal", "court", "tribunal", "authority",
        "commissioner", "ombudsman", "legal", "lodge", "petition",
        "written", "notice", "step 2",
    }

    immediate_steps: List[Dict] = []
    formal_remedies: List[Dict] = []

    for action in valid_actions:
        text_lower = action.get("text", "").lower()
        if any(kw in text_lower for kw in IMMEDIATE_KEYWORDS):
            immediate_steps.append(action)
        elif any(kw in text_lower for kw in FORMAL_KEYWORDS):
            formal_remedies.append(action)
        else:
            immediate_steps.append(action)

    # -------------------------------------------------------------------
    # Authority pathway — ordered list of escalation channels with evidence
    # -------------------------------------------------------------------
    authority_pathway: List[Dict] = list(valid_authorities)

    # -------------------------------------------------------------------
    # Escalation — authorities containing escalation-related terms
    # -------------------------------------------------------------------
    ESCALATION_KEYWORDS = {
        "ombudsman", "tribunal", "court", "high court",
        "consumer forum", "appeal", "national", "supreme",
    }
    escalation: List[Dict] = [
        a for a in valid_authorities
        if any(kw in a.get("text", "").lower() for kw in ESCALATION_KEYWORDS)
    ]

    # -------------------------------------------------------------------
    # Information to collect = valid document texts + missing info strings
    # -------------------------------------------------------------------
    information_to_collect: List[str] = (
        [d.get("text", "") for d in valid_documents if d.get("text")]
        + list(missing_info)
    )

    return {
        "immediate_steps":       immediate_steps,
        "formal_remedies":       formal_remedies,
        "authority_pathway":     authority_pathway,
        "information_to_collect": information_to_collect,
        "escalation":            escalation,
        "status":                status,
    }
