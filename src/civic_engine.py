"""
civic_engine.py
===============
CivicSync Phase 5A — Civic Engine Integration Layer

Connects all CivicSync components:
  Situation Extraction → Retrieval → Qualification → Action Plan → Grounded RAG Answer

Phase 4 clarification routing preserved.
Phase 5A adds evidence-first qualification, chunk quality analysis, and enriched context.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.situation import extract_situation
from src.retriever import retrieve
from src.context_builder import build_context
from src.qualification import qualify_case
from src.action_engine import build_action_plan
from src.llm_gateway import generate
from src.llm_client import DEFAULT_MODEL_NAME

import os
import time
import logging

logger = logging.getLogger("civicsync.engine")

# ---------------------------------------------------------------------------
# Phase 5 Grounded RAG System Prompt (structured response format)
# ---------------------------------------------------------------------------
PHASE5_SYSTEM_PROMPT = """You are CivicSync, a legal information and awareness assistant for India.
Your task is to explain retrieved legal evidence to a citizen in clear, simple language.

CRITICAL RULES:
1. Answer ONLY using the supplied retrieved legal evidence and qualification summary.
2. Do NOT invent laws, sections, penalties, authorities, URLs, or deadlines.
3. Do NOT determine guilt, innocence, or legal entitlement.
4. If evidence is insufficient, explicitly state: "The available CivicSync sources do not contain enough information to address this."
5. Do NOT pretend to be a lawyer or provide professional legal advice.
6. Preserve uncertainty when evidence is partial or ambiguous.
7. Never fabricate citations.
8. Use simple language an ordinary citizen can understand.
9. Do not mention technical implementation details (RAG, chunks, embeddings, Supabase, etc.).

REQUIRED RESPONSE STRUCTURE — follow this exactly:

SITUATION
Briefly state what you understood from the user's situation.

WHAT THE SOURCES SAY
Explain the relevant legal information found in the retrieved sources.

YOUR RIGHTS / PROTECTIONS
State rights explicitly supported by the retrieved evidence. If none: "None explicitly detailed in available sources."

WHAT YOU CAN DO
Practical next steps ONLY when directly supported by the evidence.

WHERE TO GO
Authorities or official channels explicitly mentioned in the evidence.

WHAT TO KEEP READY
Documents or evidence mentioned in the sources. If none: "None specified in available sources."

WHAT IS STILL UNCLEAR
Missing information that may affect the situation.

LEGAL BASIS
Acts/sections explicitly present in the evidence only.

SOURCES
List source files and page numbers referenced.

DISCLAIMER
This is legal information and awareness only, not professional legal advice. For personalized legal guidance, consult a qualified lawyer.
"""


def _format_qualification_context(
    qualification: Dict[str, Any],
    action_plan: Dict[str, Any],
) -> str:
    """Format qualification and action plan as readable context for the LLM prompt."""
    lines = ["==================================================",
             "STRUCTURED QUALIFICATION SUMMARY",
             "=================================================="]

    status = qualification.get("status", "unknown")
    lines.append(f"Evidence Status: {status.upper()}")
    lines.append(f"Domain: {qualification.get('domain', '?')} | Issue: {qualification.get('issue', '?')}")
    lines.append("")

    def _section(title: str, items: List[Dict]) -> None:
        if items:
            lines.append(f"{title}:")
            for item in items:
                ev_list = item.get("evidence", [])
                if ev_list and isinstance(ev_list, list):
                    first_ev = ev_list[0]
                    src = f"{first_ev.get('source_file', '?')} p.{first_ev.get('page_start', '?')}"
                else:
                    src = f"{item.get('source_file', '?')} p.{item.get('page_start', '?')}"
                lines.append(f"  • {item.get('text', '')} [{src}]")
            lines.append("")

    _section("APPLICABLE INFORMATION",  qualification.get("applicable_information", []))
    _section("RIGHTS / PROTECTIONS",    qualification.get("rights_or_protections", []))
    _section("POSSIBLE ACTIONS",        qualification.get("possible_actions", []))
    _section("AUTHORITIES / CHANNELS",  qualification.get("authorities_or_channels", []))
    _section("DOCUMENTS TO KEEP READY", qualification.get("documents_or_evidence", []))
    _section("CONDITIONS",              qualification.get("conditions", []))

    missing = qualification.get("missing_information", [])
    if missing:
        lines.append("MISSING INFORMATION:")
        for m in missing:
            lines.append(f"  • {m}")
        lines.append("")

    lines += ["==================================================",
              "ACTION PLAN",
              "=================================================="]

    def _action_section(title: str, items) -> None:
        if items:
            lines.append(f"{title}:")
            for item in items:
                text = item.get("text", item) if isinstance(item, dict) else item
                lines.append(f"  → {text}")
            lines.append("")

    if action_plan.get("immediate_steps"):
        _action_section("IMMEDIATE STEPS", action_plan["immediate_steps"])
    if action_plan.get("formal_remedies"):
        _action_section("FORMAL REMEDIES", action_plan["formal_remedies"])
    if action_plan.get("authority_pathway"):
        _action_section("AUTHORITY PATHWAY", action_plan["authority_pathway"])
    if action_plan.get("escalation"):
        _action_section("ESCALATION OPTIONS", action_plan["escalation"])

    info_to_collect = action_plan.get("information_to_collect", [])
    if info_to_collect:
        lines.append("INFORMATION TO COLLECT:")
        for info in info_to_collect:
            lines.append(f"  • {info}")
        lines.append("")

    return "\n".join(lines)


def _build_phase5_prompt(
    question: str,
    chunk_context: str,
    qual_context: str,
) -> str:
    return (
        f"{PHASE5_SYSTEM_PROMPT}\n\n"
        "==================================================\n"
        "RETRIEVED LEGAL EVIDENCE\n"
        "==================================================\n"
        f"{chunk_context}\n\n"
        f"{qual_context}\n\n"
        "==================================================\n"
        "USER SITUATION\n"
        "==================================================\n"
        f"{question}\n\n"
        "==================================================\n"
        "CIVICSYNC RESPONSE\n"
        "==================================================\n"
    )


def process_query(
    user_query: str,
    top_k: int = 5,
    client: Optional[Any] = None,
    model: Optional[str] = None,
    confidence_threshold: float = 0.45,
) -> Dict[str, Any]:
    """
    Process a user's natural language query through the complete CivicSync Engine:
      1. Situation Extraction
      2. Clarification check
      3. Supabase pgvector retrieval
      4. Legal qualification (Phase 5A)
      5. Action plan (Phase 5A)
      6. Enriched context → grounded RAG response
    """
    start_time = time.perf_counter()
    selected_model = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL_NAME)

    # Step 1: Extract situation metadata
    t0 = time.perf_counter()
    extracted_situation = extract_situation(user_query, model=selected_model)
    situation_ms = int((time.perf_counter() - t0) * 1000)

    domain     = extracted_situation.get("domain", "unknown")
    confidence = extracted_situation.get("confidence", 0.0)

    # Step 2: Clarification check
    if domain == "unknown" or confidence < confidence_threshold:
        total_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info(
            "Query complete (clarification) | situation_ms=%d | total_ms=%d",
            situation_ms, total_ms
        )
        return {
            "type":              "clarification_required",
            "message": (
                "The situation description is ambiguous or falls outside CivicSync's supported legal domains "
                "(traffic, labour, consumer, insurance, land_property, women_safety). "
                "Please provide more specific details about your situation."
            ),
            "situation":         extracted_situation,
            "qualification":     None,
            "action_plan":       None,
            "retrieved_sources": [],
            "similarity_scores": [],
            "rag_response":      None,
            "model_name":        selected_model,
        }

    # Step 3: Supabase pgvector retrieval
    t0 = time.perf_counter()
    chunks = retrieve(question=user_query, top_k=top_k, domain=domain, client=client)
    retrieval_ms = int((time.perf_counter() - t0) * 1000)

    similarity_scores = [c.get("similarity", 0.0) for c in (chunks or [])]
    retrieved_sources = [
        {
            "chunk_id":    c.get("chunk_id"),
            "domain":      c.get("domain"),
            "source_file": c.get("source_file"),
            "page_start":  c.get("page_start"),
            "page_end":    c.get("page_end"),
            "similarity":  c.get("similarity"),
        }
        for c in (chunks or [])
    ]

    # Fast local fallback if no chunks or extremely low relevance scores
    if not chunks:
        qualification_ms = 0
        llm_ms = 0
        total_ms = int((time.perf_counter() - start_time) * 1000)

        logger.info(
            "Query complete (insufficient) | situation_ms=%d | retrieval_ms=%d | qualification_ms=%d | llm_ms=%d | total_ms=%d",
            situation_ms, retrieval_ms, qualification_ms, llm_ms, total_ms
        )

        qualification = {
            "status": "insufficient_evidence",
            "domain": domain,
            "issue": extracted_situation.get("issue"),
            "applicable_information": [],
            "rights_or_protections": [],
            "possible_actions": [],
            "authorities_or_channels": [],
            "documents_or_evidence": [],
            "conditions": [],
            "missing_information": ["No relevant legal evidence retrieved."],
        }
        action_plan = {
            "immediate_steps": [],
            "formal_remedies": [],
            "authority_pathway": [],
            "escalation": [],
            "information_to_collect": [],
            "status": "insufficient_evidence",
        }

        return {
            "type":              "success",
            "situation":         extracted_situation,
            "qualification":     qualification,
            "action_plan":       action_plan,
            "rag_response":      "I don't have enough reliable information to answer that safely yet. Can you give me a little more information about your situation?",
            "retrieved_sources": [],
            "similarity_scores": [],
            "model_name":        selected_model,
        }

    # Step 4: Legal qualification
    t0 = time.perf_counter()
    qualification = qualify_case(
        situation=extracted_situation,
        retrieved_chunks=chunks or [],
        model=selected_model,
    )
    qualification_ms = int((time.perf_counter() - t0) * 1000)

    # Step 5: Action plan
    action_plan = build_action_plan(qualification)

    # Step 6: Build enriched context for RAG
    chunk_context = build_context(chunks or [])
    qual_context  = _format_qualification_context(qualification, action_plan)
    prompt        = _build_phase5_prompt(user_query, chunk_context, qual_context)

    # Step 7: Generate grounded response
    t0 = time.perf_counter()
    answer = generate(prompt=prompt, temperature=0.1, model=selected_model)
    llm_ms = int((time.perf_counter() - t0) * 1000)

    total_ms = int((time.perf_counter() - start_time) * 1000)

    logger.info(
        "Query complete | situation_ms=%d | retrieval_ms=%d | qualification_ms=%d | llm_ms=%d | total_ms=%d",
        situation_ms, retrieval_ms, qualification_ms, llm_ms, total_ms
    )

    return {
        "type":              "success",
        "situation":         extracted_situation,
        "qualification":     qualification,
        "action_plan":       action_plan,
        "rag_response":      answer,
        "retrieved_sources": retrieved_sources,
        "similarity_scores": similarity_scores,
        "model_name":        selected_model,
    }
