"""
backend/schemas.py
==================
CivicSync Phase 6A — API Request / Response Schemas

Pydantic v2 models for the FastAPI layer.
These are pure data-transfer objects — no business logic here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Incoming POST /api/query payload."""

    message: str = Field(
        ...,
        description="The citizen's natural-language question (required, max 2000 chars).",
        min_length=1,
        max_length=2000,
    )
    language: str = Field(
        default="auto",
        description="Desired response language.  Pass 'auto' to let the model decide.",
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional conversation ID to link follow-up questions.",
    )

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message cannot be empty or whitespace only")
        return v.strip()


# ---------------------------------------------------------------------------
# Nested response types
# ---------------------------------------------------------------------------

class EvidenceSource(BaseModel):
    """Source traceability record for a single retrieved chunk."""
    chunk_id:    Optional[Any]  = None
    domain:      Optional[str]  = None
    source_file: Optional[str]  = None
    page_start:  Optional[int]  = None
    page_end:    Optional[int]  = None
    similarity:  Optional[float] = None


class SituationInfo(BaseModel):
    """Extracted situation metadata from Circumstance Intelligence."""
    domain:     Optional[str]        = None
    issue:      Optional[str]        = None
    intent:     Optional[str]        = None
    confidence: Optional[float]      = None
    entities:   Optional[List[str]]  = None
    situation:  Optional[str]        = None


class QualificationInfo(BaseModel):
    """
    Legal qualification summary.
    Preserves the evidence-first structure from Phase 5A qualification engine.
    """
    status:                  Optional[str]        = None
    domain:                  Optional[str]        = None
    issue:                   Optional[str]        = None
    applicable_information:  Optional[List[Any]]  = None
    rights_or_protections:   Optional[List[Any]]  = None
    possible_actions:        Optional[List[Any]]  = None
    authorities_or_channels: Optional[List[Any]]  = None
    documents_or_evidence:   Optional[List[Any]]  = None
    conditions:              Optional[List[Any]]  = None
    missing_information:     Optional[List[str]]  = None


class ActionInfo(BaseModel):
    """Structured action plan from Phase 5A action engine."""
    immediate_steps:        Optional[List[Any]]  = None
    formal_remedies:        Optional[List[Any]]  = None
    authority_pathway:      Optional[List[Any]]  = None
    escalation:             Optional[List[Any]]  = None
    information_to_collect: Optional[List[str]]  = None


# ---------------------------------------------------------------------------
# Top-level response
# ---------------------------------------------------------------------------

class QueryResponse(BaseModel):
    """
    POST /api/query response body.

    Fields
    ------
    status
        'success' | 'clarification_required' | 'error'
    situation
        Extracted domain / issue / intent / confidence.
    qualification
        Evidence-first legal qualification summary.
    actions
        Structured action plan with evidence-backed steps.
    answer
        Grounded RAG narrative response from Ollama.
    sources
        Chunk traceability list (source file + page + similarity).
    limitations
        Any missing-information notices from qualification.
    """
    status:        str                           = "success"
    situation:     Optional[SituationInfo]       = None
    qualification: Optional[QualificationInfo]   = None
    actions:       Optional[ActionInfo]          = None
    answer:        Optional[str]                 = None
    sources:       List[EvidenceSource]          = Field(default_factory=list)
    limitations:   List[str]                     = Field(default_factory=list)
    conversation_id: Optional[str]               = None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status:  str = "ok"
    service: str = "civicsync-api"
    llm_provider: str = "groq"
