"""
backend/main.py
===============
CivicSync Phase 6A — FastAPI Application

Exposes the full CivicSync Phase 5A intelligence pipeline via HTTP.
Does NOT duplicate any business logic — all intelligence is delegated
to src.civic_engine.process_query().

Endpoints
---------
GET  /api/health      -- Liveness probe
POST /api/query       -- Full CivicSync RAG pipeline
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Project path setup — must happen before importing src.*
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from backend.schemas import (
    ActionInfo,
    EvidenceSource,
    HealthResponse,
    QualificationInfo,
    QueryRequest,
    QueryResponse,
    SituationInfo,
)

# Import process_query at module level so unit tests can patch it reliably.
# The src.* modules are imported here — ML model weights are loaded lazily
# inside the retriever on first actual query, NOT at server startup.
from src.civic_engine import process_query
from src.llm_gateway import generate

# In-memory store for conversation history.
# Schema: {conversation_id: [{"role": "user"|"assistant", "content": str}]}
conversations: Dict[str, List[Dict[str, str]]] = {}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("civicsync.api")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CivicSync API",
    description=(
        "Legal information and awareness API for Indian citizens. "
        "Powered by IRDAI / NLSA / statutory regulatory documents, "
        "pgvector retrieval, and grounded Ollama RAG."
    ),
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — permissive for development; tighten origins before production
# ---------------------------------------------------------------------------
CORS_ORIGINS: list[str] = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global exception handler — never leak stack traces or credentials
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception on %s: %s", request.url.path, repr(exc))
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status":  "error",
            "message": "An internal error occurred. Please try again later.",
            "answer":  None,
        },
    )


# ---------------------------------------------------------------------------
# Response builder — maps civic_engine output → QueryResponse schema
# ---------------------------------------------------------------------------
def _build_response(engine_result: Dict[str, Any], conversation_id: Optional[str] = None) -> QueryResponse:
    """
    Convert the dict returned by process_query() into the QueryResponse schema.
    Handles both 'success' and 'clarification_required' engine outputs.
    Never propagates internal implementation details.
    """
    result_type = engine_result.get("type", "success")

    # Situation
    raw_sit = engine_result.get("situation") or {}
    situation = SituationInfo(
        domain=raw_sit.get("domain"),
        issue=raw_sit.get("issue"),
        intent=raw_sit.get("intent"),
        confidence=raw_sit.get("confidence"),
        entities=raw_sit.get("entities"),
        situation=raw_sit.get("situation"),
    ) if raw_sit else None

    # Qualification
    raw_qual = engine_result.get("qualification") or {}
    qualification = QualificationInfo(
        status=raw_qual.get("status"),
        domain=raw_qual.get("domain"),
        issue=raw_qual.get("issue"),
        applicable_information=raw_qual.get("applicable_information"),
        rights_or_protections=raw_qual.get("rights_or_protections"),
        possible_actions=raw_qual.get("possible_actions"),
        authorities_or_channels=raw_qual.get("authorities_or_channels"),
        documents_or_evidence=raw_qual.get("documents_or_evidence"),
        conditions=raw_qual.get("conditions"),
        missing_information=raw_qual.get("missing_information"),
    ) if raw_qual else None

    # Action plan
    raw_ap = engine_result.get("action_plan") or {}
    actions = ActionInfo(
        immediate_steps=raw_ap.get("immediate_steps"),
        formal_remedies=raw_ap.get("formal_remedies"),
        authority_pathway=raw_ap.get("authority_pathway"),
        escalation=raw_ap.get("escalation"),
        information_to_collect=raw_ap.get("information_to_collect"),
    ) if raw_ap else None

    # Sources — preserve full traceability
    raw_sources = engine_result.get("retrieved_sources") or []
    sources = [
        EvidenceSource(
            chunk_id=s.get("chunk_id"),
            domain=s.get("domain"),
            source_file=s.get("source_file"),
            page_start=s.get("page_start"),
            page_end=s.get("page_end"),
            similarity=s.get("similarity"),
        )
        for s in raw_sources
    ]

    # Limitations — missing_information from qualification + clarification message
    limitations: list[str] = []
    if result_type == "clarification_required":
        msg = engine_result.get("message", "")
        if msg:
            limitations.append(msg)
    elif raw_qual:
        limitations = [
            str(m) for m in (raw_qual.get("missing_information") or []) if m
        ]

    # RAG answer
    answer = engine_result.get("rag_response") or engine_result.get("message")

    # Friendly insufficient evidence handling
    if (
        result_type == "clarification_required" or
        (qualification and qualification.status == "insufficient_evidence") or
        (answer and "do not contain enough information" in answer)
    ):
        if result_type == "clarification_required":
            answer = engine_result.get("message")
        else:
            answer = (
                "I don't have enough reliable information to answer that safely yet. "
                "Can you give me a little more information about your situation?"
            )

    # Map engine type to API status
    api_status = {
        "success":                "success",
        "clarification_required": "clarification_required",
    }.get(result_type, "success")

    return QueryResponse(
        status=api_status,
        situation=situation,
        qualification=qualification,
        actions=actions,
        answer=answer,
        sources=sources,
        limitations=limitations,
        conversation_id=conversation_id,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    tags=["Health"],
)
async def health() -> HealthResponse:
    """Returns 200 when the API is running."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()
    return HealthResponse(status="ok", service="civicsync-api", llm_provider=provider)


@app.post(
    "/api/query",
    response_model=QueryResponse,
    summary="Submit a legal query",
    tags=["Query"],
    status_code=status.HTTP_200_OK,
)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Process a citizen's natural-language legal query through the full
    CivicSync Phase 5A pipeline:

      Situation Extraction -> Retrieval -> Qualification -> Action Engine -> RAG

    The endpoint delegates entirely to ``src.civic_engine.process_query()``.
    Supports conversational memory and query reformulation for follow-up questions.
    """
    logger.info("Query received | language=%s | message_len=%d | conversation_id=%s", 
                request.language, len(request.message), request.conversation_id)

    try:
        conv_id = request.conversation_id
        if not conv_id:
            conv_id = uuid.uuid4().hex

        history = conversations.get(conv_id, [])
        user_message = request.message

        # Determine standalone query
        if history:
            # We have a conversation history, reformulate the query
            turns = []
            for turn in history:
                role = "User" if turn["role"] == "user" else "Assistant"
                turns.append(f"{role}: {turn['content']}")
            history_str = "\n".join(turns)

            prompt = (
                "You are a conversation-to-query translation assistant.\n"
                "Given the conversation history between a citizen and an assistant, "
                "translate the citizen's latest message into a standalone legal search query "
                "that incorporates the necessary context from the history.\n\n"
                "CRITICAL RULES:\n"
                "1. Do not answer the question.\n"
                "2. Just output the standalone query, nothing else. No preamble, no explanation.\n"
                "3. If the latest message is already a standalone question, return it as is.\n\n"
                f"Conversation History:\n{history_str}\n\n"
                f"Latest Message: {user_message}\n\n"
                "Standalone Query:"
            )
            reformulated_query = generate(prompt=prompt, temperature=0.0)
            logger.info("Query reformulated | original='%s' | reformulated='%s'", user_message, reformulated_query)
            query_to_run = reformulated_query
        else:
            query_to_run = user_message

        # Delegate to existing Civic Engine
        engine_result = process_query(
            user_query=query_to_run,
            top_k=5,
        )
        response = _build_response(engine_result, conversation_id=conv_id)

        # Update history
        if conv_id not in conversations:
            conversations[conv_id] = []
        conversations[conv_id].append({"role": "user", "content": user_message})
        conversations[conv_id].append({"role": "assistant", "content": response.answer or ""})

        logger.info(
            "Query complete | domain=%s | status=%s | sources=%d",
            response.situation.domain if response.situation else "unknown",
            response.status,
            len(response.sources),
        )
        return response

    except Exception as exc:
        # Log full detail server-side; return safe generic message to client
        logger.exception("process_query failed: %s", repr(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status":  "error",
                "message": "CivicSync encountered an internal error processing your query. Please try again.",
                "answer":  None,
                "sources": [],
                "limitations": [],
                "conversation_id": request.conversation_id,
            },
        )
