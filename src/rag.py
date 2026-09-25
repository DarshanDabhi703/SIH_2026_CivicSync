"""
rag.py
======
CivicSync Phase 3 — Grounded RAG Pipeline

Orchestrates retrieval, context construction, and LLM answer generation
with strict anti-hallucination and evidence-grounding constraints.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.retriever import retrieve
from src.context_builder import build_context
from src.llm_client import generate, DEFAULT_MODEL_NAME

# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are CivicSync, a legal information and awareness assistant for India.
Your job is to explain retrieved government and legal information in simple, clear language for ordinary citizens.

IMPORTANT GROUNDING & COMPLIANCE RULES:
1. Answer ONLY using the supplied retrieved legal evidence.
2. Do NOT invent laws, section numbers, penalties, legal procedures, government portals, authorities, deadlines, or rights.
3. If the retrieved evidence does not contain enough information to answer the question, explicitly state: "The available CivicSync sources do not contain enough information to answer this question."
4. Do NOT pretend to be a lawyer or offer professional legal representation.
5. Provide legal information and public awareness, not personalized professional legal advice.
6. Preserve uncertainty whenever the evidence is partial or ambiguous.
7. Never fabricate a citation, section, or source document.
8. Every important legal claim must be traceable to the supplied source material.
9. Use simple language that an ordinary citizen can understand.
10. Do not mention internal technical implementation details (like RAG, chunks, embeddings, Supabase, pgvector) unless explicitly asked.

REQUIRED RESPONSE STRUCTURE:

SITUATION
Briefly state what you understood from the user's question.

LEGAL INFORMATION
Explain the relevant legal information found in the retrieved sources.

YOUR RIGHTS / PROTECTIONS
State rights explicitly supported by the retrieved evidence. (If none mentioned in evidence, state "None explicitly detailed in available sources.")

WHAT YOU CAN DO
Give practical next steps ONLY when directly supported by the evidence.

LEGAL BASIS
List the specific law/Act/section ONLY if explicitly present in the evidence.

SOURCES
List the source files and page numbers referenced for the claims made.

LIMITATION
If the evidence is incomplete or does not fully answer the question, clearly state the limitation.
"""


def build_rag_prompt(question: str, context: str) -> str:
    """Combine system instructions, retrieved evidence context, and user question into a grounded prompt."""
    return f"""{SYSTEM_PROMPT}

==================================================
RETRIEVED LEGAL EVIDENCE
==================================================
{context}

==================================================
USER QUESTION
==================================================
{question}

==================================================
CIVICSYNC GROUNDED RESPONSE
==================================================
"""


def answer_question(
    question: str,
    top_k: int = 5,
    domain: Optional[str] = None,
    client: Optional[Any] = None,
    model: Optional[str] = None,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """
    Complete CivicSync RAG Pipeline.

    Parameters
    ----------
    question : str
        User's legal query.
    top_k : int, optional
        Number of top chunks to retrieve (default 5).
    domain : str, optional
        Domain filter (e.g. 'traffic', 'labour'). If None, searches all domains.
    client : Any, optional
        Supabase client instance (passed to retriever).
    model : str, optional
        Ollama LLM model tag.
    temperature : float, optional
        Generation temperature (default 0.1).

    Returns
    -------
    dict
        {
            "answer": str,
            "retrieved_sources": list[dict],
            "similarity_scores": list[float],
            "model_name": str,
            "question": str
        }
    """
    selected_model = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL_NAME)

    # 1. Retrieve relevant chunks using existing retriever
    chunks = retrieve(question=question, top_k=top_k, domain=domain, client=client)

    # Extract source metadata and scores
    similarity_scores = [c.get("similarity", 0.0) for c in (chunks or [])]
    retrieved_sources = [
        {
            "chunk_id": c.get("chunk_id"),
            "domain": c.get("domain"),
            "source_file": c.get("source_file"),
            "page_start": c.get("page_start"),
            "page_end": c.get("page_end"),
            "similarity": c.get("similarity"),
        }
        for c in (chunks or [])
    ]

    # 2. Build structured context
    context = build_context(chunks or [])

    # 3. Construct grounded prompt
    prompt = build_rag_prompt(question, context)

    # 4. Generate response from Ollama LLM
    answer_text = generate(
        prompt=prompt,
        temperature=temperature,
        model=selected_model,
    )

    return {
        "question": question,
        "answer": answer_text,
        "retrieved_sources": retrieved_sources,
        "similarity_scores": similarity_scores,
        "model_name": selected_model,
    }
