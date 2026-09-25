"""
retriever.py
============
CivicSync Phase 2C — Supabase pgvector Retrieval

Embeds queries using the "query: <question>" prefix and matches them
using pgvector cosine similarity via the "match_chunks" RPC.
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

from supabase import create_client, Client
from src.embedder import load_model, DEFAULT_MODEL

# Cache the model to avoid reloading weights on every call
_model = None


def get_retrieval_model():
    """Lazy-load and cache the SentenceTransformer model."""
    global _model
    if _model is None:
        _model = load_model(DEFAULT_MODEL)
    return _model


def retrieve(
    question: str,
    top_k: int = 5,
    domain: Optional[str] = None,
    client: Optional[Client] = None,
) -> List[Dict[str, Any]]:
    """
    Generate query embedding and perform vector search in Supabase using the match_chunks RPC.

    Parameters
    ----------
    question : str
        The user's search query.
    top_k : int, optional
        Maximum number of matching chunks to return. Defaults to 5.
    domain : str, optional
        Optional domain filter (e.g. 'traffic', 'labour'). If None, searches all domains.
    client : Client, optional
        Supabase client instance. If None, initialized from environment variables.

    Returns
    -------
    list[dict]
        Ranked chunks with fields: chunk_id, content, source_file, page_start, page_end, domain, similarity.
    """
    if client is None:
        SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
        SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY", "").strip()
        if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
            raise RuntimeError("SUPABASE_URL or SUPABASE_SECRET_KEY is missing from environment/env")
        client = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)

    # 1. Embed query with "query: " prefix (E5 specification)
    model = get_retrieval_model()
    query_text = f"query: {question}"

    # Generate normalized embedding list of floats
    embedding = model.encode(
        query_text,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    # 2. Call pgvector similarity search RPC
    response = client.rpc(
        "match_chunks",
        {
            "query_embedding": embedding,
            "match_count": top_k,
            "filter_domain": domain,
        },
    ).execute()

    return response.data
