"""
test_retrieval.py
=================
CivicSync Phase 2C — Retrieval Quality Check

Runs 6 domain-specific queries through the pgvector retrieval pipeline
and prints ranked results with full content for manual evaluation.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.retriever import retrieve, get_retrieval_model, DEFAULT_MODEL

# ---------------------------------------------------------------------------
# Test queries — one per domain
# ---------------------------------------------------------------------------
TEST_QUERIES = [
    {
        "label":    "TRAFFIC",
        "domain":   "traffic",
        "question": "What happens if I drive without a valid driving licence?",
    },
    {
        "label":    "LABOUR",
        "domain":   "labour",
        "question": "What can I do if my employer has not paid my wages?",
    },
    {
        "label":    "CONSUMER",
        "domain":   "consumer",
        "question": "What are my rights if I purchased a defective product?",
    },
    {
        "label":    "LAND_PROPERTY",
        "domain":   "land_property",
        "question": "What are my rights as a tenant?",
    },
    {
        "label":    "WOMEN_SAFETY",
        "domain":   "women_safety",
        "question": "What legal protection is available in a situation involving harassment?",
    },
    {
        "label":    "INSURANCE",
        "domain":   "insurance",
        "question": "What can I do if my insurance claim is rejected?",
    },
]

W = 72  # line width


def sep(ch: str = "=") -> None:
    print(ch * W)


def run() -> None:
    # Warm-up model and report dimension
    print("Loading embedding model...")
    model = get_retrieval_model()
    dim_check = model.encode("query: test", normalize_embeddings=True)
    print(f"Model         : {DEFAULT_MODEL}")
    print(f"Query emb dim : {len(dim_check)}")
    sep()

    any_error = False

    for item in TEST_QUERIES:
        label    = item["label"]
        domain   = item["domain"]
        question = item["question"]

        print(f"\nQUERY  [{label}]")
        print(f"Domain  : {domain}")
        print(f"Question: {question}")
        sep("-")

        try:
            results = retrieve(question, top_k=5, domain=domain)
        except Exception as exc:
            print(f"[ERROR] {exc}")
            if "PGRST202" in str(exc) or "match_chunks" in str(exc):
                print()
                print("The match_chunks() function has not been created yet.")
                print("Please run sql/match_chunks.sql in your Supabase SQL Editor,")
                print("then re-run this script.")
            any_error = True
            sep()
            continue

        print(f"Results : {len(results)}")
        sep("-")

        for rank, res in enumerate(results, 1):
            sim         = res.get("similarity", 0.0)
            chunk_id    = res.get("chunk_id")
            source_file = res.get("source_file", "?")
            page_start  = res.get("page_start", "?")
            page_end    = res.get("page_end", "?")
            dom         = res.get("domain", "?")
            content     = (res.get("content") or "").strip()

            # Wrap content at 68 chars
            wrapped = []
            for i in range(0, len(content), 68):
                wrapped.append("  " + content[i:i+68])
            content_display = "\n".join(wrapped) if wrapped else "  (empty)"

            print(f"Rank        : {rank}")
            print(f"Similarity  : {sim:.6f}")
            print(f"Domain      : {dom}")
            print(f"Source File : {source_file}")
            print(f"Page        : {page_start}–{page_end}")
            print(f"Chunk ID    : {chunk_id}")
            print(f"Content     :")
            print(content_display[:600] + ("..." if len(content) > 600 else ""))
            print("." * W)

        sep()

    if any_error:
        print("\n[SUMMARY] One or more queries failed. See errors above.")
        sys.exit(1)
    else:
        print("\n[SUMMARY] All 6 retrieval queries completed successfully.")


if __name__ == "__main__":
    run()
