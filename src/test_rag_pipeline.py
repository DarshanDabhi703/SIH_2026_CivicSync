"""
test_rag_pipeline.py
====================
CivicSync Phase 3 — Real RAG Pipeline Verification

Executes the 6 domain queries + 1 unsupported query through the full
RAG pipeline (Supabase pgvector -> Context Builder -> Ollama / Llama 3.1 8B).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.rag import answer_question

# ---------------------------------------------------------------------------
# Test Suite: 6 Domain Queries + 1 Grounding Test (Unsupported)
# ---------------------------------------------------------------------------
TEST_QUERIES = [
    {
        "label": "TRAFFIC",
        "domain": "traffic",
        "question": "What happens if I drive without a valid driving licence?",
    },
    {
        "label": "LABOUR",
        "domain": "labour",
        "question": "What can I do if my employer has not paid my wages?",
    },
    {
        "label": "CONSUMER",
        "domain": "consumer",
        "question": "What are my rights if I purchased a defective product?",
    },
    {
        "label": "LAND_PROPERTY",
        "domain": "land_property",
        "question": "What are my rights as a tenant?",
    },
    {
        "label": "WOMEN_SAFETY",
        "domain": "women_safety",
        "question": "What legal protection is available in a situation involving harassment?",
    },
    {
        "label": "INSURANCE",
        "domain": "insurance",
        "question": "What can I do if my insurance claim is rejected?",
    },
    {
        "label": "GROUNDING_TEST (UNSUPPORTED)",
        "domain": None,
        "question": "What is the exact procedure for filing an income tax appeal?",
    },
]

WIDTH = 76


def print_sep(ch: str = "=") -> None:
    print(ch * WIDTH)


def run_pipeline_test() -> None:
    print_sep("=")
    print("  CivicSync Phase 3 — Live RAG Pipeline Verification")
    print_sep("=")

    for idx, item in enumerate(TEST_QUERIES, 1):
        label = item["label"]
        domain = item["domain"]
        question = item["question"]

        print(f"\n[{idx}/7] QUERY [{label}]")
        print(f"Question : {question}")
        print(f"Domain   : {domain if domain else 'ALL DOMAINS (unfiltered)'}")
        print_sep("-")

        try:
            result = answer_question(
                question=question,
                top_k=5,
                domain=domain,
            )

            print(f"Model Used : {result['model_name']}")
            print(f"Sources ({len(result['retrieved_sources'])} chunks retrieved):")
            for s_idx, src in enumerate(result['retrieved_sources'], 1):
                p_start = src.get('page_start')
                p_end = src.get('page_end')
                page_str = str(p_start) if p_start == p_end else f"{p_start}–{p_end}"
                print(
                    f"  {s_idx}. Chunk ID {src.get('chunk_id')} | "
                    f"Sim: {src.get('similarity', 0.0):.4f} | "
                    f"Doc: {src.get('source_file')} (Page {page_str})"
                )

            print("\nCIVICSYNC RESPONSE:")
            print_sep(".")
            print(result["answer"])
            print_sep(".")

        except Exception as exc:
            print(f"[ERROR] Query failed: {exc}")
            print_sep("-")

    print_sep("=")
    print("  Phase 3 Live RAG Pipeline Execution Completed")
    print_sep("=")


if __name__ == "__main__":
    run_pipeline_test()
