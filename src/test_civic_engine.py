"""
test_civic_engine.py
====================
CivicSync Phase 4 — Live Integration Test

Executes the 7 integration test queries through the full Civic Engine:
Natural Language -> Situation Extraction -> Domain Filtered Retrieval -> Grounded RAG Answer.
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

from src.civic_engine import process_query

TEST_QUERIES = [
    "Traffic police stopped me because I didn't have my driving licence.",
    "My company has not paid my salary for two months.",
    "I bought a phone and the seller refuses to replace it.",
    "My landlord won't return my security deposit.",
    "My insurer rejected my hospital claim.",
    "Someone is harassing me at my workplace.",
    "What can I do if someone is troubling me?",
]

WIDTH = 76


def print_sep(ch: str = "=") -> None:
    print(ch * WIDTH)


def run_integration_tests() -> None:
    print_sep("=")
    print("  CivicSync Phase 4 — Live Civic Engine Integration Verification")
    print_sep("=")

    for idx, user_query in enumerate(TEST_QUERIES, 1):
        print(f"\n[{idx}/7] USER QUERY:")
        print(f"\"{user_query}\"")
        print_sep("-")

        try:
            res = process_query(user_query)
            sit = res.get("situation", {})

            print("CIRCUMSTANCE INTELLIGENCE (SITUATION EXTRACTION):")
            print(f"  Domain       : {sit.get('domain')}")
            print(f"  Situation    : {sit.get('situation')}")
            print(f"  Issue        : {sit.get('issue')}")
            print(f"  Intent       : {sit.get('intent')}")
            print(f"  Jurisdiction : {sit.get('jurisdiction')}")
            print(f"  Entities     : {sit.get('entities')}")
            print(f"  Confidence   : {sit.get('confidence')}")

            print_sep("-")
            res_type = res.get("type")
            print(f"PIPELINE STATUS: {res_type.upper() if res_type else 'UNKNOWN'}")

            if res_type == "clarification_required":
                print(f"\nCLARIFICATION MESSAGE:\n{res.get('message')}")

            else:
                sources = res.get("retrieved_sources", [])
                print(f"\nRETRIEVED SOURCES ({len(sources)} chunks):")
                for s_idx, src in enumerate(sources, 1):
                    p_start = src.get("page_start")
                    p_end = src.get("page_end")
                    p_str = str(p_start) if p_start == p_end else f"{p_start}–{p_end}"
                    print(
                        f"  {s_idx}. Chunk ID {src.get('chunk_id')} | "
                        f"Sim: {src.get('similarity', 0.0):.4f} | "
                        f"Doc: {src.get('source_file')} (Page {p_str})"
                    )

                print("\nFINAL RAG ANSWER:")
                print_sep(".")
                print(res.get("rag_response"))
                print_sep(".")

        except Exception as exc:
            print(f"[ERROR] Execution failed: {exc}")
            import traceback
            traceback.print_exc()

        print_sep("=")


if __name__ == "__main__":
    run_integration_tests()
