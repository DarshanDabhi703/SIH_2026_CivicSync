"""
test_qualification.py
=====================
CivicSync Phase 5A — Live Qualification & Action Engine Verification

Runs the 6 domain queries through the full CivicSync engine with
evidence-first qualification, chunk quality report, and source traceability.
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
]

WIDTH = 76


def print_sep(ch: str = "=") -> None:
    print(ch * WIDTH)


def print_sourced_items(title: str, items: list) -> None:
    print(f"\n{title}:")
    if not items:
        print("  (None explicitly supported by retrieved evidence)")
        return

    for item in items:
        if isinstance(item, dict):
            text = item.get("text", "")
            ev_list = item.get("evidence", [])
            print(f"  • {text}")
            if ev_list:
                for ev in ev_list:
                    pg = f"p.{ev.get('page_start')}" if ev.get('page_start') == ev.get('page_end') else f"p.{ev.get('page_start')}–{ev.get('page_end')}"
                    print(f"    [Source: {ev.get('source_file')} | {pg} | Chunk ID: {ev.get('chunk_id')} | Sim: {ev.get('similarity', 0):.4f}]")
            else:
                src = f"{item.get('source_file', '?')} p.{item.get('page_start', '?')}"
                print(f"    [Source: {src} | Chunk ID: {item.get('chunk_id')}]")
        else:
            print(f"  • {item}")


def run() -> None:
    print_sep("=")
    print("  CivicSync Phase 5A — Live Qualification & Action Engine Verification")
    print_sep("=")

    for idx, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[{idx}/6] USER QUERY:\n  \"{query}\"")
        print_sep("-")

        try:
            res = process_query(query)
        except Exception as exc:
            print(f"[ERROR] {exc}")
            print_sep("=")
            continue

        sit  = res.get("situation", {})
        qual = res.get("qualification") or {}
        plan = res.get("action_plan") or {}

        print("SITUATION:")
        print(f"  Original Query : {query}")
        print(f"  Situation      : {sit.get('situation')}")

        print(f"DOMAIN  : {sit.get('domain')}")
        print(f"ISSUE   : {sit.get('issue')}")
        print(f"STATUS  : {qual.get('status', '?').upper()}")

        print_sourced_items("SUPPORTED INFORMATION", qual.get("applicable_information", []))
        print_sourced_items("RIGHTS / PROTECTIONS",   qual.get("rights_or_protections", []))
        print_sourced_items("POSSIBLE ACTIONS",       qual.get("possible_actions", []))
        print_sourced_items("AUTHORITIES / CHANNELS", qual.get("authorities_or_channels", []))

        missing = qual.get("missing_information", [])
        print("\nMISSING INFORMATION:")
        if missing:
            for m in missing:
                print(f"  • {m}")
        else:
            print("  (None)")

        sources = qual.get("sources", []) or res.get("retrieved_sources", [])
        print(f"\nSOURCES ({len(sources)} chunks):")
        for s in sources:
            p = str(s.get("page_start")) if s.get("page_start") == s.get("page_end") else f"{s.get('page_start')}–{s.get('page_end')}"
            print(f"  Chunk {s.get('chunk_id'):>4} | Sim {s.get('similarity', 0):.4f} | {s.get('source_file')} (Page {p})")

        print("\nFINAL GROUNDED RAG ANSWER:")
        print_sep(".")
        print(res.get("rag_response", "(none)"))
        print_sep(".")

        print_sep("=")

    print("\nPhase 5A Live Verification Complete.")


if __name__ == "__main__":
    run()
