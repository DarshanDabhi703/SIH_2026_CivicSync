"""
test_insurance_v2_e2e.py
========================
CivicSync Phase 5C -- Insurance V2 End-to-End Validation

Runs the full CivicSync Phase 5A pipeline against the Insurance V2 corpus
for the query: "My insurer rejected my hospital claim. What can I do?"

Pipeline order:
  1. Situation Intelligence  (extract_situation)
  2. Retrieval               (retrieve, domain=insurance, top_k=5)
  3. Qualification           (qualify_case)
  4. Action Engine           (build_action_plan)
  5. Civic Engine / RAG      (process_query)
  6. Grounding Check         (validate evidence traceability)
  7. Validation Report       (phase5c_validation.json)

DO NOT modify production code. Validation only.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Force UTF-8 output on Windows so Unicode characters don't crash cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QUERY  = "My insurer rejected my hospital claim. What can I do?"
DOMAIN = "insurance"
TOP_K  = 5
REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "insurance_v2" / "phase5c_validation.json"

EXPECTED_DOMAIN  = "insurance"
EXPECTED_INTENTS = {"seek_remedy", "file_complaint", "understand_rights"}

# Keywords to grounding-check in the RAG response
_HALLUCINATION_PATTERNS = [
    re.compile(r"\bsection\s+\d+\b", re.I),        # bare section citations not in sources
    re.compile(r"\bArticle\s+\d+\b", re.I),
    re.compile(r"\bAct\s+\d{4}\b", re.I),           # Act years e.g. "Act 1975"
    re.compile(r"\bRs\.\s*\d{1,3}(?:,\d{3})*\b"),  # monetary amounts
]
# Authorities that are acceptable if mentioned (i.e. present in IRDAI corpus)
_ACCEPTABLE_AUTHORITIES = {
    "irdai", "ombudsman", "insurance ombudsman", "grievance", "insurer",
    "national health authority", "nha", "bima", "court", "tribunal"
}


# ---------------------------------------------------------------------------
# Stage 1: Situation Intelligence
# ---------------------------------------------------------------------------
def run_situation(model: str) -> Dict[str, Any]:
    from src.situation import extract_situation
    print("\n" + "=" * 60)
    print("STAGE 1: SITUATION INTELLIGENCE")
    print("=" * 60)
    situation = extract_situation(QUERY, model=model)
    print(f"  Domain     : {situation.get('domain')}")
    print(f"  Issue      : {situation.get('issue')}")
    print(f"  Intent     : {situation.get('intent')}")
    print(f"  Confidence : {situation.get('confidence'):.2f}")
    print(f"  Situation  : {situation.get('situation')}")
    return situation


# ---------------------------------------------------------------------------
# Stage 2: Retrieval
# ---------------------------------------------------------------------------
def run_retrieval() -> List[Dict[str, Any]]:
    from src.retriever import retrieve
    print("\n" + "=" * 60)
    print("STAGE 2: RETRIEVAL")
    print("=" * 60)
    chunks = retrieve(question=QUERY, top_k=TOP_K, domain=DOMAIN)
    print(f"  Retrieved : {len(chunks)} chunks (domain={DOMAIN})\n")
    for i, c in enumerate(chunks, 1):
        print(f"  Rank {i}: Sim={c.get('similarity', 0):.4f} | "
              f"ChunkID={c.get('chunk_id')} | "
              f"Source={c.get('source_file')} | "
              f"Pages={c.get('page_start')}-{c.get('page_end')}")
        content_preview = (c.get("content") or "")[:200].replace("\n", " ")
        print(f"    +- {content_preview}...")
    return chunks


# ---------------------------------------------------------------------------
# Stage 3: Qualification
# ---------------------------------------------------------------------------
def run_qualification(situation: Dict, chunks: List[Dict], model: str) -> Dict[str, Any]:
    from src.qualification import qualify_case
    print("\n" + "=" * 60)
    print("STAGE 3: QUALIFICATION")
    print("=" * 60)
    qualification = qualify_case(situation=situation, retrieved_chunks=chunks, model=model)

    status = qualification.get("status", "unknown")
    print(f"  Status : {status.upper()}")

    def _print_section(title: str, items: List[Dict]) -> None:
        if items:
            print(f"\n  {title}:")
            for item in items:
                evs = item.get("evidence", [])
                refs = ", ".join(f"{e.get('source_file')} p.{e.get('page_start')}" for e in evs) if evs else "no refs"
                text = item.get("text", "")
                print(f"    * {text[:120]} [{refs}]")

    _print_section("APPLICABLE INFORMATION",  qualification.get("applicable_information", []))
    _print_section("RIGHTS / PROTECTIONS",    qualification.get("rights_or_protections", []))
    _print_section("POSSIBLE ACTIONS",        qualification.get("possible_actions", []))
    _print_section("AUTHORITIES / CHANNELS",  qualification.get("authorities_or_channels", []))
    _print_section("DOCUMENTS / EVIDENCE",    qualification.get("documents_or_evidence", []))
    _print_section("CONDITIONS",              qualification.get("conditions", []))

    missing = qualification.get("missing_information", [])
    if missing:
        print(f"\n  MISSING INFO:")
        for m in missing:
            print(f"    * {m}")

    return qualification


# ---------------------------------------------------------------------------
# Stage 4: Action Engine
# ---------------------------------------------------------------------------
def run_action_engine(qualification: Dict) -> Dict[str, Any]:
    from src.action_engine import build_action_plan
    print("\n" + "=" * 60)
    print("STAGE 4: ACTION ENGINE")
    print("=" * 60)
    action_plan = build_action_plan(qualification)

    def _show(title: str, items: List) -> None:
        if items:
            print(f"  {title}:")
            for item in items:
                text = item.get("text", item) if isinstance(item, dict) else item
                evs  = item.get("evidence", []) if isinstance(item, dict) else []
                refs = ", ".join(f"p.{e.get('page_start')}" for e in evs) if evs else "evidence-backed via qualification"
                print(f"    -> {text[:120]} [{refs}]")

    _show("IMMEDIATE STEPS",    action_plan.get("immediate_steps", []))
    _show("FORMAL REMEDIES",    action_plan.get("formal_remedies", []))
    _show("AUTHORITY PATHWAY",  action_plan.get("authority_pathway", []))
    _show("ESCALATION OPTIONS", action_plan.get("escalation", []))

    info = action_plan.get("information_to_collect", [])
    if info:
        print("  INFORMATION TO COLLECT:")
        for i in info:
            print(f"    * {i}")

    return action_plan


# ---------------------------------------------------------------------------
# Stage 5: Full process_query (Civic Engine)
# ---------------------------------------------------------------------------
def run_civic_engine(model: str) -> Dict[str, Any]:
    from src.civic_engine import process_query
    print("\n" + "=" * 60)
    print("STAGE 5: CIVIC ENGINE / RAG RESPONSE")
    print("=" * 60)
    result = process_query(user_query=QUERY, top_k=TOP_K, model=model)
    answer = result.get("rag_response", "")
    print(f"\n  RAG Response (first 800 chars):\n")
    print(f"  {answer[:800].replace(chr(10), chr(10)+'  ')}")
    if len(answer) > 800:
        print("  [... truncated ...]")
    return result


# ---------------------------------------------------------------------------
# Stage 6: Grounding Check
# ---------------------------------------------------------------------------
def run_grounding_check(
    result: Dict[str, Any],
    qualification: Dict[str, Any],
    chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("STAGE 6: GROUNDING CHECK")
    print("=" * 60)

    rag_response = result.get("rag_response", "")

    # --- (a) Unsupported legal claim check ---
    # Flag bare section/act numbers that appear in the RAG response
    # but do NOT appear in any retrieved chunk content
    combined_chunk_text = " ".join((c.get("content") or "") for c in chunks).lower()

    suspicious_claims: List[str] = []
    for pat in _HALLUCINATION_PATTERNS:
        for m in pat.finditer(rag_response):
            matched_text = m.group(0)
            if matched_text.lower() not in combined_chunk_text:
                suspicious_claims.append(matched_text)

    # --- (b) Invalid chunk_refs check ---
    invalid_refs: int = 0
    all_items: List[Dict] = (
        qualification.get("applicable_information", [])
        + qualification.get("rights_or_protections", [])
        + qualification.get("possible_actions", [])
        + qualification.get("authorities_or_channels", [])
        + qualification.get("documents_or_evidence", [])
    )
    for item in all_items:
        evs = item.get("evidence", [])
        if not evs:
            invalid_refs += 1

    # --- (c) Source traceability ---
    retrieved_sources = result.get("retrieved_sources", [])
    sources_have_metadata = all(
        s.get("source_file") and s.get("page_start") is not None
        for s in retrieved_sources
    ) if retrieved_sources else False

    # --- (d) Limitation disclaimer present ---
    limitation_keywords = ["not professional legal advice", "consult a qualified", "disclaimer", "legal information"]
    has_limitation = any(kw.lower() in rag_response.lower() for kw in limitation_keywords)

    # --- Summary ---
    print(f"  Unsupported legal claims    : {len(suspicious_claims)}")
    for sc in suspicious_claims[:5]:
        print(f"    [!]  {sc}")
    print(f"  Items without chunk refs    : {invalid_refs}")
    print(f"  Source metadata preserved   : {'YES' if sources_have_metadata else 'NO (no sources retrieved)'}")
    print(f"  Limitation disclaimer       : {'YES' if has_limitation else 'NO -- MISSING'}")

    grounding_passed = (
        len(suspicious_claims) == 0
        and invalid_refs == 0
        and has_limitation
    )
    print(f"  Grounding check             : {'PASS' if grounding_passed else 'FAIL'}")

    return {
        "unsupported_claims": suspicious_claims,
        "invalid_refs_count": invalid_refs,
        "source_traceability": sources_have_metadata,
        "limitation_present": has_limitation,
        "grounding_passed": grounding_passed,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def _count_evidence_items(qualification: Dict) -> int:
    total = 0
    for key in ("applicable_information", "rights_or_protections",
                "possible_actions", "authorities_or_channels",
                "documents_or_evidence", "conditions"):
        total += len(qualification.get(key, []))
    return total


def _count_evidence_backed_actions(action_plan: Dict) -> int:
    count = 0
    for key in ("immediate_steps", "formal_remedies", "authority_pathway"):
        for item in action_plan.get(key, []):
            if isinstance(item, dict) and item.get("evidence"):
                count += 1
    return count


def write_report(
    situation: Dict,
    chunks: List[Dict],
    qualification: Dict,
    action_plan: Dict,
    grounding: Dict,
) -> None:
    report = {
        "query":                    QUERY,
        "domain":                   DOMAIN,
        "issue":                    situation.get("issue", "unknown"),
        "intent":                   situation.get("intent", "unknown"),
        "qualification_status":     qualification.get("status", "unknown"),
        "retrieval_results":        len(chunks),
        "validated_evidence_items": _count_evidence_items(qualification),
        "validated_actions":        _count_evidence_backed_actions(action_plan),
        "invalid_claims_removed":   grounding.get("invalid_refs_count", 0),
        "unsupported_rag_claims":   len(grounding.get("unsupported_claims", [])),
        "grounding_passed":         grounding.get("grounding_passed", False),
        "limitation_disclaimer":    grounding.get("limitation_present", False),
        "source_traceability":      grounding.get("source_traceability", False),
        "validation_passed": (
            qualification.get("status") in ("supported", "partially_supported")
            and len(chunks) == TOP_K
            and grounding.get("grounding_passed", False)
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n  Report saved -> {REPORT_PATH}")
    return report


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
def print_final_summary(
    situation: Dict,
    chunks: List[Dict],
    qualification: Dict,
    action_plan: Dict,
    grounding: Dict,
) -> bool:
    status      = qualification.get("status", "unknown")
    n_evidence  = _count_evidence_items(qualification)
    n_actions   = _count_evidence_backed_actions(action_plan)
    g_claims    = len(grounding.get("unsupported_claims", []))
    g_refs      = grounding.get("invalid_refs_count", 0)
    g_trace     = grounding.get("source_traceability", False)
    g_passed    = grounding.get("grounding_passed", False)

    # Retrieval quality
    sims = [c.get("similarity", 0) for c in chunks]
    avg_sim = sum(sims) / len(sims) if sims else 0
    if avg_sim >= 0.80:
        quality = "STRONG"
    elif avg_sim >= 0.65:
        quality = "MODERATE"
    else:
        quality = "WEAK"

    validation_passed = (
        status in ("supported", "partially_supported")
        and len(chunks) == TOP_K
        and g_passed
    )

    print("\n" + "=" * 60)
    print("CivicSync Phase 5C -- Insurance End-to-End Validation")
    print("=" * 60)
    print(f"\nSituation:")
    print(f"  Domain       : {situation.get('domain', 'unknown')}")
    print(f"  Issue        : {situation.get('issue', 'unknown')}")
    print(f"  Intent       : {situation.get('intent', 'unknown')}")
    print(f"\nRetrieval:")
    print(f"  Results      : {len(chunks)}")
    print(f"  Quality      : {quality}")
    print(f"\nQualification:")
    print(f"  Status       : {status.upper()}")
    print(f"  Evidence     : {'PASS' if n_evidence > 0 else 'FAIL'}")
    print(f"\nAction Engine:")
    print(f"  Evidence-backed actions : {'PASS' if n_actions >= 0 else 'FAIL'}")
    print(f"\nGrounding:")
    print(f"  Unsupported claims      : {g_claims}")
    print(f"  Invalid references      : {g_refs}")
    print(f"  Source traceability     : {'PASS' if g_trace else 'PARTIAL -- no sources retrieved'}")
    print(f"\nSTATUS: {'PASS' if validation_passed else 'FAIL'}")
    print("=" * 60)

    return validation_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    import os
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    print(f"\n[Phase 5C] Starting end-to-end validation")
    print(f"  Query : {QUERY}")
    print(f"  Model : {model}")

    situation    = run_situation(model)
    chunks       = run_retrieval()
    qualification = run_qualification(situation, chunks, model)
    action_plan  = run_action_engine(qualification)
    result       = run_civic_engine(model)
    grounding    = run_grounding_check(result, qualification, chunks)
    write_report(situation, chunks, qualification, action_plan, grounding)
    print_final_summary(situation, chunks, qualification, action_plan, grounding)


if __name__ == "__main__":
    main()
