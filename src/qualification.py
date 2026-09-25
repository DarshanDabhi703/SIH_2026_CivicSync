"""
qualification.py
================
CivicSync Phase 5A — Refined Legal Qualification Engine

Evidence-first approach:
  retrieved chunks → identify supported statements → attach chunk references
  → discard unsupported statements → calculate status → build action plan

Status is determined by evidence traceability, NOT similarity thresholds.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from src.llm_gateway import generate
from src.llm_client import OllamaClientError


# ---------------------------------------------------------------------------
# Status Constants
# ---------------------------------------------------------------------------
STATUS_SUPPORTED           = "supported"
STATUS_PARTIALLY_SUPPORTED = "partially_supported"
STATUS_INSUFFICIENT        = "insufficient_evidence"

# ---------------------------------------------------------------------------
# Metadata pattern detection
# ---------------------------------------------------------------------------
# Chunks that consist primarily of structural dataset metadata rather than
# substantive legal prose. A chunk is flagged as "metadata-dominant" when
# it matches more than one of these patterns at high frequency.
_METADATA_PATTERNS = [
    re.compile(r"^(category|domain|metadata|statute|section|source_url|classification|jurisdiction)\s*$", re.I | re.M),
    re.compile(r"^(support service|legal.?awareness|police.?procedure|domestic.?violence|workplace.?harassment)\s*$", re.I | re.M),
    re.compile(r"^\d+\.\d+Official government service source \d{4}", re.I | re.M),
    re.compile(r"^(WR-SUPPORT-|IN-[A-Z]{2})", re.M),
    re.compile(r"^Q:\s+I live in .+\. My insurer rejected", re.M),
    re.compile(r"^Q:\s+Can everyone in .+ use the state health scheme", re.M),
    re.compile(r"^This is legal.information content, not individualized legal advice\.\s*$", re.I | re.M),
    re.compile(r"^(attendance manipulation|labour rights general|leave entitlement)\s+(authority|helpline|lawyer|NGO|welfare|protection|for women|for men|for children|for elderly|for disabled|for migrant|for contract|for permanent|online complaint|offline complaint|government scheme|free legal help|legal aid)\s*$", re.I | re.M),
]

_SUBSTANTIVE_PATTERNS = [
    # Contains a verb-predicate clause (real sentence structure)
    re.compile(r"\b(shall|must|may|can|is entitled|has the right|is required|is prohibited|is liable|is punishable|is allowed|is entitled)\b", re.I),
    # References to legislation with section numbers
    re.compile(r"\bsection\s+\d+", re.I),
    re.compile(r"\b(Act|Rule|Code|Regulation|Order)\s*[,\s]+\d{4}\b", re.I),
    # Penalty or legal consequence language
    re.compile(r"\b(fine|penalty|imprisonment|punishable|liable|compensation|damages|relief|remedy)\b", re.I),
    # Procedural legal language
    re.compile(r"\b(complaint|grievance|tribunal|court|commissioner|ombudsman|authority|forum|appeal|FIR)\b", re.I),
]


def _score_chunk(content: str) -> Dict[str, Any]:
    """
    Score a chunk's content quality:
    - metadata_hits: count of metadata pattern matches
    - substantive_hits: count of substantive pattern matches
    - is_metadata_dominant: True if the chunk is primarily structural metadata
    - is_substantive: True if the chunk contains substantive legal content
    """
    content = (content or "").strip()
    lines   = [l.strip() for l in content.splitlines() if l.strip()]
    n_lines = max(len(lines), 1)

    meta_hits  = sum(1 for p in _METADATA_PATTERNS     if p.search(content))
    subst_hits = sum(1 for p in _SUBSTANTIVE_PATTERNS   if p.search(content))

    # A chunk is metadata-dominant when it has many metadata pattern hits
    # AND very few substantive sentences.
    meta_line_count = sum(
        1 for l in lines
        if any(p.match(l) for p in _METADATA_PATTERNS)
    )
    meta_ratio = meta_line_count / n_lines

    return {
        "metadata_hits":       meta_hits,
        "substantive_hits":    subst_hits,
        "meta_ratio":          meta_ratio,
        "is_metadata_dominant": meta_ratio > 0.45 and subst_hits < 2,
        "is_substantive":       subst_hits >= 1,
    }


# ---------------------------------------------------------------------------
# Qualification Prompt
# ---------------------------------------------------------------------------
QUALIFICATION_SYSTEM_PROMPT = """You are CivicSync's evidence extraction component.

Do not decide what the law should be.
Extract ONLY statements that are directly supported by the provided evidence chunks.
Every extracted statement must reference one or more supplied chunk numbers (chunk_refs).

CRITICAL RULES:
1. Extract ONLY from the PROVIDED CHUNKS. Do NOT use external legal knowledge.
2. Every item must be a close paraphrase or partial quote of the actual chunk text.
3. chunk_refs must be an array of valid chunk numbers (e.g. [1] or [1, 2]).
4. If a chunk is primarily structural/metadata labels and contains no legal sentences, ignore it.
5. Do NOT infer penalties, deadlines, procedures, eligibility, rights, authorities, or remedies
   that are not EXPLICITLY stated in the provided evidence.
6. If you cannot find evidence for a field, return an empty array [].
7. Do NOT fabricate chunk references. Only reference chunks whose text actually contains the claim.
8. missing_information items should state what key legal facts are absent from the evidence.
9. Return ONLY valid JSON — no markdown, no commentary, no preamble.

METADATA GUIDANCE:
Some chunks may contain dataset structural labels (e.g., "category", "support service",
"Q: I live in X. My insurer rejected my claim. What should I do?", repeated disclaimers).
These are NOT substantive legal statements. Do not extract items from pure-metadata chunks.
If a chunk contains both metadata AND substantive legal text, extract only the substantive text.

JSON schema (EXACTLY this structure):
{
  "applicable_information": [
    {"text": "direct quote or close paraphrase", "chunk_refs": [1]}
  ],
  "rights_or_protections": [
    {"text": "...", "chunk_refs": [1]}
  ],
  "possible_actions": [
    {"text": "...", "chunk_refs": [1]}
  ],
  "authorities_or_channels": [
    {"text": "...", "chunk_refs": [1]}
  ],
  "documents_or_evidence": [
    {"text": "...", "chunk_refs": [1]}
  ],
  "conditions": [
    {"text": "...", "chunk_refs": [1]}
  ],
  "missing_information": ["...", "..."]
}
"""


def _format_chunks_for_prompt(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks as a numbered list for the qualification prompt."""
    lines = []
    for i, chunk in enumerate(chunks, 1):
        source_file = chunk.get("source_file", "unknown")
        page_start  = chunk.get("page_start", "?")
        page_end    = chunk.get("page_end", "?")
        similarity  = chunk.get("similarity", 0.0)
        content     = (chunk.get("content") or "").strip()

        score = _score_chunk(content)
        quality_tag = "[METADATA-DOMINANT — likely structural labels]" if score["is_metadata_dominant"] else ""

        lines.append(
            f"[CHUNK {i}] {quality_tag}\n"
            f"Source: {source_file} | Page: {page_start}–{page_end} | "
            f"Retrieval-rank similarity: {similarity:.4f}\n"
            f"{content}"
        )
    return "\n\n".join(lines)


def _parse_json_from_llm(raw_text: str) -> Dict[str, Any]:
    """Safely parse JSON from LLM output, handling markdown code fences."""
    clean = raw_text.strip()
    if "```" in clean:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
        clean = match.group(1) if match else re.sub(r"```(?:json)?", "", clean).strip("`").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    # Fallback: extract first { … }
    s, e = clean.find("{"), clean.rfind("}")
    if s != -1 and e != -1 and s < e:
        try:
            return json.loads(clean[s : e + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Cannot parse JSON from LLM output: {raw_text[:200]}")


def _normalise_refs(raw_refs: Any) -> List[int]:
    """Accept chunk_refs as int, list[int], or string; always return list[int]."""
    if isinstance(raw_refs, int):
        return [raw_refs]
    if isinstance(raw_refs, list):
        result = []
        for r in raw_refs:
            try:
                result.append(int(r))
            except (TypeError, ValueError):
                pass
        return result
    try:
        return [int(raw_refs)]
    except (TypeError, ValueError):
        return []


def _resolve_sources(
    items: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Validate and resolve chunk_refs to full source metadata.

    Rules:
    - Every chunk_ref integer must be 1-indexed within the chunks list.
    - Items with NO valid refs are discarded.
    - Items with an empty text are discarded.
    - The resolved item uses the 'evidence' array format with full metadata.
    """
    resolved = []
    for item in items:
        text      = str(item.get("text", "")).strip()
        raw_refs  = item.get("chunk_refs", item.get("chunk_ref"))  # support old key too
        ref_list  = _normalise_refs(raw_refs)

        if not text:
            continue

        evidence = []
        for ref in ref_list:
            # 1-indexed
            if isinstance(ref, int) and 1 <= ref <= len(chunks):
                src = chunks[ref - 1]
                evidence.append({
                    "chunk_id":    src.get("chunk_id"),
                    "source_file": src.get("source_file", "unknown"),
                    "page_start":  src.get("page_start"),
                    "page_end":    src.get("page_end"),
                    "similarity":  src.get("similarity"),
                })

        if not evidence:
            # No valid refs → discard (no unsourced items allowed)
            continue

        resolved.append({"text": text, "evidence": evidence})

    return resolved


def _determine_status(
    applicable: List[Dict],
    rights: List[Dict],
    actions: List[Dict],
    authorities: List[Dict],
    chunks: List[Dict],
) -> str:
    """
    Determine qualification status using evidence traceability only.

    SUPPORTED:
        At least one item in applicable_information OR rights_or_protections
        has valid sourced evidence. One strong source is sufficient.

    PARTIALLY_SUPPORTED:
        Some evidence exists (e.g., authorities or actions are sourced) but
        applicable_information and rights are both empty.

    INSUFFICIENT_EVIDENCE:
        No retrieved chunk provides meaningful evidence for any category.

    NOTE: Similarity scores do NOT control this determination.
    """
    if not chunks:
        return STATUS_INSUFFICIENT

    has_core   = bool(applicable or rights)
    has_action = bool(actions or authorities)

    if has_core:
        return STATUS_SUPPORTED
    if has_action:
        return STATUS_PARTIALLY_SUPPORTED
    return STATUS_INSUFFICIENT


def qualify_case(
    situation: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Qualify a case by structuring retrieved legal evidence into actionable categories.

    Evidence-first pipeline:
      retrieved chunks
        → identify supported statements via LLM extraction
        → attach validated chunk references
        → discard statements without valid evidence
        → calculate status from evidence traceability (NOT similarity)
        → return structured qualification

    Parameters
    ----------
    situation : dict
        Extracted situation dict from extract_situation().
    retrieved_chunks : list[dict]
        Retrieved chunk dicts from retriever.retrieve().
    model : str, optional
        Ollama model tag.
    temperature : float, optional
        LLM sampling temperature (default 0.0 — deterministic).

    Returns
    -------
    dict with keys:
        status, domain, issue, applicable_information, rights_or_protections,
        possible_actions, authorities_or_channels, documents_or_evidence,
        conditions, missing_information, sources, chunk_quality_report
    """
    domain = situation.get("domain", "unknown")
    issue  = situation.get("issue",  "unknown")
    intent = situation.get("intent", "unknown")

    # Controlled fallback
    fallback: Dict[str, Any] = {
        "status":                  STATUS_INSUFFICIENT,
        "domain":                  domain,
        "issue":                   issue,
        "applicable_information":  [],
        "rights_or_protections":   [],
        "possible_actions":        [],
        "authorities_or_channels": [],
        "documents_or_evidence":   [],
        "conditions":              [],
        "missing_information":     [],
        "sources":                 [],
        "chunk_quality_report":    [],
    }

    if not retrieved_chunks:
        fallback["missing_information"] = [
            "No evidence was retrieved for this query. "
            "CivicSync's current knowledge base may not contain relevant information for this specific situation."
        ]
        return fallback

    # Score every chunk for metadata dominance
    quality_report = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        score = _score_chunk(chunk.get("content", ""))
        quality_report.append({
            "chunk_ref":           i,
            "chunk_id":            chunk.get("chunk_id"),
            "is_metadata_dominant": score["is_metadata_dominant"],
            "is_substantive":       score["is_substantive"],
            "substantive_hits":     score["substantive_hits"],
        })

    chunk_text = _format_chunks_for_prompt(retrieved_chunks)

    prompt = (
        f"{QUALIFICATION_SYSTEM_PROMPT}\n\n"
        f"SITUATION CONTEXT:\n"
        f"  Domain : {domain}\n"
        f"  Issue  : {issue}\n"
        f"  Intent : {intent}\n\n"
        f"RETRIEVED LEGAL EVIDENCE:\n"
        f"{chunk_text}\n\n"
        f"TASK: From the evidence above, extract only what is directly supported.\n"
        f"If a chunk is marked [METADATA-DOMINANT], do not extract items from it unless\n"
        f"it also contains substantive legal sentences.\n"
        f"Return JSON only:\n"
    )

    try:
        raw_output = generate(prompt, temperature=temperature, model=model)
        data       = _parse_json_from_llm(raw_output)

        # Resolve every list — invalid refs are discarded
        applicable  = _resolve_sources(data.get("applicable_information",  []), retrieved_chunks)
        rights      = _resolve_sources(data.get("rights_or_protections",   []), retrieved_chunks)
        actions     = _resolve_sources(data.get("possible_actions",        []), retrieved_chunks)
        authorities = _resolve_sources(data.get("authorities_or_channels", []), retrieved_chunks)
        documents   = _resolve_sources(data.get("documents_or_evidence",   []), retrieved_chunks)
        conditions  = _resolve_sources(data.get("conditions",              []), retrieved_chunks)
        missing     = [str(m).strip() for m in data.get("missing_information", []) if str(m).strip()]

        # Build de-duplicated source list from all evidence arrays
        seen_ids: set = set()
        sources:  List[Dict] = []
        for item_list in [applicable, rights, actions, authorities, documents, conditions]:
            for item in item_list:
                for ev in item.get("evidence", []):
                    cid = ev.get("chunk_id")
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        sources.append(ev)

        status = _determine_status(applicable, rights, actions, authorities, retrieved_chunks)

        # If status is INSUFFICIENT but chunks exist, add a missing_information note
        if status == STATUS_INSUFFICIENT and retrieved_chunks and not missing:
            missing = [
                "The retrieved evidence chunks appear to contain primarily structural "
                "dataset metadata rather than substantive legal prose. "
                "No directly extractable legal statements were found for this query. "
                "The underlying source documents may need to be re-processed or supplemented."
            ]

        return {
            "status":                  status,
            "domain":                  domain,
            "issue":                   issue,
            "applicable_information":  applicable,
            "rights_or_protections":   rights,
            "possible_actions":        actions,
            "authorities_or_channels": authorities,
            "documents_or_evidence":   documents,
            "conditions":              conditions,
            "missing_information":     missing,
            "sources":                 sources,
            "chunk_quality_report":    quality_report,
        }

    except (OllamaClientError, ValueError, Exception):
        fallback["chunk_quality_report"] = quality_report
        return fallback
