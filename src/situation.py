"""
situation.py
============
CivicSync Phase 4 — Circumstance Intelligence Layer

Extracts structured situation metadata from natural-language queries
using Ollama llama3.1:8b without making legal conclusions.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.llm_gateway import generate
from src.llm_client import OllamaClientError

# ---------------------------------------------------------------------------
# Controlled Vocabulary Definitions
# ---------------------------------------------------------------------------
VALID_DOMAINS = {
    "traffic",
    "labour",
    "consumer",
    "insurance",
    "land_property",
    "women_safety",
    "unknown",
}

VALID_INTENTS = {
    "understand_rights",
    "understand_consequences",
    "file_complaint",
    "seek_remedy",
    "obtain_document",
    "understand_procedure",
    "general_information",
    "unknown",
}

@dataclass
class SituationSchema:
    original_query: str
    domain: str = "unknown"
    situation: str = "unknown"
    issue: str = "unknown"
    intent: str = "unknown"
    jurisdiction: str = "India"
    entities: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def clean_and_validate(self) -> "SituationSchema":
        # Normalize domain
        norm_dom = str(self.domain).strip().lower()
        if norm_dom not in VALID_DOMAINS:
            # Attempt simple string matching fallback
            matched = False
            for d in VALID_DOMAINS:
                if d != "unknown" and d in norm_dom:
                    self.domain = d
                    matched = True
                    break
            if not matched:
                self.domain = "unknown"
        else:
            self.domain = norm_dom

        # Normalize intent
        norm_intent = str(self.intent).strip().lower()
        if norm_intent not in VALID_INTENTS:
            matched = False
            for i in VALID_INTENTS:
                if i != "unknown" and i in norm_intent:
                    self.intent = i
                    matched = True
                    break
            if not matched:
                self.intent = "unknown"
        else:
            self.intent = norm_intent

        # Normalize string fields
        self.situation = str(self.situation).strip() if self.situation else "unknown"
        self.issue = str(self.issue).strip() if self.issue else "unknown"
        self.jurisdiction = str(self.jurisdiction).strip() if self.jurisdiction else "India"

        # Normalize entities list
        if isinstance(self.entities, list):
            self.entities = [str(e).strip() for e in self.entities if e]
        else:
            self.entities = []

        # Validate confidence score (0.0 to 1.0)
        try:
            conf = float(self.confidence)
            self.confidence = max(0.0, min(1.0, conf))
        except (ValueError, TypeError):
            self.confidence = 0.0

        # If domain is unknown, default confidence should be low
        if self.domain == "unknown":
            self.confidence = min(self.confidence, 0.40)

        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Extraction Prompt
# ---------------------------------------------------------------------------
EXTRACTION_SYSTEM_PROMPT = """You are CivicSync's situation extraction component.
Your task is ONLY to identify and extract the situation described by the citizen.

DO NOT provide legal advice.
DO NOT identify laws or section numbers.
DO NOT determine guilt or liability.
DO NOT invent facts.

Extract ONLY information explicitly stated or strongly implied by the user's query.

Supported domains (MUST be one of these exact strings):
- traffic
- labour
- consumer
- insurance
- land_property
- women_safety
- unknown

Supported intents (MUST be one of these exact strings):
- understand_rights
- understand_consequences
- file_complaint
- seek_remedy
- obtain_document
- understand_procedure
- general_information
- unknown

Return ONLY valid JSON matching this exact structure (no markdown formatting, no commentary):
{
  "domain": "...",
  "situation": "...",
  "issue": "...",
  "intent": "...",
  "jurisdiction": "India",
  "entities": ["..."],
  "confidence": 0.95
}

If something is ambiguous, unknown, or not supported by the 6 domains, set domain to "unknown" and confidence to 0.40 or below.
"""


def _parse_json_from_llm(raw_text: str) -> Dict[str, Any]:
    """Safely extract JSON dictionary from raw LLM output."""
    clean_text = raw_text.strip()
    
    # Strip markdown codeblocks if present (e.g. ```json ... ```)
    if "```" in clean_text:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_text, re.DOTALL)
        if match:
            clean_text = match.group(1)
        else:
            clean_text = re.sub(r"```(?:json)?", "", clean_text).strip("`").strip()

    # Try direct parse
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        pass

    # Try finding first '{' and last '}'
    start_idx = clean_text.find("{")
    end_idx = clean_text.rfind("}")
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        json_candidate = clean_text[start_idx : end_idx + 1]
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse valid JSON from LLM output: {raw_text[:200]}")


def extract_situation(
    user_query: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Extract structured situation metadata from a user's natural language query.

    Parameters
    ----------
    user_query : str
        The citizen's query.
    model : str, optional
        Ollama model tag to use.
    temperature : float, optional
        Sampling temperature (default 0.0 for deterministic JSON extraction).

    Returns
    -------
    dict
        Validated situation dictionary containing:
        original_query, domain, situation, issue, intent, jurisdiction, entities, confidence.
    """
    if not user_query or not user_query.strip():
        schema = SituationSchema(original_query=user_query or "")
        return schema.clean_and_validate().to_dict()

    prompt = f"{EXTRACTION_SYSTEM_PROMPT}\n\nUSER QUERY:\n\"{user_query}\"\n\nJSON OUTPUT:\n"

    try:
        raw_output = generate(prompt, temperature=temperature, model=model)
        parsed_data = _parse_json_from_llm(raw_output)

        schema = SituationSchema(
            original_query=user_query,
            domain=parsed_data.get("domain", "unknown"),
            situation=parsed_data.get("situation", "unknown"),
            issue=parsed_data.get("issue", "unknown"),
            intent=parsed_data.get("intent", "unknown"),
            jurisdiction=parsed_data.get("jurisdiction", "India"),
            entities=parsed_data.get("entities", []),
            confidence=parsed_data.get("confidence", 0.5),
        )
        return schema.clean_and_validate().to_dict()

    except (OllamaClientError, ValueError, Exception) as exc:
        # Controlled fallback on any LLM or parsing error - never crash
        schema = SituationSchema(
            original_query=user_query,
            domain="unknown",
            situation="unknown",
            issue="unknown",
            intent="unknown",
            jurisdiction="India",
            entities=[],
            confidence=0.0,
        )
        return schema.clean_and_validate().to_dict()
