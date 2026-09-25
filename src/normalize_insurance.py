"""
normalize_insurance.py
======================
CivicSync Phase 5B — Insurance Dataset Normalization

Combines and normalizes all IRDAI/NHA insurance regulatory JSONL chunks
from data/processed/insurance_v2/ into a single normalized dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "insurance_v2"
OUTPUT_FILE = INPUT_DIR / "insurance_chunks_v2.jsonl"
VALIDATION_FILE = INPUT_DIR / "insurance_validation.json"


def load_jsonl_files(input_dir: str | Path) -> Tuple[List[Dict[str, Any]], int, List[str]]:
    """
    Load all JSONL files from input_dir (excluding insurance_chunks_v2.jsonl).

    Returns
    -------
    tuple
        (raw_records, file_count, file_names)
    """
    path = Path(input_dir)
    files = sorted([
        f for f in path.glob("*.jsonl")
        if f.name != "insurance_chunks_v2.jsonl"
    ])

    raw_records: List[Dict[str, Any]] = []
    file_names: List[str] = []

    for f in files:
        file_names.append(f.name)
        with open(f, mode="r", encoding="utf-8") as fh:
            for line in fh:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    record = json.loads(line_str)
                    raw_records.append(record)
                except json.JSONDecodeError:
                    continue

    return raw_records, len(files), file_names


def deduplicate_records(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Filter out empty/whitespace text and exact duplicate text records.

    Returns
    -------
    tuple
        (cleaned_records, empty_removed_count, duplicates_removed_count)
    """
    cleaned: List[Dict[str, Any]] = []
    seen_texts: set[str] = set()

    empty_removed = 0
    duplicates_removed = 0

    for rec in records:
        text = (rec.get("text") or rec.get("content") or "").strip()
        if not text:
            empty_removed += 1
            continue

        if text in seen_texts:
            duplicates_removed += 1
            continue

        seen_texts.add(text)
        cleaned.append(rec)

    return cleaned, empty_removed, duplicates_removed


def normalize_record(raw_record: Dict[str, Any], index: int) -> Dict[str, Any]:
    """
    Normalize a single record:
    - Generate chunk_id: INS-V2-000001, INS-V2-000002, ...
    - Force domain: "insurance"
    - Preserve source_file, page_start, page_end, text
    - Add/update metadata dict
    """
    chunk_id = f"INS-V2-{index:06d}"
    domain = "insurance"
    source_file = str(raw_record.get("source_file", "")).strip()
    page_start = int(raw_record.get("page_start", 1))
    page_end = int(raw_record.get("page_end", page_start))
    text = str(raw_record.get("text") or raw_record.get("content") or "").strip()

    metadata = {
        "domain": domain,
        "source_file": source_file,
        "page_start": page_start,
        "page_end": page_end,
        "source_type": "regulatory",
    }

    return {
        "chunk_id": chunk_id,
        "domain": domain,
        "source_file": source_file,
        "page_start": page_start,
        "page_end": page_end,
        "text": text,
        "metadata": metadata,
    }


def validate_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate normalized records against all quality rules.

    Returns
    -------
    dict
        Validation results summary.
    """
    results = {
        "unique_ids": True,
        "domain": True,
        "source_metadata": True,
        "non_empty_text": True,
        "page_metadata": True,
        "jsonl_validity": True,
    }

    seen_ids: set[str] = set()

    for rec in records:
        cid = rec.get("chunk_id", "")
        if not cid.startswith("INS-V2-") or cid in seen_ids:
            results["unique_ids"] = False
        seen_ids.add(cid)

        if rec.get("domain") != "insurance":
            results["domain"] = False

        if not rec.get("source_file"):
            results["source_metadata"] = False

        meta = rec.get("metadata", {})
        if not meta.get("source_file") or meta.get("domain") != "insurance":
            results["source_metadata"] = False

        txt = (rec.get("text") or "").strip()
        if not txt:
            results["non_empty_text"] = False

        ps = rec.get("page_start")
        pe = rec.get("page_end")
        if ps is None or pe is None or ps > pe:
            results["page_metadata"] = False

        try:
            json.dumps(rec)
        except (TypeError, ValueError):
            results["jsonl_validity"] = False

    all_passed = all(results.values())
    results["all_passed"] = all_passed
    return results


def save_jsonl(records: List[Dict[str, Any]], output_path: str | Path) -> None:
    """Save normalized records to a JSONL file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode="w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_pipeline() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Execute complete insurance normalization pipeline:
    Load -> Deduplicate -> Normalize -> Validate -> Save -> Write Report.
    """
    raw_records, file_count, _ = load_jsonl_files(INPUT_DIR)
    input_chunk_count = len(raw_records)

    cleaned_records, empty_removed, duplicates_removed = deduplicate_records(raw_records)

    normalized_records = [
        normalize_record(rec, idx)
        for idx, rec in enumerate(cleaned_records, 1)
    ]

    val_results = validate_records(normalized_records)

    # Count unique source documents
    source_docs = {rec["source_file"] for rec in normalized_records if rec.get("source_file")}

    # Save output JSONL
    save_jsonl(normalized_records, OUTPUT_FILE)

    # Save validation report
    validation_report = {
        "input_files": file_count,
        "input_chunks": input_chunk_count,
        "output_chunks": len(normalized_records),
        "duplicates_removed": duplicates_removed,
        "empty_records_removed": empty_removed,
        "source_documents": len(source_docs),
        "validation_passed": val_results["all_passed"],
    }

    VALIDATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(VALIDATION_FILE, mode="w", encoding="utf-8") as fh:
        json.dump(validation_report, fh, indent=2)

    # Print final formatted report
    print("=" * 60)
    print("CivicSync Phase 5B - Insurance Normalization")
    print("=" * 60)
    print(f"Input JSONL files : {file_count}")
    print(f"Input chunks      : {input_chunk_count}")
    print(f"Output chunks     : {len(normalized_records)}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Empty removed     : {empty_removed}")
    print(f"Source documents  : {len(source_docs)}")
    print("Domain            : insurance")
    print()
    print("Validation:")
    print(f"  Unique IDs       : {'PASS' if val_results['unique_ids'] else 'FAIL'}")
    print(f"  Domain           : {'PASS' if val_results['domain'] else 'FAIL'}")
    print(f"  Source metadata  : {'PASS' if val_results['source_metadata'] else 'FAIL'}")
    print(f"  Non-empty text   : {'PASS' if val_results['non_empty_text'] else 'FAIL'}")
    print(f"  Page metadata    : {'PASS' if val_results['page_metadata'] else 'FAIL'}")
    print(f"  JSONL validity   : {'PASS' if val_results['jsonl_validity'] else 'FAIL'}")
    print()
    print(f"STATUS: {'SUCCESS' if val_results['all_passed'] else 'FAILED'}")
    print("=" * 60)

    return normalized_records, validation_report


if __name__ == "__main__":
    run_pipeline()
