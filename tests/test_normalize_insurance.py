"""
test_normalize_insurance.py
============================
Unit & Integration tests for CivicSync Phase 5B — Insurance Dataset Normalization.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.normalize_insurance import (
    load_jsonl_files,
    normalize_record,
    deduplicate_records,
    validate_records,
    save_jsonl,
    run_pipeline,
    INPUT_DIR,
    OUTPUT_FILE,
    VALIDATION_FILE,
)


def test_normalize_record_structure():
    raw = {
        "source_file": "irdai-test.pdf",
        "page_start": 2,
        "page_end": 4,
        "text": "Sample regulatory clause text.",
    }
    norm = normalize_record(raw, 42)

    assert norm["chunk_id"] == "INS-V2-000042"
    assert norm["domain"] == "insurance"
    assert norm["source_file"] == "irdai-test.pdf"
    assert norm["page_start"] == 2
    assert norm["page_end"] == 4
    assert norm["text"] == "Sample regulatory clause text."
    assert norm["metadata"]["domain"] == "insurance"
    assert norm["metadata"]["source_file"] == "irdai-test.pdf"
    assert norm["metadata"]["page_start"] == 2
    assert norm["metadata"]["page_end"] == 4
    assert norm["metadata"]["source_type"] == "regulatory"


def test_deduplicate_records():
    raw_list = [
        {"source_file": "a.pdf", "text": "Clause 1 text"},
        {"source_file": "a.pdf", "text": "   "},  # empty
        {"source_file": "b.pdf", "text": "Clause 1 text"},  # duplicate text
        {"source_file": "c.pdf", "text": "Clause 2 text"},
    ]
    cleaned, empty_cnt, dup_cnt = deduplicate_records(raw_list)

    assert len(cleaned) == 2
    assert empty_cnt == 1
    assert dup_cnt == 1
    assert cleaned[0]["text"] == "Clause 1 text"
    assert cleaned[1]["text"] == "Clause 2 text"


def test_validate_records_pass():
    records = [
        normalize_record({"source_file": "doc1.pdf", "page_start": 1, "page_end": 1, "text": "Content A"}, 1),
        normalize_record({"source_file": "doc2.pdf", "page_start": 2, "page_end": 3, "text": "Content B"}, 2),
    ]
    val = validate_records(records)
    assert val["all_passed"] is True
    assert val["unique_ids"] is True
    assert val["domain"] is True
    assert val["source_metadata"] is True
    assert val["non_empty_text"] is True
    assert val["page_metadata"] is True
    assert val["jsonl_validity"] is True


def test_pipeline_execution():
    """Run full pipeline and test all requirements on generated output files."""
    records, report = run_pipeline()

    # 1. Output exists
    assert OUTPUT_FILE.exists()
    assert VALIDATION_FILE.exists()

    # 2. Report checks
    assert report["validation_passed"] is True
    assert report["input_files"] == 16
    assert report["input_chunks"] == 647
    assert report["output_chunks"] > 0

    # 3. Read generated output file and verify line-by-line
    seen_ids = set()
    with open(OUTPUT_FILE, mode="r", encoding="utf-8") as fh:
        lines = fh.readlines()

    assert len(lines) == len(records)

    for line in lines:
        rec = json.loads(line)  # Valid JSONL

        # Every record has domain == "insurance"
        assert rec["domain"] == "insurance"
        assert rec["metadata"]["domain"] == "insurance"

        # Every chunk ID starts with "INS-V2-"
        cid = rec["chunk_id"]
        assert cid.startswith("INS-V2-")

        # Chunk IDs are unique
        assert cid not in seen_ids
        seen_ids.add(cid)

        # Source file exists and is preserved
        assert rec["source_file"] != ""
        assert rec["metadata"]["source_file"] == rec["source_file"]

        # Text is non-empty
        assert rec["text"].strip() != ""

        # page_start <= page_end
        assert rec["page_start"] <= rec["page_end"]
