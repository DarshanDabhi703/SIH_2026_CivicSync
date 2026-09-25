"""
test_pipeline.py
================
Automated tests for the CivicSync Phase 1 PDF pipeline.

Tests verify:
  1. A PDF can be opened by parse_pdf()
  2. Text is extracted (at least one page has non-empty text)
  3. Page boundaries are preserved (each record has a page number)
  4. Chunks are generated from parsed pages
  5. Every chunk has the required metadata fields
  6. JSONL output can be written and read back correctly
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup — works when run from project root or from tests/
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pdf_parser import parse_pdf          # noqa: E402
from src.chunker import create_chunks, clean_text  # noqa: E402
from src.pipeline import save_chunks_jsonl, infer_domain  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PDFS_DIR = PROJECT_ROOT / "data" / "pdfs"


def _get_any_pdf() -> Path:
    """Return the first available PDF for generic tests."""
    pdfs = sorted(PDFS_DIR.glob("*.pdf"))
    if not pdfs:
        pytest.skip(f"No PDF files found in {PDFS_DIR}")
    return pdfs[0]


# ---------------------------------------------------------------------------
# 1. PDF can be opened
# ---------------------------------------------------------------------------

def test_pdf_can_be_opened():
    """parse_pdf() must return a ParseResult without raising."""
    pdf_path = _get_any_pdf()
    result = parse_pdf(pdf_path)
    assert result is not None, "parse_pdf returned None"
    assert result["source_file"] == pdf_path.name
    assert result["total_pages"] > 0, "PDF reported 0 pages"


# ---------------------------------------------------------------------------
# 2. Text is extracted
# ---------------------------------------------------------------------------

def test_text_is_extracted():
    """At least one page must yield non-empty text."""
    pdf_path = _get_any_pdf()
    result = parse_pdf(pdf_path)
    non_empty = [r for r in result["records"] if r["text"].strip()]
    assert len(non_empty) > 0, (
        f"No text was extracted from '{pdf_path.name}'. "
        "The PDF may be image-only."
    )


# ---------------------------------------------------------------------------
# 3. Page boundaries are preserved
# ---------------------------------------------------------------------------

def test_page_boundaries_preserved():
    """Every PageRecord must carry a valid page number."""
    pdf_path = _get_any_pdf()
    result = parse_pdf(pdf_path)
    for record in result["records"]:
        assert "page" in record, "PageRecord missing 'page' field"
        assert isinstance(record["page"], int), "'page' must be int"
        assert record["page"] >= 1, "'page' must be >= 1"
        assert "source_file" in record, "PageRecord missing 'source_file'"


# ---------------------------------------------------------------------------
# 4. Chunks are generated
# ---------------------------------------------------------------------------

def test_chunks_are_generated():
    """create_chunks() must return at least one chunk for a text-bearing PDF."""
    pdf_path = _get_any_pdf()
    result = parse_pdf(pdf_path)
    domain = infer_domain(pdf_path.name)
    chunks = create_chunks(result["records"], domain=domain, source_file=pdf_path.name)
    assert len(chunks) > 0, (
        f"create_chunks returned 0 chunks for '{pdf_path.name}'"
    )


# ---------------------------------------------------------------------------
# 5. Every chunk has required metadata fields
# ---------------------------------------------------------------------------

REQUIRED_CHUNK_FIELDS = {"chunk_id", "source_file", "page_start", "page_end", "text", "metadata"}
REQUIRED_METADATA_FIELDS = {"domain"}


def test_chunk_metadata_completeness():
    """All chunks must contain the mandatory fields and non-empty text."""
    pdf_path = _get_any_pdf()
    result = parse_pdf(pdf_path)
    domain = infer_domain(pdf_path.name)
    chunks = create_chunks(result["records"], domain=domain, source_file=pdf_path.name)

    for i, chunk in enumerate(chunks):
        missing = REQUIRED_CHUNK_FIELDS - chunk.keys()
        assert not missing, f"Chunk #{i} missing fields: {missing}"

        meta_missing = REQUIRED_METADATA_FIELDS - chunk["metadata"].keys()
        assert not meta_missing, f"Chunk #{i} metadata missing: {meta_missing}"

        assert chunk["text"].strip(), f"Chunk #{i} has empty text"
        assert chunk["chunk_id"], f"Chunk #{i} has empty chunk_id"
        assert chunk["source_file"], f"Chunk #{i} has empty source_file"
        assert chunk["page_start"] >= 1, f"Chunk #{i} page_start < 1"
        assert chunk["page_end"] >= chunk["page_start"], (
            f"Chunk #{i} page_end < page_start"
        )


# ---------------------------------------------------------------------------
# 6. JSONL output can be written and read back
# ---------------------------------------------------------------------------

def test_jsonl_roundtrip():
    """Chunks saved as JSONL must be deserializable and data-identical."""
    pdf_path = _get_any_pdf()
    result = parse_pdf(pdf_path)
    domain = infer_domain(pdf_path.name)
    chunks = create_chunks(result["records"], domain=domain, source_file=pdf_path.name)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test_chunks.jsonl"
        save_chunks_jsonl(chunks, out_path)

        assert out_path.exists(), "JSONL file was not created"

        loaded: list[dict] = []
        with out_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    loaded.append(json.loads(line))

    assert len(loaded) == len(chunks), (
        f"Expected {len(chunks)} lines, got {len(loaded)}"
    )
    for orig, read_back in zip(chunks, loaded):
        assert orig["chunk_id"] == read_back["chunk_id"]
        assert orig["text"] == read_back["text"]
        assert orig["metadata"]["domain"] == read_back["metadata"]["domain"]


# ---------------------------------------------------------------------------
# Extra: test clean_text does not alter legal numbers
# ---------------------------------------------------------------------------

def test_clean_text_preserves_legal_content():
    """clean_text must not alter section numbers, dates, or key legal terms."""
    sample = (
        "Section 185 of the Motor Vehicles Act, 1988 states that driving\n"
        "under the influence of alcohol (blood alcohol > 30 mg/100 ml)\n"
        "is punishable with imprisonment up to 6 months and/or fine of Rs. 2,000.\n"
    )
    result = clean_text(sample)
    # Check critical legal content is intact
    assert "Section 185" in result
    assert "Motor Vehicles Act, 1988" in result
    assert "30 mg/100 ml" in result
    assert "Rs. 2,000" in result
    assert "6 months" in result


# ---------------------------------------------------------------------------
# Extra: infer_domain mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename,expected_domain", [
    ("india_traffic_rules_formatted.pdf", "traffic"),
    ("Indian_Labour_Rights_ML_Dataset_v5 (1).pdf", "labour"),
    ("Women Safety and Rights.pdf", "women_safety"),
    ("consumer_rights_formatted.pdf", "consumer"),
    ("india_insurance_financial_rights_dataset_formatted.pdf", "insurance"),
    ("Indian_Land_Property_Tenancy_ML_Dataset.pdf", "land_property"),
])
def test_domain_inference(filename: str, expected_domain: str):
    assert infer_domain(filename) == expected_domain, (
        f"Expected domain '{expected_domain}' for '{filename}', "
        f"got '{infer_domain(filename)}'"
    )
