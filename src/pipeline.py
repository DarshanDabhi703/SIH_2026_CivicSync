"""
pipeline.py
===========
CivicSync Phase 1 - PDF -> clean text -> chunks -> JSONL

Usage:
    python src/pipeline.py

Automatically discovers all PDFs from data/pdfs/, processes each one,
and writes JSONL output to data/processed/.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — works when run as  python src/pipeline.py  from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pdf_parser import parse_pdf          # noqa: E402
from src.chunker import create_chunks         # noqa: E402

# ---------------------------------------------------------------------------
# Domain mapping  (inferred from filename; extend as PDFs are renamed)
# ---------------------------------------------------------------------------
DOMAIN_MAP: dict[str, str] = {
    "traffic":    "traffic",
    "labour":     "labour",
    "labor":      "labour",
    "women":      "women_safety",
    "consumer":   "consumer",
    "insurance":  "insurance",
    "financial":  "insurance",
    "land":       "land_property",
    "property":   "land_property",
    "tenancy":    "land_property",
}

OUTPUT_STEM_MAP: dict[str, str] = {
    "traffic":       "traffic_chunks",
    "labour":        "labour_chunks",
    "women_safety":  "women_safety_chunks",
    "consumer":      "consumer_chunks",
    "insurance":     "insurance_chunks",
    "land_property": "land_property_chunks",
}


def infer_domain(filename: str) -> str:
    """Infer a domain label from the PDF filename."""
    lower = filename.lower()
    for keyword, domain in DOMAIN_MAP.items():
        if keyword in lower:
            return domain
    # Fallback: use the stem, cleaned
    return Path(filename).stem.lower().replace(" ", "_")


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def save_chunks_jsonl(chunks: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _sep(char: str = "-", width: int = 60) -> str:
    return char * width


def print_report(
    source_file: str,
    total_pages: int,
    extracted: int,
    failed: list[int],
    empty: list[int],
    chunks: list[dict],
    output_path: Path,
) -> None:
    lengths = [len(c["text"]) for c in chunks]
    avg = int(sum(lengths) / len(lengths)) if lengths else 0
    mn  = min(lengths) if lengths else 0
    mx  = max(lengths) if lengths else 0

    print(_sep("="))
    print(f"  {source_file.upper()}")
    print(_sep("-"))
    print(f"  Pages        : {total_pages}")
    print(f"  Extracted    : {extracted}")
    print(f"  Empty pages  : {len(empty)}" + (f"  {empty}" if empty else ""))
    print(f"  Failed pages : {len(failed)}" + (f"  {failed}" if failed else ""))
    print(f"  Chunks       : {len(chunks)}")
    print(f"  Avg chars    : {avg}")
    print(f"  Min chars    : {mn}")
    print(f"  Max chars    : {mx}")
    print(f"  Output       : {output_path}")

    if chunks:
        print()
        print("  -- First 2 chunks (preview) ------------------------------------------")
        for i, chunk in enumerate(chunks[:2], 1):
            preview = chunk["text"][:300].replace("\n", " ")
            print(f"\n  [{i}] chunk_id   : {chunk['chunk_id']}")
            print(f"       pages      : {chunk['page_start']}–{chunk['page_end']}")
            print(f"       domain     : {chunk['metadata']['domain']}")
            print(f"       chars      : {len(chunk['text'])}")
            print(f"       text       : {preview}…")
    print()


# ---------------------------------------------------------------------------
# Per-PDF processing
# ---------------------------------------------------------------------------

def process_pdf(pdf_path: Path, output_dir: Path) -> dict:
    """Process a single PDF. Returns a summary dict for the final report."""
    source_file = pdf_path.name
    print(f"\n{'='*60}")
    print(f"Processing: {source_file}")
    print(f"{'='*60}")

    summary = {
        "source_file": source_file,
        "status": "ok",
        "error": None,
        "total_pages": 0,
        "extracted": 0,
        "failed": [],
        "empty": [],
        "chunks": 0,
        "output": None,
    }

    # Parse
    try:
        parse_result = parse_pdf(pdf_path)
    except RuntimeError as exc:
        summary["status"] = "error"
        summary["error"] = str(exc)
        print(f"  [ERROR] {exc}")
        return summary

    summary["total_pages"] = parse_result["total_pages"]
    summary["extracted"]   = len(parse_result["extracted_pages"])
    summary["failed"]      = parse_result["failed_pages"]
    summary["empty"]       = parse_result["empty_pages"]

    # Infer domain and chunk
    domain = infer_domain(source_file)
    chunks = create_chunks(
        pages=parse_result["records"],
        domain=domain,
        source_file=source_file,
    )
    summary["chunks"] = len(chunks)

    # Save
    output_stem = OUTPUT_STEM_MAP.get(domain, f"{domain}_chunks")
    output_path = output_dir / f"{output_stem}.jsonl"
    save_chunks_jsonl(chunks, output_path)
    summary["output"] = str(output_path)

    # Report
    print_report(
        source_file=source_file,
        total_pages=parse_result["total_pages"],
        extracted=len(parse_result["extracted_pages"]),
        failed=parse_result["failed_pages"],
        empty=parse_result["empty_pages"],
        chunks=chunks,
        output_path=output_path,
    )

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pipeline(pdfs_dir: Path | None = None, output_dir: Path | None = None) -> list[dict]:
    """
    Discover and process all PDFs in pdfs_dir.

    Parameters
    ----------
    pdfs_dir : Path, optional
        Directory containing PDF files. Defaults to <project>/data/pdfs/.
    output_dir : Path, optional
        Directory for output JSONL files. Defaults to <project>/data/processed/.

    Returns
    -------
    list[dict]
        One summary dict per PDF.
    """
    if pdfs_dir is None:
        pdfs_dir = PROJECT_ROOT / "data" / "pdfs"
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "processed"

    pdf_files = sorted(pdfs_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[WARNING] No PDF files found in: {pdfs_dir}")
        return []

    print(f"\nCivicSync — PDF Processing Pipeline")
    print(f"PDFs directory : {pdfs_dir}")
    print(f"Output dir     : {output_dir}")
    print(f"PDFs found     : {len(pdf_files)}")

    summaries: list[dict] = []
    for pdf_path in pdf_files:
        summary = process_pdf(pdf_path, output_dir)
        summaries.append(summary)

    # Final summary
    print(_sep("=", 60))
    print("  PIPELINE COMPLETE - SUMMARY")
    print(_sep("-", 60))
    total_chunks = sum(s["chunks"] for s in summaries)
    ok_count = sum(1 for s in summaries if s["status"] == "ok")
    err_count = sum(1 for s in summaries if s["status"] == "error")

    for s in summaries:
        status_icon = "✓" if s["status"] == "ok" else "✗"
        print(
            f"  {status_icon}  {s['source_file']:<55} "
            f"pages={s['total_pages']}  chunks={s['chunks']}"
        )
        if s["error"]:
            print(f"      ERROR: {s['error']}")

    print(_sep("-", 60))
    print(f"  PDFs processed : {ok_count}/{len(summaries)}")
    print(f"  Errors         : {err_count}")
    print(f"  Total chunks   : {total_chunks}")
    print(f"  JSONL files    : {output_dir}")
    print(_sep("=", 60))

    return summaries


if __name__ == "__main__":
    run_pipeline()
