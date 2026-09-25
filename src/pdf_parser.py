"""
pdf_parser.py
=============
Extracts text from PDF files page-by-page using pypdf.

Returns structured records preserving page boundaries.
Does NOT modify, summarize, or rewrite legal content.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import TypedDict

try:
    from pypdf import PdfReader
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pypdf is required: install with  pip install pypdf>=4.0.0"
    ) from exc


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class PageRecord(TypedDict):
    source_file: str   # basename of the PDF, e.g. "traffic.pdf"
    page: int          # 1-based page number
    text: str          # raw extracted text (may be empty for image pages)


class ParseResult(TypedDict):
    source_file: str
    total_pages: int
    extracted_pages: list[int]   # page numbers that yielded text
    failed_pages: list[int]      # page numbers that raised exceptions
    empty_pages: list[int]       # page numbers with no extractable text
    records: list[PageRecord]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_pdf(pdf_path: str | Path) -> ParseResult:
    """
    Open a PDF and extract text from every page.

    Parameters
    ----------
    pdf_path : str | Path
        Absolute or relative path to the PDF file.

    Returns
    -------
    ParseResult
        Structured result containing per-page records and diagnostic info.
    """
    pdf_path = Path(pdf_path)
    source_file = pdf_path.name

    result: ParseResult = {
        "source_file": source_file,
        "total_pages": 0,
        "extracted_pages": [],
        "failed_pages": [],
        "empty_pages": [],
        "records": [],
    }

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        # Cannot open the file at all — caller will handle this
        raise RuntimeError(
            f"Cannot open PDF '{pdf_path}': {exc}"
        ) from exc

    result["total_pages"] = len(reader.pages)

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001
            # Record extraction failure for this page; do not crash pipeline
            result["failed_pages"].append(page_num)
            print(
                f"  [WARNING] Failed to extract page {page_num} "
                f"of '{source_file}':\n"
                f"  {traceback.format_exc(limit=2).strip()}"
            )
            continue

        if not text.strip():
            result["empty_pages"].append(page_num)
            # Still append a record so page boundaries are preserved
            result["records"].append(
                PageRecord(source_file=source_file, page=page_num, text="")
            )
        else:
            result["extracted_pages"].append(page_num)
            result["records"].append(
                PageRecord(source_file=source_file, page=page_num, text=text)
            )

    return result
