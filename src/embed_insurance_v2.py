"""
embed_insurance_v2.py
=====================
CivicSync Phase 5B-2 — Insurance V2 Embedding Pipeline

Generates 768-dimensional L2-normalized embeddings for normalized Insurance V2 chunks
using intfloat/multilingual-e5-base with 'passage: ' prefix.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embedder import (
    load_model,
    embed_chunks,
    read_chunks_jsonl,
    write_embeddings_jsonl,
    DEFAULT_MODEL,
    EMBEDDING_DIM,
)

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "insurance_v2" / "insurance_chunks_v2.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "embeddings" / "insurance_v2_embeddings.jsonl"


def validate_insurance_v2_embeddings(
    records: List[Dict[str, Any]],
    expected_count: int,
) -> Dict[str, Any]:
    """
    Validate output embedding records for Phase 5B-2.

    Returns
    -------
    dict
        Validation status flags.
    """
    val_status = {
        "count_match": len(records) == expected_count and expected_count > 0,
        "dimension": True,
        "numeric_vectors": True,
        "no_nulls": True,
        "metadata": True,
        "normalized": True,
    }

    for rec in records:
        emb = rec.get("embedding")
        if emb is None:
            val_status["no_nulls"] = False
            val_status["numeric_vectors"] = False
            val_status["dimension"] = False
            continue

        if len(emb) != EMBEDDING_DIM:
            val_status["dimension"] = False

        if any(v is None for v in emb):
            val_status["no_nulls"] = False

        if any(not isinstance(v, (int, float)) or math.isnan(v) for v in emb):
            val_status["numeric_vectors"] = False

        # Metadata checks
        cid = rec.get("chunk_id", "")
        if not cid.startswith("INS-V2-"):
            val_status["metadata"] = False
        if rec.get("domain") != "insurance":
            val_status["metadata"] = False
        if not rec.get("source_file"):
            val_status["metadata"] = False
        if not rec.get("text"):
            val_status["metadata"] = False

        # L2 Normalization check (||v||_2 ≈ 1.0)
        if isinstance(emb, list) and len(emb) == EMBEDDING_DIM:
            norm_sq = sum(v * v for v in emb)
            if abs(norm_sq - 1.0) > 1e-3:
                val_status["normalized"] = False

    val_status["all_passed"] = all(val_status.values())
    return val_status


def generate_insurance_v2_embeddings(
    input_path: Path = INPUT_FILE,
    output_path: Path = OUTPUT_FILE,
    batch_size: int = 32,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generate and save embeddings for Insurance V2 chunks.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    chunks = read_chunks_jsonl(input_path)
    input_count = len(chunks)

    model = load_model(DEFAULT_MODEL)
    records = embed_chunks(chunks=chunks, model=model, batch_size=batch_size, model_name=DEFAULT_MODEL)

    val_status = validate_insurance_v2_embeddings(records, expected_count=input_count)

    write_embeddings_jsonl(records, output_path)

    # Print final formatted report
    print("=" * 60)
    print("CivicSync Phase 5B-2 — Insurance Embeddings")
    print("=" * 60)
    print(f"Input chunks      : {input_count}")
    print(f"Embeddings        : {len(records)}")
    print(f"Model             : {DEFAULT_MODEL}")
    print(f"Dimensions        : {EMBEDDING_DIM}")
    print("Prefix            : passage:")
    print()
    print("Validation:")
    print(f"  Count match      : {'PASS' if val_status['count_match'] else 'FAIL'}")
    print(f"  Dimension        : {'PASS' if val_status['dimension'] else 'FAIL'}")
    print(f"  Numeric vectors  : {'PASS' if val_status['numeric_vectors'] else 'FAIL'}")
    print(f"  No nulls         : {'PASS' if val_status['no_nulls'] else 'FAIL'}")
    print(f"  Metadata         : {'PASS' if val_status['metadata'] else 'FAIL'}")
    print()
    print(f"STATUS: {'SUCCESS' if val_status['all_passed'] else 'FAILED'}")
    print("=" * 60)

    return records, val_status


if __name__ == "__main__":
    generate_insurance_v2_embeddings()
