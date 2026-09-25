"""
embed_pipeline.py
=================
CivicSync Phase 2A - Embedding Pipeline Runner

Reads all six JSONL chunk files from data/processed/,
generates 768-dim embeddings using intfloat/multilingual-e5-base,
and writes output to data/embeddings/.

Usage:
    python src/embed_pipeline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — works when run as  python src/embed_pipeline.py  from root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embedder import (          # noqa: E402
    load_model,
    embed_chunks,
    read_chunks_jsonl,
    write_embeddings_jsonl,
    validate_embeddings,
    DEFAULT_MODEL,
    EMBEDDING_DIM,
    DEFAULT_BATCH,
)

# ---------------------------------------------------------------------------
# Domain mapping: input file -> output file
# ---------------------------------------------------------------------------
DOMAIN_FILES: list[tuple[str, str, str]] = [
    ("traffic",       "traffic_chunks.jsonl",       "traffic_embeddings.jsonl"),
    ("labour",        "labour_chunks.jsonl",         "labour_embeddings.jsonl"),
    ("women_safety",  "women_safety_chunks.jsonl",   "women_safety_embeddings.jsonl"),
    ("consumer",      "consumer_chunks.jsonl",       "consumer_embeddings.jsonl"),
    ("insurance",     "insurance_chunks.jsonl",      "insurance_embeddings.jsonl"),
    ("land_property", "land_property_chunks.jsonl",  "land_property_embeddings.jsonl"),
]


def _sep(char: str = "-", width: int = 56) -> str:
    return char * width


def run_embed_pipeline(
    processed_dir: Path | None = None,
    embeddings_dir: Path | None = None,
    batch_size: int = DEFAULT_BATCH,
    model_name: str = DEFAULT_MODEL,
) -> dict:
    """
    Full embedding pipeline.

    Returns
    -------
    dict with keys: total_chunks, total_embedded, dimension, domain_stats
    """
    if processed_dir is None:
        processed_dir = PROJECT_ROOT / "data" / "processed"
    if embeddings_dir is None:
        embeddings_dir = PROJECT_ROOT / "data" / "embeddings"

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(_sep("=", 56))
    print("  CivicSync — Phase 2A: Embedding Pipeline")
    print(_sep("=", 56))
    print(f"  Model      : {model_name}")
    print(f"  Batch size : {batch_size}")
    print(f"  Input dir  : {processed_dir}")
    print(f"  Output dir : {embeddings_dir}")
    print(_sep("-", 56))
    print("  Loading model...")
    t0 = time.perf_counter()
    model = load_model(model_name)
    print(f"  Model loaded in {time.perf_counter() - t0:.1f}s")
    print(_sep("-", 56))

    # ------------------------------------------------------------------
    # Process each domain
    # ------------------------------------------------------------------
    total_chunks   = 0
    total_embedded = 0
    domain_stats: list[dict] = []

    for domain, in_name, out_name in DOMAIN_FILES:
        in_path  = processed_dir  / in_name
        out_path = embeddings_dir / out_name

        if not in_path.exists():
            print(f"  [SKIP] {in_name} not found")
            continue

        # Read
        chunks = read_chunks_jsonl(in_path)
        n_chunks = len(chunks)

        # Embed
        t1 = time.perf_counter()
        records = embed_chunks(chunks, model, batch_size=batch_size, model_name=model_name)
        elapsed = time.perf_counter() - t1

        # Validate
        validate_embeddings(records, expected_count=n_chunks)

        # Write
        write_embeddings_jsonl(records, out_path)

        # Per-domain report
        print(f"  {domain}")
        print(f"    Chunks   : {n_chunks}")
        print(f"    Embedded : {len(records)}")
        print(f"    Dimension: {EMBEDDING_DIM}")
        print(f"    Time     : {elapsed:.2f}s")
        print(f"    Output   : {out_path.name}")
        print()

        total_chunks   += n_chunks
        total_embedded += len(records)
        domain_stats.append({
            "domain":    domain,
            "chunks":    n_chunks,
            "embedded":  len(records),
            "dimension": EMBEDDING_DIM,
            "output":    str(out_path),
            "size_bytes": out_path.stat().st_size,
        })

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(_sep("=", 56))
    print("  TOTAL")
    print(f"    Chunks     : {total_chunks}")
    print(f"    Embeddings : {total_embedded}")
    print(f"    Dimension  : {EMBEDDING_DIM}")
    print(_sep("-", 56))

    if total_chunks != total_embedded:
        raise RuntimeError(
            f"FATAL: chunk count ({total_chunks}) != "
            f"embedding count ({total_embedded})"
        )
    print("  [OK] All counts match")
    print(_sep("=", 56))

    return {
        "total_chunks":   total_chunks,
        "total_embedded": total_embedded,
        "dimension":      EMBEDDING_DIM,
        "domain_stats":   domain_stats,
    }


if __name__ == "__main__":
    result = run_embed_pipeline()

    # ------------------------------------------------------------------
    # Final display (as requested)
    # ------------------------------------------------------------------
    print()
    print(_sep("=", 56))
    print("  FILES CREATED")
    print(_sep("-", 56))
    for ds in result["domain_stats"]:
        size_kb = ds["size_bytes"] / 1024
        print(f"  {Path(ds['output']).name:<40} {size_kb:>8.1f} KB")

    print()
    print(_sep("=", 56))
    print("  SUMMARY")
    print(_sep("-", 56))
    print(f"  Total chunks     : {result['total_chunks']}")
    print(f"  Total embeddings : {result['total_embedded']}")
    print(f"  Embedding dim    : {result['dimension']}")
    print(_sep("=", 56))

    # Sample record from first output
    embeddings_dir = PROJECT_ROOT / "data" / "embeddings"
    first_output   = embeddings_dir / "traffic_embeddings.jsonl"
    if first_output.exists():
        import json
        with first_output.open("r", encoding="utf-8") as fh:
            sample = json.loads(fh.readline())
        print()
        print("  SAMPLE EMBEDDING RECORD")
        print(_sep("-", 56))
        print(f"  chunk_id    : {sample['chunk_id']}")
        print(f"  source_file : {sample['source_file']}")
        print(f"  domain      : {sample['domain']}")
        print(f"  page_start  : {sample['page_start']}")
        print(f"  page_end    : {sample['page_end']}")
        print(f"  model_name  : {sample['model_name']}")
        print(f"  dimensions  : {sample['dimensions']}")
        print(f"  embedding   : [{', '.join(f'{v:.6f}' for v in sample['embedding'][:5])} ...]")
        print(f"  text (first 80 chars): {sample['text'][:80]}...")
        print(_sep("=", 56))
