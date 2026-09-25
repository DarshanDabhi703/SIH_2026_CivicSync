"""
ingest_traffic.py
=================
CivicSync Phase 2B — Complete Fresh Supabase Ingestion

Ingests all 358 chunks and embeddings across 6 domains into:
  public.documents  (6 rows)
  public.chunks     (358 rows)
  public.embeddings (358 rows)

Schema (confirmed):
  documents : id, filename, domain, page_count, source_type, created_at
  chunks    : id, document_id, chunk_index, content, page_start, page_end, metadata, created_at
  embeddings: id, chunk_id, model_name, embedding, created_at

Idempotent: existing records are detected and skipped, never duplicated.
Does NOT regenerate embeddings — uses vectors stored in local JSONL files.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from supabase import create_client, Client

# ---------------------------------------------------------------------------
# Env check
# ---------------------------------------------------------------------------
SUPABASE_URL        = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY", "").strip()

if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
    print("[ERROR] SUPABASE_URL or SUPABASE_SECRET_KEY not set in .env")
    sys.exit(1)

MODEL_NAME   = "intfloat/multilingual-e5-base"
SOURCE_TYPE  = "legal_dataset"

# ---------------------------------------------------------------------------
# Domain configuration (page_counts provided by user)
# ---------------------------------------------------------------------------
DOMAIN_CONFIGS = [
    {
        "domain":         "traffic",
        "pdf_filename":   "india_traffic_rules_formatted.pdf",
        "page_count":     9,
        "chunks_file":    "traffic_chunks.jsonl",
        "embeddings_file":"traffic_embeddings.jsonl",
        "expected":       8,
    },
    {
        "domain":         "labour",
        "pdf_filename":   "Indian_Labour_Rights_ML_Dataset_v5 (1).pdf",
        "page_count":     45,
        "chunks_file":    "labour_chunks.jsonl",
        "embeddings_file":"labour_embeddings.jsonl",
        "expected":       86,
    },
    {
        "domain":         "women_safety",
        "pdf_filename":   "Women Safety and Rights.pdf",
        "page_count":     41,
        "chunks_file":    "women_safety_chunks.jsonl",
        "embeddings_file":"women_safety_embeddings.jsonl",
        "expected":       32,
    },
    {
        "domain":         "consumer",
        "pdf_filename":   "consumer_rights_formatted.pdf",
        "page_count":     34,
        "chunks_file":    "consumer_chunks.jsonl",
        "embeddings_file":"consumer_embeddings.jsonl",
        "expected":       36,
    },
    {
        "domain":         "insurance",
        "pdf_filename":   "india_insurance_financial_rights_dataset_formatted.pdf",
        "page_count":     30,
        "chunks_file":    "insurance_chunks.jsonl",
        "embeddings_file":"insurance_embeddings.jsonl",
        "expected":       12,
    },
    {
        "domain":         "land_property",
        "pdf_filename":   "Indian_Land_Property_Tenancy_ML_Dataset.pdf",
        "page_count":     49,
        "chunks_file":    "land_property_chunks.jsonl",
        "embeddings_file":"land_property_embeddings.jsonl",
        "expected":       184,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    return records


def parse_vector(raw) -> list[float]:
    """Parse embedding from string or list and validate dimension."""
    vec = json.loads(raw) if isinstance(raw, str) else raw
    if vec is None:
        raise ValueError("Null embedding")
    if len(vec) != 768:
        raise ValueError(f"Embedding dimension {len(vec)} != 768")
    return vec


# ---------------------------------------------------------------------------
# Per-domain ingestion
# ---------------------------------------------------------------------------

def ingest_domain(client: Client, cfg: dict) -> tuple[int, int, int, int]:
    """
    Ingest one domain. Returns (chunks_inserted, chunks_skipped,
    embeddings_inserted, embeddings_skipped).
    """
    domain   = cfg["domain"]
    proc_path = PROJECT_ROOT / "data" / "processed" / cfg["chunks_file"]
    emb_path  = PROJECT_ROOT / "data" / "embeddings" / cfg["embeddings_file"]

    if not proc_path.exists():
        raise FileNotFoundError(f"Missing processed file: {proc_path}")
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {emb_path}")

    processed  = {c["chunk_id"]: c for c in read_jsonl(proc_path)}
    emb_records = read_jsonl(emb_path)

    if len(emb_records) != cfg["expected"]:
        raise ValueError(
            f"{domain}: expected {cfg['expected']} records, "
            f"got {len(emb_records)}"
        )

    # --- Document: find or create ---
    doc_q = (
        client.table("documents")
        .select("id")
        .eq("filename", cfg["pdf_filename"])
        .execute()
    )
    if doc_q.data:
        doc_id = doc_q.data[0]["id"]
        print(f"  [{domain}] Document ID {doc_id} already exists — reusing")
    else:
        res = client.table("documents").insert({
            "filename":    cfg["pdf_filename"],
            "domain":      domain,
            "page_count":  cfg["page_count"],
            "source_type": SOURCE_TYPE,
        }).execute()
        if not res.data:
            raise RuntimeError(f"{domain}: failed to insert document")
        doc_id = res.data[0]["id"]
        print(f"  [{domain}] Created document ID {doc_id}")

    # --- Chunks + Embeddings ---
    ch_ins = ch_skip = emb_ins = emb_skip = 0

    for idx, rec in enumerate(emb_records):
        orig     = processed.get(rec["chunk_id"], {})
        metadata = orig.get("metadata", {"domain": domain})

        # Chunk
        chunk_q = (
            client.table("chunks")
            .select("id")
            .eq("document_id", doc_id)
            .eq("chunk_index", idx)
            .execute()
        )
        if chunk_q.data:
            db_chunk_id = chunk_q.data[0]["id"]
            ch_skip += 1
        else:
            cr = client.table("chunks").insert({
                "document_id": doc_id,
                "chunk_index": idx,
                "content":     rec["text"],
                "page_start":  rec["page_start"],
                "page_end":    rec["page_end"],
                "metadata":    metadata,
            }).execute()
            if not cr.data:
                raise RuntimeError(f"{domain}: chunk insert failed at index {idx}")
            db_chunk_id = cr.data[0]["id"]
            ch_ins += 1

        # Embedding
        emb_q = (
            client.table("embeddings")
            .select("id")
            .eq("chunk_id", db_chunk_id)
            .execute()
        )
        if emb_q.data:
            emb_skip += 1
        else:
            er = client.table("embeddings").insert({
                "chunk_id":   db_chunk_id,
                "model_name": MODEL_NAME,
                "embedding":  rec["embedding"],
            }).execute()
            if not er.data:
                raise RuntimeError(f"{domain}: embedding insert failed for chunk {db_chunk_id}")
            emb_ins += 1

    print(
        f"  [{domain}] chunks inserted={ch_ins} skipped={ch_skip} | "
        f"embeddings inserted={emb_ins} skipped={emb_skip}"
    )
    return ch_ins, ch_skip, emb_ins, emb_skip


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_all(client: Client) -> dict[str, tuple[int, int]]:
    """Query Supabase and confirm every domain's counts match expectations."""
    results: dict[str, tuple[int, int]] = {}

    doc_res = (
        client.table("documents")
        .select("id, domain")
        .execute()
    )
    domain_to_doc_id = {row["domain"]: row["id"] for row in doc_res.data}

    for cfg in DOMAIN_CONFIGS:
        dom    = cfg["domain"]
        doc_id = domain_to_doc_id.get(dom)
        if not doc_id:
            raise ValueError(f"Validation: document for domain '{dom}' not found")

        # Chunks
        ch_res = (
            client.table("chunks")
            .select("id, content, document_id")
            .eq("document_id", doc_id)
            .execute()
        )
        n_chunks = len(ch_res.data)
        if n_chunks != cfg["expected"]:
            raise ValueError(
                f"Validation: {dom} expected {cfg['expected']} chunks, got {n_chunks}"
            )
        for chk in ch_res.data:
            if not chk["content"]:
                raise ValueError(f"Validation: chunk {chk['id']} has empty content")
            if chk["document_id"] != doc_id:
                raise ValueError(f"Validation: chunk {chk['id']} has wrong document_id")

        chunk_ids = [c["id"] for c in ch_res.data]

        # Embeddings
        emb_res = (
            client.table("embeddings")
            .select("id, chunk_id, embedding")
            .in_("chunk_id", chunk_ids)
            .execute()
        )
        n_embs = len(emb_res.data)
        if n_embs != cfg["expected"]:
            raise ValueError(
                f"Validation: {dom} expected {cfg['expected']} embeddings, got {n_embs}"
            )
        emb_chunk_ids = [e["chunk_id"] for e in emb_res.data]
        if sorted(emb_chunk_ids) != sorted(chunk_ids):
            raise ValueError(f"Validation: {dom} embedding/chunk ID mismatch")

        for emb in emb_res.data:
            vec = parse_vector(emb["embedding"])
            if any(not isinstance(v, (int, float)) for v in vec):
                raise ValueError(f"Validation: non-numeric embedding for chunk {emb['chunk_id']}")

        results[dom] = (n_chunks, n_embs)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    client: Client = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)

    print("=" * 56)
    print("  CivicSync — Complete Fresh Supabase Ingestion")
    print("=" * 56)

    total_ch_ins = total_ch_skip = total_emb_ins = total_emb_skip = 0

    for cfg in DOMAIN_CONFIGS:
        print(f"\nProcessing: {cfg['domain']} ({cfg['expected']} records)")
        ci, cs, ei, es = ingest_domain(client, cfg)
        total_ch_ins  += ci;  total_ch_skip  += cs
        total_emb_ins += ei;  total_emb_skip += es

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------
    print("\nValidating all domains...")
    val = validate_all(client)

    print()
    print(f"{'Domain':<15} {'Chunks':>8} {'Embeddings':>12}")
    print("-" * 37)
    total_chunks = total_embs = 0
    for cfg in DOMAIN_CONFIGS:
        dom = cfg["domain"]
        ch, em = val[dom]
        print(f"{dom:<15} {ch:>8} {em:>12}")
        total_chunks += ch
        total_embs   += em
    print("-" * 37)
    print(f"{'TOTAL':<15} {total_chunks:>8} {total_embs:>12}")

    # Grand totals
    doc_count = client.table("documents").select("id", count="exact").execute()
    n_docs    = len(doc_count.data)

    print()
    print(f"Documents:  {n_docs}")
    print(f"Chunks:     {total_chunks}")
    print(f"Embeddings: {total_embs}")
    print()
    print(f"New chunks inserted:     {total_ch_ins}")
    print(f"Chunks skipped (exist):  {total_ch_skip}")
    print(f"New embeddings inserted: {total_emb_ins}")
    print(f"Embeddings skipped:      {total_emb_skip}")
    print()

    assert n_docs       == 6,   f"Expected 6 documents, got {n_docs}"
    assert total_chunks == 358, f"Expected 358 chunks, got {total_chunks}"
    assert total_embs   == 358, f"Expected 358 embeddings, got {total_embs}"

    print("All assertions passed.")
    print("Status: SUCCESS")
    print("=" * 56)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[FATAL] {exc}")
        sys.exit(1)
