"""
replace_insurance_v2.py
======================
CivicSync Phase 5B-3 — Insurance Replacement Pipeline

Replaces ONLY the insurance domain dataset in Supabase with the normalized V2 dataset.
Other domains (traffic, labour, consumer, women_safety, land_property) are untouched.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from supabase import create_client, Client

CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "insurance_v2" / "insurance_chunks_v2.jsonl"
EMBEDDINGS_FILE = PROJECT_ROOT / "data" / "embeddings" / "insurance_v2_embeddings.jsonl"

MODEL_NAME = "intfloat/multilingual-e5-base"
EXPECTED_OTHER_COUNTS = {
    "traffic": 8,
    "labour": 86,
    "consumer": 36,
    "women_safety": 32,
    "land_property": 184,
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL file into a list of dicts."""
    records = []
    with open(path, mode="r", encoding="utf-8") as fh:
        for line in fh:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))
    return records


def validate_local_inputs(
    chunks: List[Dict[str, Any]],
    embeddings: List[Dict[str, Any]],
    expected_count: int = 647,
    expected_dim: int = 768,
) -> Tuple[bool, str]:
    """
    Validate local chunks and embeddings before database modification.
    """
    if len(chunks) != expected_count:
        return False, f"Chunk count mismatch: got {len(chunks)}, expected {expected_count}"

    if len(embeddings) != expected_count:
        return False, f"Embedding count mismatch: got {len(embeddings)}, expected {expected_count}"

    chunk_map = {c.get("chunk_id"): c for c in chunks}

    for c in chunks:
        cid = c.get("chunk_id", "")
        if not cid.startswith("INS-V2-"):
            return False, f"Invalid chunk_id format: {cid}"
        if c.get("domain") != "insurance":
            return False, f"Chunk {cid} domain is not 'insurance'"
        if not c.get("source_file"):
            return False, f"Chunk {cid} missing source_file"
        if not (c.get("text") or "").strip():
            return False, f"Chunk {cid} text is empty"
        if c.get("page_start") is None or c.get("page_end") is None:
            return False, f"Chunk {cid} missing page numbers"

    for e in embeddings:
        cid = e.get("chunk_id", "")
        if not cid.startswith("INS-V2-"):
            return False, f"Invalid embedding chunk_id format: {cid}"
        if cid not in chunk_map:
            return False, f"Embedding chunk_id {cid} not found in chunks"

        vec = e.get("embedding")
        if vec is None or not isinstance(vec, list):
            return False, f"Embedding {cid} is None or not a list"
        if len(vec) != expected_dim:
            return False, f"Embedding {cid} dimension is {len(vec)}, expected {expected_dim}"
        if any(not isinstance(v, (int, float)) or math.isnan(v) for v in vec):
            return False, f"Embedding {cid} contains non-numeric/NaN values"

    return True, "Local validation passed"


def _query_with_retry(fn, max_retries: int = 5):
    """Execute query with retry on transient network connection errors."""
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_retries:
                raise RuntimeError(f"Query failed after {max_retries} attempts: {exc}") from exc
            time.sleep(2.0 * attempt)


def insert_chunks_idempotent(client: Client, new_doc_id: int, chunks: List[Dict[str, Any]], batch_size: int = 100) -> List[Dict[str, Any]]:
    """Idempotently insert chunk batches, handling retries safely."""
    chunk_records = []
    for idx, c in enumerate(chunks, 1):
        meta = dict(c.get("metadata", {}))
        meta["chunk_id"] = c["chunk_id"]
        meta["domain"] = "insurance"

        chunk_records.append({
            "document_id": new_doc_id,
            "chunk_index": idx,
            "content": c["text"],
            "page_start": c["page_start"],
            "page_end": c["page_end"],
            "metadata": meta,
        })

    for i in range(0, len(chunk_records), batch_size):
        batch = chunk_records[i : i + batch_size]
        start_idx = batch[0]["chunk_index"]
        end_idx = batch[-1]["chunk_index"]

        # Check existing
        existing = _query_with_retry(
            lambda: client.table("chunks")
            .select("id, chunk_index")
            .eq("document_id", new_doc_id)
            .gte("chunk_index", start_idx)
            .lte("chunk_index", end_idx)
            .execute()
        ).data or []

        if len(existing) == len(batch):
            continue

        if existing:
            _query_with_retry(
                lambda: client.table("chunks")
                .delete()
                .eq("document_id", new_doc_id)
                .gte("chunk_index", start_idx)
                .lte("chunk_index", end_idx)
                .execute()
            )

        _query_with_retry(lambda: client.table("chunks").insert(batch).execute())

    all_inserted = _query_with_retry(
        lambda: client.table("chunks")
        .select("*")
        .eq("document_id", new_doc_id)
        .order("chunk_index")
        .execute()
    ).data or []

    return all_inserted


def insert_embeddings_idempotent(client: Client, embedding_records: List[Dict[str, Any]], batch_size: int = 25) -> int:
    """Idempotently insert embedding batches, handling retries safely."""
    total_inserted = 0
    for i in range(0, len(embedding_records), batch_size):
        batch = embedding_records[i : i + batch_size]
        batch_cids = [b["chunk_id"] for b in batch]

        existing = _query_with_retry(
            lambda: client.table("embeddings")
            .select("id, chunk_id")
            .in_("chunk_id", batch_cids)
            .execute()
        ).data or []

        if len(existing) == len(batch):
            total_inserted += len(batch)
            continue

        if existing:
            _query_with_retry(
                lambda: client.table("embeddings")
                .delete()
                .in_("chunk_id", batch_cids)
                .execute()
            )

        res = _query_with_retry(lambda: client.table("embeddings").insert(batch).execute())
        total_inserted += len(res.data or [])

        if (i + batch_size) % 100 == 0 or (i + len(batch)) == len(embedding_records):
            print(f"  Progress: {min(i + batch_size, len(embedding_records))} / {len(embedding_records)} embeddings processed...")

    return total_inserted


def replace_insurance_data() -> Dict[str, Any]:
    """
    Main replacement flow:
    1. Local validation
    2. Locate old data
    3. Backup summary
    4. Delete old data
    5. Insert new document
    6. Insert 647 chunks
    7. Insert 647 embeddings
    8. Final validation
    """
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_SECRET_KEY", "").strip()

    if not url or not key:
        raise ValueError("SUPABASE_URL or SUPABASE_SECRET_KEY not set in .env")

    # Step 1: Local validation
    print("Step 1: Running local validation...")
    chunks = read_jsonl(CHUNKS_FILE)
    embeddings = read_jsonl(EMBEDDINGS_FILE)

    is_valid, msg = validate_local_inputs(chunks, embeddings)
    if not is_valid:
        raise RuntimeError(f"LOCAL VALIDATION FAILED: {msg}. ABORTING.")

    print("Local validation passed: 647 chunks and 647 embeddings verified.")

    client: Client = create_client(url, key)

    # Step 2: Locate old insurance data
    print("\nStep 2: Locating existing insurance data in Supabase...")
    doc_res = _query_with_retry(
        lambda: client.table("documents").select("id, filename, domain").eq("domain", "insurance").execute()
    )
    old_docs = doc_res.data or []
    old_doc_ids = [d["id"] for d in old_docs]

    old_chunks = []
    if old_doc_ids:
        ch_res = _query_with_retry(
            lambda: client.table("chunks").select("id, document_id").in_("document_id", old_doc_ids).execute()
        )
        old_chunks = ch_res.data or []

    old_chunk_ids = [c["id"] for c in old_chunks]

    old_embeddings = []
    if old_chunk_ids:
        for i in range(0, len(old_chunk_ids), 200):
            sub_cids = old_chunk_ids[i : i + 200]
            emb_res = _query_with_retry(
                lambda: client.table("embeddings").select("id, chunk_id").in_("chunk_id", sub_cids).execute()
            )
            old_embeddings.extend(emb_res.data or [])

    print(f"OLD INSURANCE DOCUMENTS : {len(old_docs)}")
    print(f"OLD INSURANCE CHUNKS    : {len(old_chunks)}")
    print(f"OLD INSURANCE EMBEDDINGS: {len(old_embeddings)}")

    # Step 3: Backup summary
    print("\nStep 3: Backup summary of data to be deleted:")
    print("OLD INSURANCE DATA")
    print("------------------")
    print(f"documents : {len(old_docs)}")
    print(f"chunks    : {len(old_chunks)}")
    print(f"embeddings: {len(old_embeddings)}")

    # Step 4: Delete existing insurance data
    print("\nStep 4: Deleting old insurance data...")
    if old_chunk_ids:
        for i in range(0, len(old_chunk_ids), 200):
            sub_cids = old_chunk_ids[i : i + 200]
            _query_with_retry(lambda: client.table("embeddings").delete().in_("chunk_id", sub_cids).execute())

    if old_doc_ids:
        for i in range(0, len(old_doc_ids), 50):
            sub_dids = old_doc_ids[i : i + 50]
            _query_with_retry(lambda: client.table("chunks").delete().in_("document_id", sub_dids).execute())
        _query_with_retry(lambda: client.table("documents").delete().in_("id", old_doc_ids).execute())

    # Verify deletion
    post_del_docs = _query_with_retry(
        lambda: client.table("documents").select("id").eq("domain", "insurance").execute().data or []
    )
    if post_del_docs:
        raise RuntimeError("Deletion failed: insurance documents still exist.")

    print("Deletion verified: 0 insurance documents, chunks, and embeddings remain.")

    # Step 5: Insert new document
    print("\nStep 5: Inserting new Insurance V2 document record...")
    new_doc_res = _query_with_retry(
        lambda: client.table("documents").insert({
            "filename": "civicsync_insurance_regulatory_v2",
            "domain": "insurance",
            "page_count": 0,
            "source_type": "regulatory",
        }).execute()
    )

    if not new_doc_res.data:
        raise RuntimeError("Failed to insert new insurance document")

    new_doc_id = new_doc_res.data[0]["id"]
    print(f"New document inserted with ID: {new_doc_id}")

    # Step 6: Insert 647 chunks idempotently
    print("\nStep 6: Inserting 647 normalized chunks into Supabase...")
    inserted_chunks = insert_chunks_idempotent(client, new_doc_id, chunks)

    if len(inserted_chunks) != len(chunks):
        raise RuntimeError(f"Inserted chunk count {len(inserted_chunks)} != expected {len(chunks)}")

    # Map INS-V2-xxxxxx to new DB chunk.id
    ins_v2_to_db_chunk_id: Dict[str, int] = {}
    for orig_chunk, db_chunk in zip(chunks, inserted_chunks):
        ins_v2_to_db_chunk_id[orig_chunk["chunk_id"]] = db_chunk["id"]

    print(f"Successfully inserted {len(inserted_chunks)} chunks.")

    # Step 7: Insert 647 embeddings idempotently
    print("\nStep 7: Inserting 647 vector embeddings into Supabase (batch_size=25)...")
    embedding_records_to_insert = []
    for e in embeddings:
        cid = e["chunk_id"]
        db_cid = ins_v2_to_db_chunk_id[cid]
        embedding_records_to_insert.append({
            "chunk_id": db_cid,
            "model_name": MODEL_NAME,
            "embedding": e["embedding"],
        })

    n_inserted_embs = insert_embeddings_idempotent(client, embedding_records_to_insert, batch_size=25)
    print(f"Successfully processed {n_inserted_embs} embeddings.")

    # Step 8: Final validation
    print("\nStep 8: Running final database validation...")

    # Check insurance doc count
    ins_docs = _query_with_retry(
        lambda: client.table("documents").select("id").eq("domain", "insurance").execute().data or []
    )
    if len(ins_docs) != 1:
        raise RuntimeError(f"Insurance doc count mismatch: got {len(ins_docs)}, expected 1")

    # Check insurance chunks
    ins_chunks = _query_with_retry(
        lambda: client.table("chunks").select("id").eq("document_id", new_doc_id).execute().data or []
    )
    if len(ins_chunks) != 647:
        raise RuntimeError(f"Insurance chunk count mismatch: got {len(ins_chunks)}, expected 647")

    ins_chunk_ids = [c["id"] for c in ins_chunks]

    # Check insurance embeddings in sub-queries
    total_ins_embs = 0
    for i in range(0, len(ins_chunk_ids), 200):
        sub_ids = ins_chunk_ids[i : i + 200]
        ins_embs_batch = _query_with_retry(
            lambda: client.table("embeddings").select("id, chunk_id").in_("chunk_id", sub_ids).execute().data or []
        )
        total_ins_embs += len(ins_embs_batch)

    if total_ins_embs != 647:
        raise RuntimeError(f"Insurance embedding count mismatch: got {total_ins_embs}, expected 647")

    # Check other domains
    other_counts: Dict[str, int] = {}
    for dom, expected_c in EXPECTED_OTHER_COUNTS.items():
        doc_q = _query_with_retry(
            lambda: client.table("documents").select("id").eq("domain", dom).execute().data or []
        )
        if not doc_q:
            raise RuntimeError(f"Missing document for domain {dom}")
        d_id = doc_q[0]["id"]
        c_q = _query_with_retry(
            lambda: client.table("chunks").select("id", count="exact").eq("document_id", d_id).execute()
        )
        count = len(c_q.data or [])
        other_counts[dom] = count
        if count != expected_c:
            raise RuntimeError(f"Domain '{dom}' count changed: got {count}, expected {expected_c}")

    total_all_chunks = sum(other_counts.values()) + len(ins_chunks)
    if total_all_chunks != 993:
        raise RuntimeError(f"Total chunk count mismatch: got {total_all_chunks}, expected 993")

    # Final summary report printout
    print("\n" + "=" * 60)
    print("CivicSync Phase 5B-3 — Insurance Replacement")
    print("=" * 60)
    print("Old insurance:")
    print(f"  Documents  : {len(old_docs)}")
    print(f"  Chunks     : {len(old_chunks)}")
    print(f"  Embeddings : {len(old_embeddings)}")
    print()
    print("New insurance:")
    print("  Documents  : 1")
    print("  Chunks     : 647")
    print("  Embeddings : 647")
    print()
    print("Vector dimension: 768")
    print()
    print("Other domains:")
    print(f"  Traffic       : {other_counts.get('traffic', 0)}")
    print(f"  Labour        : {other_counts.get('labour', 0)}")
    print(f"  Consumer      : {other_counts.get('consumer', 0)}")
    print(f"  Women Safety  : {other_counts.get('women_safety', 0)}")
    print(f"  Land Property : {other_counts.get('land_property', 0)}")
    print()
    print("STATUS: SUCCESS")
    print("=" * 60)

    return {
        "old_docs": len(old_docs),
        "old_chunks": len(old_chunks),
        "old_embeddings": len(old_embeddings),
        "new_docs": 1,
        "new_chunks": 647,
        "new_embeddings": 647,
        "other_counts": other_counts,
        "total_chunks": total_all_chunks,
    }


if __name__ == "__main__":
    replace_insurance_data()
