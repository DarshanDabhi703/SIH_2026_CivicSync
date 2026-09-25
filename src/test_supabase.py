"""
test_supabase.py
================
CivicSync — Supabase connection test.

Verifies that the Python backend can reach Supabase and query the
public.documents table.  No data is inserted, updated, or deleted.

Usage:
    python src/test_supabase.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
except ImportError:
    print("[ERROR] python-dotenv is not installed.")
    print("        Run:  pip install python-dotenv")
    sys.exit(1)

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

import os  # noqa: E402  (must come after load_dotenv)

# ---------------------------------------------------------------------------
# Read required variables
# ---------------------------------------------------------------------------
SUPABASE_URL        = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY", "").strip()

url_ok    = bool(SUPABASE_URL)
secret_ok = bool(SUPABASE_SECRET_KEY)

if not url_ok:
    print("[ERROR] SUPABASE_URL is missing or empty in .env")
if not secret_ok:
    print("[ERROR] SUPABASE_SECRET_KEY is missing or empty in .env")
if not url_ok or not secret_ok:
    sys.exit(1)

# ---------------------------------------------------------------------------
# Create Supabase client
# ---------------------------------------------------------------------------
try:
    from supabase import create_client, Client
except ImportError:
    print("[ERROR] supabase-py is not installed.")
    print("        Run:  pip install supabase")
    sys.exit(1)

try:
    client: Client = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
except Exception as exc:
    print(f"[ERROR] Failed to create Supabase client: {exc}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Harmless SELECT — 1 row from public.documents
# ---------------------------------------------------------------------------
try:
    response = (
        client
        .table("documents")
        .select("*")
        .limit(1)
        .execute()
    )
except Exception as exc:
    print(f"[ERROR] Database query failed: {exc}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Report — never print the secret key
# ---------------------------------------------------------------------------
print("Supabase connection successful")
print(f"URL configured: {url_ok}")
print(f"Secret configured: {secret_ok}")
print("Database query successful")
