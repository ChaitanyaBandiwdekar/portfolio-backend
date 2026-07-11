"""Shared Supabase client construction (sync client).

Used by ingest.py and app/retrieval.py. Async concerns are Phase 6's.
"""

import os

_supabase_client = None


def get_supabase_client():
    global _supabase_client
    if _supabase_client is None:
        from supabase import create_client

        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_SERVICE_KEY"]
        _supabase_client = create_client(url, key)
    return _supabase_client
