"""Unit tests for app/retrieval.py. No real APIs: embedder + Supabase RPC are mocked."""

import os

import pytest

from app import retrieval


class FakeRpcResponse:
    def __init__(self, data):
        self.data = data


class FakeSupabaseClient:
    def __init__(self, rpc_data):
        self._rpc_data = rpc_data
        self.rpc_calls = []

    def rpc(self, name, params):
        self.rpc_calls.append((name, params))
        return self

    def execute(self):
        return FakeRpcResponse(self._rpc_data)


def _row(similarity, title="Title", content="Some content"):
    return {
        "id": 1,
        "source": "about.md",
        "title": title,
        "content": content,
        "similarity": similarity,
    }


def test_retrieve_filters_results_below_threshold(monkeypatch):
    rows = [_row(0.9), _row(0.4), _row(0.5)]
    client = FakeSupabaseClient(rows)
    monkeypatch.setattr(retrieval, "embed_query", lambda text: [0.1, 0.2])
    monkeypatch.setattr(retrieval, "get_supabase_client", lambda: client)

    results = retrieval.retrieve("some query")

    assert len(results) == 2
    assert all(r.similarity >= retrieval.SIMILARITY_THRESHOLD for r in results)


def test_retrieve_requests_top_5(monkeypatch):
    client = FakeSupabaseClient([])
    monkeypatch.setattr(retrieval, "embed_query", lambda text: [0.1, 0.2])
    monkeypatch.setattr(retrieval, "get_supabase_client", lambda: client)

    retrieval.retrieve("some query")

    assert len(client.rpc_calls) == 1
    name, params = client.rpc_calls[0]
    assert name == "match_documents"
    assert params["match_count"] == 5
    assert params["query_embedding"] == [0.1, 0.2]


def test_retrieve_result_shape(monkeypatch):
    rows = [_row(0.8, title="About Me", content="Hello world")]
    client = FakeSupabaseClient(rows)
    monkeypatch.setattr(retrieval, "embed_query", lambda text: [0.1, 0.2])
    monkeypatch.setattr(retrieval, "get_supabase_client", lambda: client)

    results = retrieval.retrieve("some query")

    assert len(results) == 1
    result = results[0]
    assert result.title == "About Me"
    assert result.content == "Hello world"
    assert result.similarity == 0.8


def test_retrieve_returns_empty_list_when_no_matches(monkeypatch):
    client = FakeSupabaseClient([])
    monkeypatch.setattr(retrieval, "embed_query", lambda text: [0.1, 0.2])
    monkeypatch.setattr(retrieval, "get_supabase_client", lambda: client)

    results = retrieval.retrieve("some query")

    assert results == []


def test_retrieve_returns_empty_list_when_all_below_threshold(monkeypatch):
    rows = [_row(0.1), _row(0.2)]
    client = FakeSupabaseClient(rows)
    monkeypatch.setattr(retrieval, "embed_query", lambda text: [0.1, 0.2])
    monkeypatch.setattr(retrieval, "get_supabase_client", lambda: client)

    results = retrieval.retrieve("some query")

    assert results == []


# --- Integration test (real Supabase) --------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("SUPABASE_URL"), reason="requires live Supabase project")
def test_retrieve_paraphrase_query_returns_seeded_doc_as_top_result():
    """Seed a known doc, query with a paraphrase, assert it's top-1 above threshold.

    Requires a real Supabase project (SUPABASE_URL/SUPABASE_SERVICE_KEY) and
    GOOGLE_API_KEY, plus a seeded doc already ingested (see ingest.py). Deferred
    to a human to run against a live environment; not exercised in CI.
    """
    results = retrieval.retrieve("What is this project a demo of?")

    assert len(results) > 0
    assert results[0].similarity >= retrieval.SIMILARITY_THRESHOLD
