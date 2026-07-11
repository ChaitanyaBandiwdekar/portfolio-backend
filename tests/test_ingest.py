"""Unit tests for ingest.py. No real APIs: embeddings and Supabase are faked."""

import ingest
from app import llm


# --- Chunking ----------------------------------------------------------


def test_chunk_no_headings_returns_single_chunk_with_no_heading_path():
    chunks = ingest.chunk_markdown("Just a short line of text, no headings.")
    assert len(chunks) == 1
    assert chunks[0]["heading_path"] == ""
    assert chunks[0]["text"] == "Just a short line of text, no headings."


def test_chunk_short_doc_under_chunk_size_stays_one_chunk():
    md = "# Title\n\nA short paragraph well under the 2000 char chunk size."
    chunks = ingest.chunk_markdown(md)
    assert len(chunks) == 1
    assert chunks[0]["heading_path"] == "Title"
    assert chunks[0]["text"].startswith("Title\n\n")


def test_chunk_long_section_splits_with_overlap():
    # One section with 5000 chars of content should split into multiple
    # RecursiveCharacterTextSplitter chunks (chunk_size=2000, overlap=200),
    # and consecutive chunks should share overlapping text.
    body = "".join(f"sentence number {i}. " for i in range(300))
    assert len(body) > 4000
    md = f"# Title\n\n{body}"
    chunks = ingest.chunk_markdown(md)
    assert len(chunks) > 1
    # every chunk carries the heading path prefix
    for c in chunks:
        assert c["heading_path"] == "Title"
        assert c["text"].startswith("Title\n\n")
    # overlap: the tail of chunk N should reappear at the head of chunk N+1's body
    first_body = chunks[0]["text"][len("Title\n\n") :]
    second_body = chunks[1]["text"][len("Title\n\n") :]
    tail = first_body[-100:]
    assert tail in second_body


def test_chunk_heading_path_includes_nested_headers():
    md = "# H1\n\n## H2\n\n### H3\n\nDeep content.\n"
    chunks = ingest.chunk_markdown(md)
    assert chunks[-1]["heading_path"] == "H1 > H2 > H3"
    assert chunks[-1]["text"] == "H1 > H2 > H3\n\nDeep content."


# --- Hashing -------------------------------------------------------------


def test_content_hash_is_deterministic():
    h1 = ingest.content_hash("about.md", "some chunk text")
    h2 = ingest.content_hash("about.md", "some chunk text")
    assert h1 == h2


def test_content_hash_differs_by_source_or_text():
    base = ingest.content_hash("about.md", "same text")
    assert base != ingest.content_hash("projects.md", "same text")
    assert base != ingest.content_hash("about.md", "different text")


def test_build_rows_hash_matches_content_hash_of_final_content():
    rows = ingest.build_rows("about.md", "About", "# About\n\nHello there.")
    for row in rows:
        assert row["content_hash"] == ingest.content_hash("about.md", row["content"])


# --- Stale hash diffing ----------------------------------------------------


def test_stale_hashes_returns_only_hashes_missing_from_current_run():
    existing = {"a", "b", "c"}
    current = {"b", "c", "d"}
    assert ingest.stale_hashes(existing, current) == {"a"}


def test_stale_hashes_empty_when_nothing_removed():
    existing = {"a", "b"}
    current = {"a", "b", "c"}
    assert ingest.stale_hashes(existing, current) == set()


# --- title_from_markdown ---------------------------------------------------


def test_title_from_markdown_finds_first_h1():
    assert ingest.title_from_markdown("intro line\n# The Title\nmore") == "The Title"


def test_title_from_markdown_none_when_no_h1():
    assert ingest.title_from_markdown("no headings here") is None


# --- Fake Supabase client for ingest_file / ingest_docs tests --------------


class FakeTable:
    def __init__(self, store, name):
        self.store = store
        self.name = name
        self._filters = []
        self._select_cols = None
        self._delete_hashes = None

    def select(self, cols):
        self._select_cols = cols
        return self

    def eq(self, col, val):
        self._filters.append((col, val))
        return self

    def in_(self, col, values):
        self._delete_hashes = set(values)
        return self

    def upsert(self, rows, on_conflict):
        assert on_conflict == "content_hash"
        for row in rows:
            self.store[row["content_hash"]] = dict(row)
        return self

    def delete(self):
        return self

    def execute(self):
        if self._delete_hashes is not None:
            for h in list(self._delete_hashes):
                # only delete rows matching the eq() filters (source)
                row = self.store.get(h)
                if row and all(row.get(k) == v for k, v in self._filters):
                    del self.store[h]
            return _Result([])
        if self._select_cols is not None:
            rows = [
                {"content_hash": h}
                for h, row in self.store.items()
                if all(row.get(k) == v for k, v in self._filters)
            ]
            return _Result(rows)
        return _Result(list(self.store.values()))


class _Result:
    def __init__(self, data):
        self.data = data


class FakeSupabaseClient:
    def __init__(self):
        self.store = {}

    def table(self, name):
        return FakeTable(self.store, name)


def test_ingest_file_upserts_rows_into_fake_client(monkeypatch):
    monkeypatch.setattr(ingest, "embed_chunks", lambda texts: [[0.1, 0.2]] * len(texts))
    client = FakeSupabaseClient()
    rows = ingest.build_rows("about.md", "About", "# About\n\nHello.")
    ingest.ingest_file("about.md", rows, client)
    assert len(client.store) == len(rows)
    for row in client.store.values():
        assert row["embedding"] == [0.1, 0.2]


def test_ingest_file_deletes_stale_rows_after_edit(monkeypatch):
    monkeypatch.setattr(ingest, "embed_chunks", lambda texts: [[0.1, 0.2]] * len(texts))
    client = FakeSupabaseClient()

    original_rows = ingest.build_rows("about.md", "About", "# About\n\nOriginal content.")
    ingest.ingest_file("about.md", original_rows, client)
    original_hashes = {r["content_hash"] for r in original_rows}
    assert original_hashes <= set(client.store.keys())

    edited_rows = ingest.build_rows("about.md", "About", "# About\n\nEdited content now.")
    ingest.ingest_file("about.md", edited_rows, client)
    edited_hashes = {r["content_hash"] for r in edited_rows}

    # old hash is gone, new hash present
    assert not (original_hashes - edited_hashes) & set(client.store.keys())
    assert edited_hashes <= set(client.store.keys())


def test_ingest_docs_reads_docs_dir_and_calls_ingest_file(monkeypatch, tmp_path):
    (tmp_path / "one.md").write_text("# One\n\nContent one.", encoding="utf-8")
    (tmp_path / "two.md").write_text("# Two\n\nContent two.", encoding="utf-8")

    monkeypatch.setattr(ingest, "embed_chunks", lambda texts: [[0.1, 0.2]] * len(texts))
    fake_client = FakeSupabaseClient()
    monkeypatch.setattr(ingest, "get_supabase_client", lambda: fake_client)

    total = ingest.ingest_docs(docs_dir=tmp_path)

    assert total == 2
    assert len(fake_client.store) == 2


# --- Retry / grouping (embed_chunks, _embed_with_retry) ---------------------


def test_embed_with_retry_succeeds_after_429s_with_backoff(monkeypatch):
    calls = {"n": 0}

    def flaky_embed(texts):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("429 RESOURCE_EXHAUSTED: quota")
        return [[1.0, 0.0]] * len(texts)

    sleeps = []
    monkeypatch.setattr(ingest.llm, "embed_documents", flaky_embed)
    monkeypatch.setattr(ingest.time, "sleep", sleeps.append)

    result = ingest._embed_with_retry(["a", "b"])

    assert result == [[1.0, 0.0], [1.0, 0.0]]
    assert calls["n"] == 3
    assert sleeps == [1.0, 2.0]  # exponential backoff between retries


def test_embed_with_retry_raises_when_retries_exhausted(monkeypatch):
    def always_429(texts):
        raise RuntimeError("429 RESOURCE_EXHAUSTED: quota")

    monkeypatch.setattr(ingest.llm, "embed_documents", always_429)
    monkeypatch.setattr(ingest.time, "sleep", lambda s: None)

    try:
        ingest._embed_with_retry(["a"])
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "429" in str(exc)


def test_embed_with_retry_reraises_non_429_immediately(monkeypatch):
    calls = {"n": 0}

    def broken_embed(texts):
        calls["n"] += 1
        raise ValueError("bad request")

    sleeps = []
    monkeypatch.setattr(ingest.llm, "embed_documents", broken_embed)
    monkeypatch.setattr(ingest.time, "sleep", sleeps.append)

    try:
        ingest.embed_chunks(["a"])
        assert False, "expected ValueError"
    except ValueError:
        pass
    assert calls["n"] == 1  # no retry for non-rate-limit errors
    assert sleeps == []


def test_embed_chunks_groups_of_five_with_inter_group_sleeps(monkeypatch):
    groups = []

    def record_embed(texts):
        groups.append(list(texts))
        return [[1.0, 0.0]] * len(texts)

    sleeps = []
    monkeypatch.setattr(ingest.llm, "embed_documents", record_embed)
    monkeypatch.setattr(ingest.time, "sleep", sleeps.append)

    texts = [f"t{i}" for i in range(12)]
    vectors = ingest.embed_chunks(texts)

    assert len(vectors) == 12
    assert [len(g) for g in groups] == [5, 5, 2]
    assert sleeps == [ingest.EMBED_SLEEP_SECONDS] * 2  # between groups, not after last


# --- llm.py embedding wrapper: L2 normalization -----------------------------


def test_embed_query_l2_normalizes(monkeypatch):
    class FakeEmbedder:
        def embed_query(self, text):
            return [3.0, 4.0]  # norm = 5

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(llm, "_query_embedder", None)
    monkeypatch.setattr(llm, "_get_query_embedder", lambda: FakeEmbedder())

    result = llm.embed_query("hello")
    assert result == [0.6, 0.8]


def test_embed_documents_l2_normalizes_each_vector(monkeypatch):
    class FakeEmbedder:
        def embed_documents(self, texts):
            return [[3.0, 4.0], [0.0, 5.0]]

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(llm, "_doc_embedder", None)
    monkeypatch.setattr(llm, "_get_doc_embedder", lambda: FakeEmbedder())

    result = llm.embed_documents(["a", "b"])
    assert result == [[0.6, 0.8], [0.0, 1.0]]


def test_embed_query_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(llm, "_query_embedder", None)
    try:
        llm.embed_query("hello")
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "GOOGLE_API_KEY" in str(exc)


def test_import_llm_module_does_not_require_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    import importlib

    importlib.reload(llm)  # must not raise
