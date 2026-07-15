"""Phase 7 security-hardening tests: body size cap, non-JSON rejection,
CORS origin lockdown, and /docs exposure toggled by ENV.

CORS and /docs tests build a fresh app via app.main.create_app() under
monkeypatched env, since allow_origins/docs_url are read once at app
creation time (see app/main.py's create_app docstring).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.main import MAX_BODY_BYTES, app, get_agent, get_supabase


def _fake_supabase():
    fake_client = MagicMock()
    fake_client.table.return_value.select.return_value.limit.return_value.execute.return_value = None
    return fake_client


class _FakeAgent:
    async def astream(self, *args, **kwargs):
        return
        yield  # pragma: no cover - makes this an async generator


@pytest.fixture(autouse=True)
def _reset_state():
    from app import guardrails as guardrails_module

    guardrails_module.reset_rate_limits()
    guardrails_module._daily_count = 0
    yield
    app.dependency_overrides.clear()
    guardrails_module.reset_rate_limits()


@pytest.fixture
def client():
    return TestClient(app)


# --- non-JSON body ----------------------------------------------------------


def test_non_json_body_rejected(client):
    app.dependency_overrides[get_agent] = lambda: _FakeAgent()
    resp = client.post(
        "/chat", content=b"not json", headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 422


# --- oversized payload -------------------------------------------------------


def test_oversized_payload_rejected(client):
    app.dependency_overrides[get_agent] = lambda: _FakeAgent()
    big_text = "a" * (MAX_BODY_BYTES + 1000)
    body = json.dumps({"messages": [{"role": "user", "content": big_text}]})

    resp = client.post(
        "/chat", content=body, headers={"Content-Type": "application/json"}
    )

    assert resp.status_code == 413


def test_malformed_content_length_returns_400_not_500(client):
    app.dependency_overrides[get_agent] = lambda: _FakeAgent()
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()

    resp = client.post(
        "/chat",
        content=body,
        headers={"Content-Type": "application/json", "Content-Length": "not-a-number"},
    )

    assert resp.status_code == 400


def test_small_payload_not_rejected_by_size_cap(client):
    # A well-under-cap request should not be rejected for size (may still
    # take other paths, e.g. greeting fast path) — asserts we're not too
    # aggressive.
    app.dependency_overrides[get_agent] = lambda: _FakeAgent()
    resp = client.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200


# --- security headers --------------------------------------------------------


def test_security_headers_present(client):
    app.dependency_overrides[get_supabase] = _fake_supabase

    resp = client.get("/health")
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["x-frame-options"] == "DENY"
    assert resp.headers["referrer-policy"] == "no-referrer"


# --- CORS ---------------------------------------------------------------------


def test_cors_preflight_disallowed_origin_rejected(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://portfolio.example.com")
    test_app = main_module.create_app()
    test_client = TestClient(test_app)

    resp = test_client.options(
        "/chat",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert resp.status_code == 400
    assert resp.text == "Disallowed CORS origin"


def test_cors_simple_request_disallowed_origin_no_acao_header(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://portfolio.example.com")
    test_app = main_module.create_app()
    test_app.dependency_overrides[get_supabase] = _fake_supabase
    test_client = TestClient(test_app)

    resp = test_client.get("/health", headers={"Origin": "https://evil.example.com"})

    assert resp.status_code == 200
    assert "access-control-allow-origin" not in resp.headers


def test_cors_allowed_origin_gets_acao_header(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://portfolio.example.com")
    test_app = main_module.create_app()
    test_app.dependency_overrides[get_supabase] = _fake_supabase
    test_client = TestClient(test_app)

    resp = test_client.get(
        "/health", headers={"Origin": "https://portfolio.example.com"}
    )

    assert resp.status_code == 200
    assert (
        resp.headers["access-control-allow-origin"] == "https://portfolio.example.com"
    )


# --- docs disabled in production ----------------------------------------------


def test_docs_disabled_in_production(monkeypatch):
    monkeypatch.setenv("ENV", "production")
    prod_app = main_module.create_app()
    prod_client = TestClient(prod_app)

    resp = prod_client.get("/docs")

    assert resp.status_code == 404


def test_docs_available_in_development(monkeypatch):
    monkeypatch.setenv("ENV", "development")
    dev_app = main_module.create_app()
    dev_client = TestClient(dev_app)

    resp = dev_client.get("/docs")

    assert resp.status_code == 200


# --- HSTS in production --------------------------------------------------------


def test_hsts_header_present_in_production(monkeypatch):
    monkeypatch.setenv("ENV", "production")
    prod_app = main_module.create_app()
    prod_app.dependency_overrides[get_supabase] = _fake_supabase
    prod_client = TestClient(prod_app)

    resp = prod_client.get("/health")

    assert resp.headers["strict-transport-security"] == "max-age=31536000; includeSubDomains"


def test_hsts_header_absent_outside_production(monkeypatch):
    monkeypatch.setenv("ENV", "development")
    dev_app = main_module.create_app()
    dev_app.dependency_overrides[get_supabase] = _fake_supabase
    dev_client = TestClient(dev_app)

    resp = dev_client.get("/health")

    assert "strict-transport-security" not in resp.headers


# --- search_documents error path does not leak exception details --------------


def test_search_documents_error_does_not_leak_exception_details(monkeypatch):
    from app import agent as agent_module

    def failing_retrieve(query):
        raise RuntimeError("secret-db-detail")

    monkeypatch.setattr(agent_module, "retrieve", failing_retrieve)

    content, artifact = agent_module.search_documents.func("query")

    assert "secret-db-detail" not in content
    assert all("secret-db-detail" not in str(value) for value in artifact.values())
    assert artifact["error"] == "RuntimeError"


# --- format_chunks_for_model escapes titles ------------------------------------


def test_format_chunks_for_model_escapes_title():
    from app.agent import format_chunks_for_model
    from app.retrieval import RetrievedChunk

    chunk = RetrievedChunk(title='He said "hi" <script>', content="body", similarity=0.9)

    rendered = format_chunks_for_model([chunk])

    assert "&quot;" in rendered
    assert "&lt;" in rendered
    assert 'title="' in rendered
    assert "<script>" not in rendered
