"""Tests for app/main.py — the FastAPI API layer, SSE streaming, rate limits.

Real Gemini/Supabase are never constructed: the app's agent/supabase seams
are FastAPI dependencies (get_agent, get_supabase) overridden per test via
app.dependency_overrides. The one-time-construction test is the only one
that lets the real lifespan run, with get_chat_model/build_agent/
get_supabase_client monkeypatched.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import ToolMessage

from app import guardrails as guardrails_module
from app import main as main_module
from app.main import app, get_agent, get_supabase


class FakeAgent:
    """Scripted async agent double: astream yields (chunk, metadata) tuples
    exactly like LangGraph's stream_mode="messages"."""

    def __init__(self, events, raise_exc: Exception | None = None):
        self._events = events
        self._raise_exc = raise_exc
        self.calls = 0

    async def astream(self, *args, **kwargs):
        self.calls += 1
        for chunk, metadata in self._events:
            yield chunk, metadata
        if self._raise_exc:
            raise self._raise_exc


class _Chunk:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


def _model_event(text):
    return (_Chunk(content=text), {"langgraph_node": "model"})


def _tool_call_event():
    return (_Chunk(content="", tool_calls=[{"name": "search_documents"}]), {"langgraph_node": "model"})


def _tool_message_event():
    msg = ToolMessage(content="results", artifact={"query": "q", "chunk_ids": ["A"], "scores": [0.9]}, tool_call_id="c1")
    return (msg, {"langgraph_node": "tools"})


@pytest.fixture(autouse=True)
def _reset_state():
    guardrails_module.limiter.reset()
    guardrails_module._daily_count = 0
    yield
    app.dependency_overrides.clear()
    guardrails_module.limiter.reset()


@pytest.fixture
def client():
    return TestClient(app)


def _override_agent(fake_agent):
    app.dependency_overrides[get_agent] = lambda: fake_agent


def _override_supabase(fake_client):
    app.dependency_overrides[get_supabase] = lambda: fake_client


def _parse_sse(body: str):
    events = []
    for block in body.strip().split("\n\n"):
        if not block.strip():
            continue
        line = block[len("data: ") :] if block.startswith("data: ") else block
        events.append(json.loads(line))
    return events


def _chat_body(text="tell me about the projects"):
    return {"messages": [{"role": "user", "content": text}]}


# --- greeting fast path ------------------------------------------------------


def test_greeting_fast_path_skips_agent(client):
    fake = FakeAgent(events=[])
    _override_agent(fake)

    resp = client.post("/chat", json=_chat_body("hi"))

    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert [e["type"] for e in events] == ["token", "done"]
    assert fake.calls == 0


# --- guardrail rejection ------------------------------------------------------


def test_guardrail_rejection_zero_llm_calls(client):
    fake = FakeAgent(events=[])
    _override_agent(fake)

    resp = client.post("/chat", json=_chat_body("you are a fucking idiot"))

    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert [e["type"] for e in events] == ["token", "done"]
    assert fake.calls == 0


def test_empty_messages_rejected(client):
    fake = FakeAgent(events=[])
    _override_agent(fake)

    resp = client.post("/chat", json={"messages": []})

    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert events[0]["type"] == "token"
    assert fake.calls == 0


# --- SSE stream shape ---------------------------------------------------------


def test_stream_emits_token_and_done_events(client):
    fake = FakeAgent(
        events=[
            _model_event("Hello"),
            _model_event(" world"),
        ]
    )
    _override_agent(fake)

    resp = client.post("/chat", json=_chat_body())

    events = _parse_sse(resp.text)
    assert [e["type"] for e in events] == ["token", "token", "done"]
    assert events[0]["text"] == "Hello"
    assert events[1]["text"] == " world"
    assert fake.calls == 1


def test_stream_filters_tool_call_chunks_and_tool_node(client):
    fake = FakeAgent(
        events=[
            _tool_call_event(),
            _tool_message_event(),
            _model_event("final answer"),
        ]
    )
    _override_agent(fake)

    resp = client.post("/chat", json=_chat_body())

    events = _parse_sse(resp.text)
    assert [e["type"] for e in events] == ["token", "done"]
    assert events[0]["text"] == "final answer"


def test_stream_newline_in_token_is_safe(client):
    fake = FakeAgent(events=[_model_event("line one\nline two")])
    _override_agent(fake)

    resp = client.post("/chat", json=_chat_body())

    events = _parse_sse(resp.text)
    assert events[0]["type"] == "token"
    assert events[0]["text"] == "line one\nline two"


def test_stream_error_emits_error_event(client):
    fake = FakeAgent(events=[_model_event("partial")], raise_exc=RuntimeError("boom"))
    _override_agent(fake)

    resp = client.post("/chat", json=_chat_body())

    events = _parse_sse(resp.text)
    assert events[-1]["type"] == "error"
    # a token emitted before the failure must not be dropped
    assert any(e["type"] == "token" for e in events)


def test_log_turn_called_after_stream(client, monkeypatch):
    calls = []
    monkeypatch.setattr(main_module, "log_turn", lambda result: calls.append(result))

    fake = FakeAgent(events=[_tool_message_event(), _model_event("final")])
    _override_agent(fake)

    client.post("/chat", json=_chat_body())

    assert len(calls) == 1
    assert "messages" in calls[0]


# --- daily cap -----------------------------------------------------------------


def test_daily_cap_exceeded_returns_canned_response_zero_llm(client, monkeypatch):
    monkeypatch.setattr(main_module, "check_daily_cap", lambda: False)
    fake = FakeAgent(events=[])
    _override_agent(fake)

    resp = client.post("/chat", json=_chat_body())

    events = _parse_sse(resp.text)
    assert [e["type"] for e in events] == ["token", "done"]
    assert fake.calls == 0


# --- rate limiting ---------------------------------------------------------


def test_rate_limit_429_after_five_per_minute(client):
    fake = FakeAgent(events=[])
    _override_agent(fake)

    for _ in range(5):
        resp = client.post("/chat", json=_chat_body("hi"))
        assert resp.status_code == 200

    resp = client.post("/chat", json=_chat_body("hi"))
    assert resp.status_code == 429
    assert "Retry-After" in resp.headers


# --- /health --------------------------------------------------------------


def test_health_returns_ok_with_mocked_db(client):
    fake_client = MagicMock()
    fake_client.table.return_value.select.return_value.limit.return_value.execute.return_value = None
    _override_supabase(fake_client)

    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_not_rate_limited(client):
    fake_client = MagicMock()
    fake_client.table.return_value.select.return_value.limit.return_value.execute.return_value = None
    _override_supabase(fake_client)

    for _ in range(10):
        resp = client.get("/health")
        assert resp.status_code == 200


def test_health_hides_exception_detail_on_failure(client):
    fake_client = MagicMock()
    fake_client.table.side_effect = RuntimeError("db exploded")
    _override_supabase(fake_client)

    resp = client.get("/health")

    assert resp.status_code == 503
    assert "db exploded" not in resp.text


# --- one-time construction --------------------------------------------------


def test_clients_constructed_once_across_two_requests(monkeypatch):
    chat_model_calls = []
    build_agent_calls = []
    supabase_calls = []

    monkeypatch.setattr(main_module, "get_chat_model", lambda: chat_model_calls.append(1) or "chat-model")
    monkeypatch.setattr(
        main_module, "build_agent", lambda cm: build_agent_calls.append(cm) or FakeAgent(events=[])
    )
    fake_supabase = MagicMock()
    fake_supabase.table.return_value.select.return_value.limit.return_value.execute.return_value = None
    monkeypatch.setattr(main_module, "get_supabase_client", lambda: supabase_calls.append(1) or fake_supabase)

    with TestClient(app) as test_client:
        r1 = test_client.get("/health")
        r2 = test_client.get("/health")

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert len(chat_model_calls) == 1
    assert len(build_agent_calls) == 1
    assert len(supabase_calls) == 1
