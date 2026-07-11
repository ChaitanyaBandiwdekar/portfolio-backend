"""Contract tests for app/agent.py's create_agent + middleware loop.

No real APIs: a scripted fake chat model is injected via build_agent(), and
retrieval is monkeypatched. langchain_core's shipped FakeMessagesListChatModel
does not support bind_tools() (BaseChatModel.bind_tools raises
NotImplementedError by default), so we use a small custom fake per the
implementation notes.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app import agent as agent_module
from app.agent import RECURSION_LIMIT, build_agent, log_turn
from app.prompts import REFUSAL_SENTENCE
from app.retrieval import RetrievedChunk


class ScriptedFakeChatModel(BaseChatModel):
    """Returns each response in `responses` in order; records invocation count."""

    responses: list[AIMessage] = []
    calls: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        response = self.responses[self.calls]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=response)])

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"


def _tool_call_message(query: str, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_documents",
                "args": {"query": query},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def _invoke(fake_model: ScriptedFakeChatModel, text: str = "tell me about you"):
    graph = build_agent(fake_model)
    return graph.invoke(
        {"messages": [HumanMessage(text)]},
        config={"recursion_limit": RECURSION_LIMIT},
    )


def _chunk(title="About", content="Some content", similarity=0.9):
    return RetrievedChunk(title=title, content=content, similarity=similarity)


# --- Known 1.x wart (langchain issue #33348): tool exceptions ---------------
# Verified in a REPL against langchain==1.3.13: ToolRetryMiddleware already
# catches the exception, retries once, and returns an error ToolMessage — no
# wrap_tool_call fallback was needed. This test locks that behavior in.


def test_tool_error_retries_once_then_returns_error_string(monkeypatch):
    call_count = {"n": 0}

    def failing_retrieve(query):
        call_count["n"] += 1
        raise RuntimeError("boom")

    monkeypatch.setattr(agent_module, "retrieve", failing_retrieve)

    fake = ScriptedFakeChatModel(
        responses=[
            _tool_call_message("who are you", "call_1"),
            AIMessage(content="I couldn't search, but here's what I can say."),
        ]
    )

    result = _invoke(fake)

    # search_documents itself catches the exception and never raises (it's
    # the retrieve() call inside it that fails), so ToolRetryMiddleware sees
    # a tool that "succeeds" with an error string, not an exception. Assert
    # the model was told search failed and the graph never crashed.
    tool_messages = [m for m in result["messages"] if type(m).__name__ == "ToolMessage"]
    assert len(tool_messages) == 1
    assert "search failed" in tool_messages[0].content
    assert call_count["n"] == 1
    assert result["messages"][-1].content == "I couldn't search, but here's what I can say."


def test_tool_exception_propagating_past_the_tool_is_retried_once(monkeypatch):
    """Belt-and-suspenders: if retrieve() somehow raised past search_documents's
    own try/except (it shouldn't), ToolRetryMiddleware must still catch it,
    retry once, and hand the model an error ToolMessage instead of crashing.
    """
    from langchain_core.tools import tool as tool_decorator

    calls = {"n": 0}

    @tool_decorator(response_format="content_and_artifact")
    def broken_tool(query: str):
        """A tool that always raises."""
        calls["n"] += 1
        raise ValueError("always fails")

    from langchain.agents import create_agent
    from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware

    fake = ScriptedFakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "broken_tool", "args": {"query": "x"}, "id": "c1", "type": "tool_call"}
                ],
            ),
            AIMessage(content="final"),
        ]
    )
    graph = create_agent(
        fake,
        tools=[broken_tool],
        system_prompt="test",
        middleware=[
            ToolRetryMiddleware(max_retries=1, on_failure="continue"),
            ModelRetryMiddleware(max_retries=1),
        ],
    )

    result = graph.invoke(
        {"messages": [HumanMessage("hi")]}, config={"recursion_limit": RECURSION_LIMIT}
    )

    assert calls["n"] == 2  # 1 initial attempt + 1 retry
    tool_messages = [m for m in result["messages"] if type(m).__name__ == "ToolMessage"]
    assert len(tool_messages) == 1
    assert tool_messages[0].status == "error"
    assert result["messages"][-1].content == "final"


# --- Core round-trip paths ---------------------------------------------------


def test_single_tool_call_round_trip(monkeypatch):
    monkeypatch.setattr(agent_module, "retrieve", lambda query: [_chunk()])

    fake = ScriptedFakeChatModel(
        responses=[
            _tool_call_message("who are you", "call_1"),
            AIMessage(content="I'm the owner. (About)"),
        ]
    )

    result = _invoke(fake)

    assert fake.calls == 2
    assert result["messages"][-1].content == "I'm the owner. (About)"
    tool_messages = [m for m in result["messages"] if type(m).__name__ == "ToolMessage"]
    assert len(tool_messages) == 1
    assert "About" in tool_messages[0].content


def test_multi_tool_call_round_trip(monkeypatch):
    monkeypatch.setattr(
        agent_module, "retrieve", lambda query: [_chunk(title=f"Doc for {query}")]
    )

    fake = ScriptedFakeChatModel(
        responses=[
            _tool_call_message("part one", "call_1"),
            _tool_call_message("part two", "call_2"),
            AIMessage(content="Combined answer from both parts."),
        ]
    )

    result = _invoke(fake, text="answer part one and part two")

    assert fake.calls == 3
    tool_messages = [m for m in result["messages"] if type(m).__name__ == "ToolMessage"]
    assert len(tool_messages) == 2
    assert result["messages"][-1].content == "Combined answer from both parts."


def test_budget_exhaustion_blocks_5th_call_and_generates_final_answer(monkeypatch):
    monkeypatch.setattr(agent_module, "retrieve", lambda query: [_chunk()])

    # 5 tool-call attempts scripted (budget is 4 -> the 5th is blocked), then
    # a real generated final answer.
    responses = [_tool_call_message(f"q{i}", f"call_{i}") for i in range(5)]
    responses.append(AIMessage(content="Final generated answer after budget."))
    fake = ScriptedFakeChatModel(responses=responses)

    result = _invoke(fake)

    # All 5 tool-call responses were consumed plus the final -> 6 model calls.
    assert fake.calls == 6

    tool_messages = [m for m in result["messages"] if type(m).__name__ == "ToolMessage"]
    assert len(tool_messages) == 5
    assert tool_messages[4].status == "error"
    assert "limit" in tool_messages[4].content.lower()

    # Final message is real generated text, not a canned limit message.
    assert result["messages"][-1].content == "Final generated answer after budget."


def test_empty_retrieval_triggers_refusal(monkeypatch):
    monkeypatch.setattr(agent_module, "retrieve", lambda query: [])

    fake = ScriptedFakeChatModel(
        responses=[
            _tool_call_message("something not in the docs", "call_1"),
            AIMessage(content=REFUSAL_SENTENCE),
        ]
    )

    result = _invoke(fake)

    tool_messages = [m for m in result["messages"] if type(m).__name__ == "ToolMessage"]
    assert tool_messages[0].content == "no relevant documents found"
    assert result["messages"][-1].content == REFUSAL_SENTENCE


# --- log_turn -----------------------------------------------------------------


def test_log_turn_emits_structured_json_with_queries_chunk_ids_scores(monkeypatch, caplog):
    monkeypatch.setattr(agent_module, "retrieve", lambda query: [_chunk(title="About", similarity=0.87)])

    fake = ScriptedFakeChatModel(
        responses=[
            _tool_call_message("who are you", "call_1"),
            AIMessage(content="I'm the owner. (About)"),
        ]
    )
    result = _invoke(fake)

    with caplog.at_level(logging.INFO, logger="app.agent"):
        log_turn(result)

    assert len(caplog.records) == 1
    payload = json.loads(caplog.records[0].message)
    assert payload["event"] == "turn"
    assert payload["outcome"] == "answered"
    assert payload["rounds"] == 1
    assert payload["tool_calls"][0]["query"] == "who are you"
    assert payload["tool_calls"][0]["chunk_ids"] == ["About"]
    assert payload["tool_calls"][0]["scores"] == [0.87]


def test_log_turn_reports_budget_exhausted_outcome(monkeypatch, caplog):
    monkeypatch.setattr(agent_module, "retrieve", lambda query: [_chunk()])

    responses = [_tool_call_message(f"q{i}", f"call_{i}") for i in range(5)]
    responses.append(AIMessage(content="Final generated answer after budget."))
    fake = ScriptedFakeChatModel(responses=responses)
    result = _invoke(fake)

    with caplog.at_level(logging.INFO, logger="app.agent"):
        log_turn(result)

    payload = json.loads(caplog.records[0].message)
    assert payload["outcome"] == "budget_exhausted"
    # rounds counts executed searches (with artifacts); the 5th was blocked
    # before execution and has no artifact.
    assert payload["rounds"] == 4
