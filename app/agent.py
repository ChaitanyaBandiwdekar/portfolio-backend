"""The agentic RAG loop: `create_agent` + middleware, no hand-built StateGraph.

`build_agent(chat_model)` is a factory so production passes the real Gemini
chat model (app/llm.py) and tests inject a scripted fake — the seam the
implementation notes ask for.
"""

from __future__ import annotations

import html
import json
import logging
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from app.prompts import REFUSAL_SENTENCE
from app.retrieval import RetrievedChunk, retrieve

logger = logging.getLogger("app.agent")

TOOL_CALL_BUDGET = 4  # matches ToolCallLimitMiddleware's run_limit below

# Empirically measured (not the notes' assumed 16): with this middleware stack,
# each tool-call round is 4 graph steps (before_model / model / after_model /
# tools), not 2. A full budget-exhaustion round trip (4 executed calls + 1
# blocked call + the final answer) takes 22 steps. 16 raises GraphRecursionError
# before the model ever gets to answer. 30 leaves headroom while still bounding
# a runaway loop. Verified in a REPL against langchain==1.3.13 / langgraph==1.2.9.
RECURSION_LIMIT = 30

NO_RESULTS_MESSAGE = "no relevant documents found"


def format_chunks_for_model(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as delimited, titled text for the model."""
    if not chunks:
        return NO_RESULTS_MESSAGE
    parts = ["<search_results>"]
    for chunk in chunks:
        parts.append(
            f'<result title="{html.escape(chunk.title, quote=True)}">\n{chunk.content}\n</result>'
        )
    parts.append("</search_results>")
    return "\n".join(parts)


@tool(response_format="content_and_artifact")
def search_documents(query: str) -> tuple[str, dict[str, Any]]:
    """Search the owner's documents (resume, projects, about page).

    Use for any question about the owner. Call multiple times with different
    queries for multi-part questions.
    """
    try:
        chunks = retrieve(query)
    except Exception as exc:  # never raise through the graph
        logger.exception("search_documents failed for query=%r", query)
        return "search failed, you may retry once", {
            "query": query,
            "chunk_ids": [],
            "scores": [],
            "error": type(exc).__name__,
        }

    content = format_chunks_for_model(chunks)
    return content, {
        "query": query,
        "chunk_ids": [chunk.title for chunk in chunks],
        "scores": [chunk.similarity for chunk in chunks],
    }


class ForceAnswerMiddleware(AgentMiddleware):
    """Once the tool-call budget is spent, nudge the model to answer now.

    `ToolCallLimitMiddleware(exit_behavior="continue")` blocks further tool
    calls with an error ToolMessage, but on its own the model may keep trying
    to search (burning rounds until recursion_limit) instead of producing a
    real answer. This before_model hook injects an explicit instruction once
    the budget is spent so the final message is a generated answer (grounded
    in whatever was retrieved, or a refusal), not a stalled loop.
    """

    def before_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        run_count = state.get("run_tool_call_count", {}).get("__all__", 0)
        if run_count < TOOL_CALL_BUDGET:
            return None
        nudge = SystemMessage(
            content=(
                "You cannot search again. Answer now from the search results "
                "above, or refuse per your instructions."
            )
        )
        return {"messages": [nudge]}


def build_agent(chat_model: Any):
    """Compile the agent graph. No checkpointer: the API is stateless, the
    client holds history and passes the full message list per invocation."""
    from app.prompts import SYSTEM_PROMPT

    return create_agent(
        chat_model,
        tools=[search_documents],
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            ToolCallLimitMiddleware(run_limit=TOOL_CALL_BUDGET, exit_behavior="continue"),
            ToolRetryMiddleware(max_retries=1, on_failure="continue"),
            ModelRetryMiddleware(max_retries=1),
            ForceAnswerMiddleware(),
        ],
    )


def log_turn(result: dict[str, Any]) -> None:
    """Emit one structured JSON log line summarizing a completed turn.

    `result` is the state dict returned by `agent.invoke(...)`. Chunk ids,
    scores, and queries come from each ToolMessage's artifact (never from the
    text the model saw), so this works regardless of what the model quoted.
    """
    messages = result.get("messages", [])
    tool_calls = [
        {
            "query": msg.artifact.get("query"),
            "chunk_ids": msg.artifact.get("chunk_ids", []),
            "scores": msg.artifact.get("scores", []),
        }
        for msg in messages
        if isinstance(msg, ToolMessage) and isinstance(msg.artifact, dict)
    ]

    # ToolCallLimitMiddleware's run_tool_call_count is a private/untracked
    # state field and isn't present on the dict invoke() returns, so budget
    # exhaustion is inferred from how many search_documents calls the model
    # actually requested (present in the AIMessages regardless of whether
    # they were blocked).
    requested_calls = sum(
        1
        for msg in messages
        if isinstance(msg, AIMessage)
        for call in (msg.tool_calls or [])
        if call.get("name") == "search_documents"
    )
    final_ai = next(
        (msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None
    )
    final_text = final_ai.content if final_ai else ""

    if requested_calls > TOOL_CALL_BUDGET:
        outcome = "budget_exhausted"
    elif any(
        isinstance(msg, ToolMessage) and msg.status == "error" for msg in messages
    ):
        outcome = "error"
    elif isinstance(final_text, str) and REFUSAL_SENTENCE in final_text:
        outcome = "refused"
    else:
        outcome = "answered"

    logger.info(
        json.dumps(
            {
                "event": "turn",
                "tool_calls": tool_calls,
                "rounds": len(tool_calls),
                "outcome": outcome,
            }
        )
    )
