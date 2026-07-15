"""FastAPI app: POST /chat (SSE token streaming) and GET /health.

Client/agent construction happens once, in the lifespan, and is exposed to
request handlers as overridable dependencies (get_agent / get_supabase) —
the seam tests use to inject a scripted fake agent without ever building the
real Gemini model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware

from app.agent import RECURSION_LIMIT, build_agent, log_turn
from app.guardrails import (
    check_daily_cap,
    check_input,
    check_rate_limit,
    client_ip,
    consume_rate_limit,
    is_greeting,
)
from app.llm import get_chat_model
from app.supabase_client import get_supabase_client

load_dotenv()

logger = logging.getLogger("app.main")

GREETING_RESPONSE = "Hi! Ask me anything about the owner's background, projects, or experience."
DAILY_CAP_RESPONSE = "This assistant has hit its usage limit for today. Please try again tomorrow."
GENERIC_ERROR_MESSAGE = "Something went wrong on my end. Please try again."

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

MAX_BODY_BYTES = 16 * 1024


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Rejects requests whose declared Content-Length exceeds the cap.
    Header check only — no chunked-body accounting."""

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                parsed_length = int(content_length)
            except ValueError:
                return JSONResponse({"detail": "Invalid Content-Length header"}, status_code=400)
            if parsed_length > MAX_BODY_BYTES:
                return JSONResponse({"detail": "Payload too large"}, status_code=413)
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        if os.environ.get("ENV") == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


def _parse_allowed_origins() -> list[str]:
    raw = os.environ.get("ALLOWED_ORIGINS", "")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Built once here and reused across every request — never construct
    # these inside a request handler.
    chat_model = get_chat_model()
    app.state.agent = build_agent(chat_model)
    app.state.supabase_client = get_supabase_client()
    yield


router = APIRouter()


def create_app() -> FastAPI:
    """App factory: docs exposure and CORS origins are read from env at
    creation time (ENV, ALLOWED_ORIGINS), so tests can build a fresh app
    under monkeypatched env without disturbing the module-level `app`."""
    is_production = os.environ.get("ENV") == "production"
    new_app = FastAPI(
        lifespan=lifespan,
        docs_url=None if is_production else "/docs",
        redoc_url=None if is_production else "/redoc",
        openapi_url=None if is_production else "/openapi.json",
    )
    new_app.add_middleware(BodySizeLimitMiddleware)
    new_app.add_middleware(SecurityHeadersMiddleware)
    new_app.add_middleware(
        CORSMiddleware,
        allow_origins=_parse_allowed_origins(),
        allow_methods=["POST", "GET", "OPTIONS"],
    )
    new_app.include_router(router)
    return new_app


app = create_app()


def get_agent(request: Request) -> Any:
    return request.app.state.agent


def get_supabase(request: Request) -> Any:
    return request.app.state.supabase_client


def _sse_event(event_type: str, text: str) -> str:
    # JSON-encode every event: raw token text can contain newlines, which
    # would otherwise break the "data: ...\n\n" SSE framing.
    return f"data: {json.dumps({'type': event_type, 'text': text})}\n\n"


async def _canned_stream(text: str):
    """A single-message SSE response for paths that never touch the LLM
    (guardrail rejection, greeting fast path, daily cap) — keeps /chat's
    response shape uniform for the client regardless of path taken."""
    yield _sse_event("token", text)
    yield _sse_event("done", "")


def _to_lc_messages(messages: list[dict[str, str]]) -> list[Any]:
    return [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in messages
    ]


def _content_text(content: Any) -> str:
    """Newer Gemini models stream content as a list of blocks (with
    thought signatures), not a plain str — extract just the text."""
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _accumulate_final_state(chunks: list[tuple[Any, dict]]) -> dict[str, Any]:
    """Best-effort reconstruction of an agent.invoke()-shaped result dict from
    streamed message chunks, so log_turn can be reused unchanged. Token
    deltas on the "model" node are concatenated into one final AIMessage;
    ToolMessage chunks (not token-streamed) pass through as-is. This is
    logging-only — it never affects what's sent to the client."""
    messages: list[Any] = []
    buffer = ""
    tool_calls: list[dict] = []
    for chunk, metadata in chunks:
        if isinstance(chunk, ToolMessage):
            if buffer or tool_calls:
                messages.append(AIMessage(content=buffer, tool_calls=tool_calls))
                buffer, tool_calls = "", []
            messages.append(chunk)
        elif metadata.get("langgraph_node") == "model":
            if chunk.content:
                buffer += _content_text(chunk.content)
            if chunk.tool_calls:
                tool_calls.extend(chunk.tool_calls)
    if buffer or tool_calls:
        messages.append(AIMessage(content=buffer, tool_calls=tool_calls))
    return {"messages": messages}


async def _agent_event_stream(agent: Any, lc_messages: list[Any]):
    raw_chunks: list[tuple[Any, dict]] = []
    try:
        # Canonical LangGraph pattern (stream_mode="messages"): known risk —
        # langchain-google-genai has had astream/ainvoke issues in some
        # event-loop setups while sync works. Untested against real Gemini
        # here (no API key). If this breaks in production, the sanctioned
        # fallback is agent.stream(...) (sync) driven through
        # starlette.concurrency.iterate_in_threadpool, with the same
        # event-filtering logic below.
        async with asyncio.timeout(60):
            async for chunk, metadata in agent.astream(
                {"messages": lc_messages},
                config={"recursion_limit": RECURSION_LIMIT},
                stream_mode="messages",
            ):
                raw_chunks.append((chunk, metadata))
                text = _content_text(chunk.content)
                if (
                    metadata.get("langgraph_node") == "model"
                    and text
                    and not chunk.tool_calls
                ):
                    yield _sse_event("token", text)
        yield _sse_event("done", "")
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        yield _sse_event("error", GENERIC_ERROR_MESSAGE)
    except Exception:
        logger.exception("agent stream failed")
        yield _sse_event("error", GENERIC_ERROR_MESSAGE)
    finally:
        try:
            log_turn(_accumulate_final_state(raw_chunks))
        except Exception:
            logger.exception("log_turn failed")


@router.post("/chat")
async def chat(request: Request, body: ChatRequest, agent: Any = Depends(get_agent)):
    raw_messages = [m.model_dump() for m in body.messages]

    result = check_input(raw_messages)
    if not result.ok:
        return StreamingResponse(
            _canned_stream(result.canned_response), media_type="text/event-stream", headers=SSE_HEADERS
        )

    last_user_text = next(
        (m["content"] for m in reversed(result.messages) if m["role"] == "user"), ""
    )
    if is_greeting(last_user_text):
        return StreamingResponse(
            _canned_stream(GREETING_RESPONSE), media_type="text/event-stream", headers=SSE_HEADERS
        )

    rate_status = check_rate_limit(client_ip(request))
    if not rate_status.allowed:
        return JSONResponse(
            {"error": "Rate limit exceeded"},
            status_code=429,
            headers={"Retry-After": str(rate_status.retry_after)},
        )

    if not check_daily_cap():
        return StreamingResponse(
            _canned_stream(DAILY_CAP_RESPONSE), media_type="text/event-stream", headers=SSE_HEADERS
        )

    consume_rate_limit(client_ip(request))

    lc_messages = _to_lc_messages(result.messages)
    return StreamingResponse(
        _agent_event_stream(agent, lc_messages), media_type="text/event-stream", headers=SSE_HEADERS
    )


@router.get("/health")
async def health(supabase_client: Any = Depends(get_supabase)):
    try:
        await run_in_threadpool(
            lambda: supabase_client.table("documents").select("id").limit(1).execute()
        )
        return {"status": "ok"}
    except Exception:
        logger.exception("health check failed")
        return JSONResponse({"status": "error"}, status_code=503)
