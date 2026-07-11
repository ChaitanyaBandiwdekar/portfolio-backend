# Agentic RAG backend for portfolio chatbot — Overview

## Context

The portfolio site (React/Vite static frontend) needs a backend chatbot that answers visitor questions about its owner, grounded in the owner's own documents. This is deliberately a learning/showcase project for recruiters: a real RAG pipeline (chunking, embeddings, pgvector retrieval) plus a real agentic loop built on **LangChain 1.x `create_agent`** (which compiles to a LangGraph graph) with the plumbing configured through its **middleware system** — built-in tool-call-limit and retry middleware plus one small custom `AgentMiddleware` for graceful budget exhaustion — so the loop is resume-relevant (LangChain, LangGraph, LangSmith), uses the right level of abstraction, and every control point (budget, retries, forced final answer) is still an explicit, explainable decision rather than a single fixed embed→retrieve→answer call. Budget is $0/month. Python backend, existing `.venv` in `C:\Users\chait\Projects\portfolio-backend`. The document corpus is expected to grow over time, so ingestion must be idempotent and re-runnable.

## Stack decisions (final)

| Layer | Choice |
|---|---|
| API | FastAPI + uvicorn, SSE streaming ([Neon guide](https://neon.com/guides/react-fastapi-rag-portfolio)) |
| Vector store | Supabase free-tier Postgres + pgvector, HNSW (`m=16, ef_construction=64`), `match_documents(query_embedding, match_count)` RPC ([pgvector guide](https://supabase.com/docs/guides/database/extensions/pgvector), [semantic search](https://supabase.com/docs/guides/ai/semantic-search), [hybrid search upgrade path](https://supabase.com/docs/guides/ai/hybrid-search), [pgvector vs alternatives](https://www.kalviumlabs.ai/blog/vector-databases-compared-pgvector-pinecone-qdrant-weaviate/)) |
| Embeddings | Gemini `gemini-embedding-001`, 768 dims, via `langchain-google-genai` `GoogleGenerativeAIEmbeddings` ([GA post](https://developers.googleblog.com/gemini-embedding-available-gemini-api/), [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)) |
| Generation | Gemini 2.5 Flash via `langchain-google-genai` `ChatGoogleGenerativeAI` + `bind_tools`; free tier ~10 RPM — check real daily quota in AI Studio dashboard ([rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)) |
| Agent orchestration | LangChain 1.x **`create_agent`** (compiles to a LangGraph graph) + middleware: `ToolCallLimitMiddleware(run_limit=4, exit_behavior="continue")`, `ToolRetryMiddleware`, `ModelRetryMiddleware`, one small custom `AgentMiddleware` (forced-final-answer nudge); `recursion_limit` backstop ([built-in middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in), [custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom)) |
| Chunking | `langchain-text-splitters`: `MarkdownHeaderTextSplitter` → `RecursiveCharacterTextSplitter` ([splitters](https://docs.langchain.com/oss/python/integrations/splitters)) |
| Observability | LangSmith free tier (5k traces/mo): zero-code tracing of every agent turn, `@traceable` on ingestion ([pricing](https://www.langchain.com/pricing)) |
| Hosting | Render free tier, no card, idle spin-down ([Render free tiers](https://render.com/articles/platforms-with-a-real-free-tier-for-developers-in-2026), [PaaS comparison](https://www.birjob.com/blog/paas-comparison-railway-render-fly-vercel-2026)) |
| Provider isolation | Model/embeddings construction only in `llm.py` — swapping providers (e.g. Claude Haiku via `langchain-anthropic` + [Citations API](https://platform.claude.com/docs/en/build-with-claude/citations)) means changing one chat-model class |

## Project layout

Layout: `app/{main.py, agent.py, retrieval.py, llm.py, prompts.py, guardrails.py, config.py}`, `ingest.py`, `supabase/schema.sql`, `docs/*.md`, `tests/`, `requirements.txt` (fastapi, uvicorn, langchain, langgraph, langchain-google-genai, langchain-text-splitters, langsmith, supabase, slowapi, pytest, httpx, python-dotenv), `render.yaml`, `.env.example` (`GOOGLE_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `ALLOWED_ORIGINS`, `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`). API surface: `POST /chat` `{messages:[...]}` (client holds history, server stateless), `GET /health`.

## Phase index

All nine phases live in [01-implementation.md](01-implementation.md). **Before implementing any phase, read [02-implementation-notes.md](02-implementation-notes.md)** — it contains mandatory API-level details and gotchas the phase descriptions assume.

1. Phase 1 — Supabase schema + project setup
2. Phase 2 — Ingestion pipeline
3. Phase 3 — Retrieval layer
4. Phase 4 — Agentic pipeline
5. Phase 5 — Guardrails & abuse protection
6. Phase 6 — API layer, streaming & request/response optimization
7. Phase 7 — Security hardening
8. Phase 8 — Deployment, keep-alive & frontend handoff
9. Phase 9 — End-to-end verification checklist
