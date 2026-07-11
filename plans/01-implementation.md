# Agentic RAG backend for portfolio chatbot — Phased Implementation Plan

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

Layout: `app/{main.py, agent.py, retrieval.py, llm.py, prompts.py, guardrails.py, config.py}`, `ingest.py`, `supabase/schema.sql`, `docs/*.md`, `tests/`, `requirements.txt` (fastapi, uvicorn, langchain, langgraph, langchain-google-genai, langchain-text-splitters, langsmith, supabase, slowapi, pytest, httpx, python-dotenv), `render.yaml`, `.env.example` (`GOOGLE_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `ALLOWED_ORIGINS`, `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`). API surface: `POST /chat` `{messages:[...]}` (client holds history, server stateless), `GET /health`.

> **Required reading:** [02-implementation-notes.md](02-implementation-notes.md) — per-phase SDK details, exact code patterns, and gotchas (LangChain 1.x package scope, embedding normalization, the `create_agent` + middleware setup and its version-specific warts, `astream` SSE filtering + async fallback, LangSmith wiring, slowapi wiring). Consult its matching section before starting each phase below.

---

## Phase 1 — Supabase schema + project setup

**Goal**: A queryable vector store ready for ingestion.

Steps:
- Create Supabase free project (no credit card).
- `supabase/schema.sql`: `documents(id, source, title, content, content_hash, embedding vector(768))`, HNSW index (`m=16, ef_construction=64`), `match_documents(query_embedding, match_count)` SQL function ([pgvector guide](https://supabase.com/docs/guides/database/extensions/pgvector), [semantic search](https://supabase.com/docs/guides/ai/semantic-search)).
- Enable RLS on `documents` with no anon policies (service-role key bypasses RLS server-side).
- `.env.example` with `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`.

Files: `supabase/schema.sql`, `.env.example`.

**Verify**: run `schema.sql` in Supabase SQL editor; confirm table, HNSW index, and `match_documents` function exist; confirm RLS is on with zero policies for `anon`.

---

## Phase 2 — Ingestion pipeline

**Goal**: Turn `docs/*.md` into embedded, idempotent rows in `documents`.

Steps:
- `ingest.py` CLI: heading-aware markdown chunking via `MarkdownHeaderTextSplitter` (headers become metadata) piped into `RecursiveCharacterTextSplitter` (~2000 chars ≈ 500 tokens, ~200 char overlap), both from `langchain-text-splitters`.
- Embed each chunk via `GoogleGenerativeAIEmbeddings` (`gemini-embedding-001`, `output_dimensionality=768`, `task_type="RETRIEVAL_DOCUMENT"`), L2-normalized (see notes).
- Decorate the ingestion pipeline function with LangSmith `@traceable` so ingest runs are traced too.
- Upsert keyed on `content_hash` (hash of chunk content) so re-runs only touch changed chunks — supports growing the corpus over time.

Files: `ingest.py`, `docs/*.md` (seed with real owner content).

**Verify**: `pytest` unit tests on the chunker (boundary cases: no headings, short doc, overlap correctness); run `python ingest.py` against seeded docs and confirm row count matches expected chunk count; re-run and confirm no duplicate rows / only changed chunks re-embedded.

---

## Phase 3 — Retrieval layer

**Goal**: Query → ranked, titled chunks.

Steps:
- `retrieval.py`: embed the incoming query via `GoogleGenerativeAIEmbeddings` (same model/dims/normalization as ingestion, `task_type="RETRIEVAL_QUERY"`), call `match_documents` RPC (parameterized, no string-built SQL), return top-5 chunks with `title`, `content`, similarity score.
- Deliberately do NOT use LangChain's `SupabaseVectorStore` — it lives in the archived, unmaintained `langchain-community` package; direct RPC on our own schema is the maintained path (and a good interview talking point).
- Apply a cosine similarity floor (match threshold) so weak matches don't get passed to the model.

Files: `app/retrieval.py`.

**Verify**: integration test — seed a known doc, query with a paraphrase of its content, assert it's returned as top-1 with score above threshold; assert an unrelated query returns nothing above threshold.

---

## Phase 4 — Agentic pipeline (most thorough phase)

**Goal**: A bounded, model-driven tool-calling loop built on **LangChain 1.x `create_agent` + middleware** — the "agentic" core of the project, not just fixed RAG, and the resume centerpiece (LangChain middleware configured deliberately, LangSmith tracing every turn).

Steps:
- `agent.py`: `create_agent(model, tools=[search_documents], system_prompt=SYSTEM_PROMPT, middleware=[...])` — returns a compiled LangGraph graph. No checkpointer: the server is stateless by design, the client holds history and the full message list is passed per invocation.
- Middleware stack (each choice is deliberate and documented in the notes):
  - `ToolCallLimitMiddleware(run_limit=4, exit_behavior="continue")` — the 5th search attempt is blocked with an error ToolMessage and the model continues, so it must answer from what it already retrieved.
  - `ToolRetryMiddleware(max_retries=1, on_failure="continue")` — one retry on tool failure, then a structured error string back to the model instead of a crash.
  - `ModelRetryMiddleware` — retry with backoff on Gemini 429/5xx before the request fails.
  - One small **custom `AgentMiddleware`** (a `before_model` hook reading the tool-call count from agent state): when the budget is spent, inject a "stop searching — answer now from the search results, or refuse" instruction so the final answer is a real generated response, not a canned limit message. This is the interview artifact: extending the framework where the built-ins stop.
- One tool: `search_documents(query: str)` via the `@tool` decorator (with `response_format="content_and_artifact"` so the model sees formatted text while chunk ids/scores travel as the artifact for logging), wired to `retrieval.py`.
- Bounds: `run_limit=4` in middleware, `recursion_limit` in the invoke config as a hard backstop (catch `GraphRecursionError`), hard timeout per turn.
- Model decides itself when/how often to call the tool — query reformulation, multiple searches for multi-part questions (no orchestration logic forces this; it's a property of tool-binding + system prompt).
- `llm.py`: constructs the `ChatGoogleGenerativeAI` and `GoogleGenerativeAIEmbeddings` instances (the only module importing `langchain_google_genai`).
- LangSmith: set `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` and every agent run is traced automatically — inputs, each node execution, tool calls, token usage.
- `prompts.py` system prompt contract: first-person persona ("answer as the owner"), answer ONLY from retrieved chunks, cite document titles, explicit refusal template for empty/low-similarity retrieval (tied to the Phase 3 similarity floor).
- Failure handling beyond the middleware: any error escaping the graph is caught in the API layer and converted to a graceful "try again later" SSE event; stack traces never surface to the client.
- Structured logging per turn (in addition to LangSmith): tool calls made, queries used, chunk ids returned, similarity scores — for offline retrieval-quality tuning without vendor dependence.

Files: `app/agent.py`, `app/llm.py`, `app/prompts.py`.

**Verify**: `pytest` contract tests with a scripted fake chat model injected via an `build_agent(chat_model)` factory — single tool-call round trip, multi-tool-call round trip (multi-part question), budget-exhaustion path (assert the 5th tool call is blocked AND the final message is generated text, not the canned limit message), empty-retrieval refusal path, simulated tool error + retry (verify against the pinned langchain version that tool exceptions reach the middleware rather than propagating — known issue #33348 in early 1.x), simulated 429 → graceful SSE event. Manually run one real turn and confirm it appears in the LangSmith project with the tool call visible in the trace tree; confirm log lines contain queries/chunk ids/scores.

---

## Phase 5 — Guardrails & abuse protection (thorough)

**Goal**: Cheap, pre-LLM defenses against bad input, injection, and quota drain.

Steps:
- `guardrails.py`, run BEFORE any LLM call:
  - Max message length (~1-2k chars) and max history length (truncate oldest turns).
  - Strip/normalize control characters; reject empty or binary-garbage input.
  - Cheap profanity/vulgarity screen (wordlist, e.g. `better-profanity`) → canned polite response with **no LLM call spent**.
  - Off-topic scoping is handled by the system prompt (Phase 4) plus the retrieval-threshold refusal path, not a separate classifier.
- Prompt-injection defense: user content and retrieved chunks are treated as DATA, clearly delimited in the prompt; system prompt explicitly instructs the model to ignore instructions embedded in user messages or documents; user text is never placed in the system role; the agent has exactly one read-only tool, so injection blast radius is ~zero (no write/exec tools to hijack).
- Output guardrails: response length cap; refusal template preserves persona; explicit instruction + test forbidding the model from echoing/revealing the system prompt.
- Per-IP rate limiting via `slowapi`, keyed on client IP with `X-Forwarded-For` handling (Render sits behind a proxy): e.g. 5 req/min + 30 req/day per IP, plus a global daily cap (default 50 turns — see notes; one turn can spend up to 5 Gemini calls against a ~250 req/day free quota) protecting the shared Gemini free quota (10 RPM upstream). `/health` is exempt from rate limiting (keep-alive cron + widget pre-warm hit it). 429 responses include `Retry-After`. Optional: lightweight per-session token issued to the widget for additional per-session limiting.

Files: `app/guardrails.py`, wiring into `app/main.py`.

**Verify**: `pytest` unit tests — oversize message rejected, oversize history truncated, control-char garbage rejected, profanity wordlist hit returns canned response with zero LLM calls, prompt-injection string in user message doesn't alter persona/behavior (mocked LLM), "reveal your system prompt" test asserts refusal, 429 test asserts rate limiter triggers and `Retry-After` is set.

---

## Phase 6 — API layer, streaming & request/response optimization

**Goal**: Low perceived latency, bounded token/prompt cost.

Steps:
- `main.py`: `POST /chat` streams tokens via SSE using `graph.astream(..., stream_mode="messages")`, filtering to final-answer tokens by `langgraph_node` metadata (known `langchain-google-genai` async caveats — see notes for the fallback); `GET /health` does a trivial DB query (doubles as Supabase keep-alive target).
- Async embed-query and DB calls; construct the compiled graph, chat model, embeddings, and Supabase client once at startup and reuse them.
- Trim conversation history sent upstream to last N turns; cap top-k=5 and chunk size to bound prompt tokens.
- Fast path: static response for greetings ("hi", "hello") with no retrieval/LLM call.
- Response caching noted as optional/later, not built now.

Files: `app/main.py`.

**Verify**: manual curl against `/chat` observing token-by-token SSE stream; `pytest` asserting greeting fast path skips retrieval (mock assertion); confirm client reuse (no per-request client construction) by code inspection/test.

---

## Phase 7 — Security hardening (thorough: "no leaking, no hacking")

**Goal**: Close off secret leakage, corpus exfiltration, and abuse vectors.

Steps:
- Secrets only in env vars: Render dashboard in prod, `.env` gitignored, `.env.example` committed. `SUPABASE_SERVICE_KEY` and `GOOGLE_API_KEY` live only server-side, never shipped to the frontend.
- Supabase: RLS enabled with no anon policies (service-role key bypasses server-side, anon key gets nothing) ([RAG with permissions](https://supabase.com/docs/guides/ai/rag-with-permissions)); retrieval strictly via parameterized `match_documents` RPC, no string-built SQL.
- CORS locked to the exact portfolio origin (no wildcard); only `POST /chat` and `GET /health` exposed; FastAPI docs disabled in prod (`docs_url=None`).
- No leaking: generic error messages to clients, detailed errors logged server-side only; no stack traces surfaced; top-k + similarity threshold + rate limits together bound bulk corpus exfiltration; treat everything under `docs/` as effectively public (don't put anything there you wouldn't say publicly).
- Payload limits: request body size cap, reject non-JSON, pydantic validation on all inputs; security headers middleware; pin dependency versions in `requirements.txt`; `pip audit` in CI noted as optional.
- Threat model summary:

| Threat | Mitigation |
|---|---|
| Prompt injection | Data/instruction delimiting, single read-only tool, system prompt hardening |
| Quota-drain DoS | Per-IP + global rate limits, greeting fast path, bounded tool-call budget |
| Corpus exfiltration | top-k + similarity threshold + rate limits, RLS, no raw SQL |
| Secret leakage | Env-var-only secrets, service key server-side only, CORS lockdown |
| XSS via bot responses | Frontend renders output as text/sanitized markdown, never raw HTML |

Files: `app/main.py` (CORS, docs_url, headers middleware, body size limit), `requirements.txt` (pinned versions).

**Verify**: `pytest` — non-JSON body rejected, oversized payload rejected, CORS preflight from a non-portfolio origin rejected, `/docs` returns 404 in prod config; manual check that `.env` is gitignored and no secret appears in any committed file or client-visible response.

---

## Phase 8 — Deployment, keep-alive & frontend handoff

**Goal**: Live, reachable, self-sustaining free-tier deployment.

Steps:
- `render.yaml` blueprint (free plan); env vars set in Render dashboard — including the `LANGSMITH_*` vars so production turns are traced (5k traces/mo free covers a portfolio bot comfortably; if ever near the cap, set `LANGSMITH_TRACING=false` and everything still works).
- GitHub Actions weekly cron hitting `GET /health` to prevent the Supabase free project from pausing after ~7 days idle.
- Frontend handoff contract: endpoint URL, `{messages:[...]}` request shape, SSE response shape, chat-widget pings `GET /health` on mount to pre-warm Render's cold start (~1 min).

Files: `render.yaml`, `.github/workflows/*.yml` (keep-alive cron), handoff notes for the frontend repo.

**Verify**: curl the deployed Render URL end-to-end (`/health`, then `/chat`); confirm GitHub Actions cron runs on schedule and returns 200; confirm CORS allows requests from the deployed Vite origin.

---

## Phase 9 — End-to-end verification checklist

- `pytest` green across all suites: chunking (Phase 2), retrieval integration (Phase 3), agent-loop contract with mocked LLM (Phase 4), guardrail unit tests incl. profanity/injection/oversize inputs (Phase 5), rate-limit/429 tests (Phase 5), security tests (Phase 7).
- Local e2e: `python ingest.py` → run `uvicorn` → ask a question answerable only from a seeded doc → grounded answer citing the source title; ask an off-topic question → polite refusal; ask a multi-part question → confirm multiple `search_documents` calls both in the Phase 4 turn logs AND in the LangSmith trace tree (open the trace, verify the loop shows model→tools→model→tools→model with distinct queries).
- Deployed e2e: same three checks against the live Render URL, called from the actual Vite portfolio origin (CORS confirmed).
- Prompt-leak attempt: ask the deployed bot to reveal its system prompt or ignore its instructions → confirm refusal, no leakage.
