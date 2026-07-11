# Frontend handoff — portfolio chatbot backend

## Base URL

```
https://REPLACE-ME.onrender.com
```

(Set to the actual Render service URL once deployed.)

## CORS

The backend only allows requests from origins listed in its `ALLOWED_ORIGINS`
env var (comma-separated exact origins, set in the Render dashboard). Make
sure the deployed frontend's origin (e.g. `https://your-portfolio.vercel.app`)
is added there, or requests will be blocked by CORS.

## `POST /chat`

Request body:

```json
{
  "messages": [
    { "role": "user", "content": "What projects has the owner worked on?" },
    { "role": "assistant", "content": "..." },
    { "role": "user", "content": "Tell me more about the RAG one." }
  ]
}
```

- `role` is `"user"` or `"assistant"`.
- `content` is the message text (length-limited server-side).
- Send the full running conversation each time (the backend is stateless per request).

Response: `Content-Type: text/event-stream` (SSE), streamed as the agent
generates the reply.

## SSE event shapes

Every event is a single `data: <json>\n\n` line. The JSON payload always has
a `type` and a `text` field:

```
data: {"type": "token", "text": "The"}

data: {"type": "token", "text": " owner"}

data: {"type": "token", "text": " built"}

data: {"type": "token", "text": " a RAG-powered portfolio chatbot."}

data: {"type": "done", "text": ""}

```

- `token` — a chunk of the assistant's reply text. Append each `text` in
  order to build up the visible message as it streams in.
- `done` — marks the end of a successful stream. `text` is always empty.
  Stop listening after this event.
- `error` — something failed server-side. `text` is a generic, user-safe
  error message (never internal details). Example:

```
data: {"type": "error", "text": "Something went wrong on my end. Please try again."}

```

A stream emits either a `done` event or an `error` event as its final event,
never both. Parse each `data:` line with `JSON.parse` — token text may
contain literal newlines, which is exactly why it's JSON-encoded rather than
sent as raw SSE text.

## Cold start / health ping on widget mount

Render's free tier spins the service down after ~15 minutes idle. The first
request after that takes about a minute to respond (official figure).

- On chat-widget mount, fire `GET /health` immediately (don't wait for the
  user to send a message).
- While that ping is in flight (or if it's slow), show a "waking up..." state
  in the widget rather than a bare spinner — a plain spinner reads as broken
  or hung after a few seconds, and a naive request timeout may fire before
  the ~1 minute cold start completes.
- `GET /health` returns `{"status": "ok"}` (200) when Supabase is reachable,
  or `{"status": "error"}` (503) otherwise. Either response means the server
  process itself is now warm.
