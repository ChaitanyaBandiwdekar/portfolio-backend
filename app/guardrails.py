"""Cheap, pre-LLM defenses: bad input, quota drain. Run BEFORE any LLM call.

`check_input` is the one entry point. It must never import app.llm — that's
what makes "profanity rejections spend zero Gemini calls" true by
construction, not by mocking. Also home to pieces wired up in Phase 6: the
greeting fast-path check, the global daily cap, and the slowapi Limiter +
client IP key_func.
"""

from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass
from datetime import date

from better_profanity import profanity
from slowapi import Limiter

profanity.load_censor_words()  # once at import, not per request

MAX_MESSAGE_LENGTH = 2000
MAX_HISTORY = 10
MAX_RESPONSE_LENGTH = 2000  # output cap, applied to generated responses in Phase 6

CANNED_INVALID_RESPONSE = "Sorry, I couldn't understand that message. Could you rephrase?"
CANNED_PROFANITY_RESPONSE = (
    "I'd rather keep this conversation respectful — could you rephrase that?"
)

GREETINGS = {"hi", "hello", "hey", "yo", "sup", "hiya", "howdy", "hi!", "hello!", "hey!", "yo!"}


@dataclass
class GuardrailResult:
    ok: bool
    messages: list[dict[str, str]] | None = None
    canned_response: str | None = None
    reason: str | None = None


def _strip_control_chars(text: str) -> str:
    return "".join(c for c in text if c in "\n\t" or unicodedata.category(c) != "Cc")


def _reject(reason: str, response: str = CANNED_INVALID_RESPONSE) -> GuardrailResult:
    return GuardrailResult(ok=False, canned_response=response, reason=reason)


def check_input(messages: list[dict[str, str]]) -> GuardrailResult:
    """Validate and clean incoming chat history. Check order matters:
    shape -> length caps -> history truncation -> control-char strip ->
    empty/blank -> profanity. Blank and profanity rejection only look at the
    newest user message — stale profane or blank turns already sitting in
    client-held history must not reject an otherwise-clean new message."""
    if not isinstance(messages, list) or not messages:
        return _reject("empty")

    for msg in messages:
        if not isinstance(msg, dict) or not isinstance(msg.get("content"), str):
            return _reject("empty")

    for msg in messages:
        if len(msg["content"]) > MAX_MESSAGE_LENGTH:
            return _reject("oversize_message")

    messages = messages[-MAX_HISTORY:]

    messages = [{**msg, "content": _strip_control_chars(msg["content"])} for msg in messages]

    user_messages = [msg for msg in messages if msg["role"] == "user"]
    newest = user_messages[-1] if user_messages else messages[-1]

    if not newest["content"].strip():
        return _reject("empty")

    if profanity.contains_profanity(newest["content"]):
        return _reject("profanity", CANNED_PROFANITY_RESPONSE)

    return GuardrailResult(ok=True, messages=messages)


def is_greeting(text: str) -> bool:
    """Exact match only (not substring) — "hi, what did he work on?" must not
    match. Wired into the Phase 6 fast path."""
    return text.strip().lower() in GREETINGS


# --- Global daily cap -------------------------------------------------------
# In-process counter + date, reset when the date changes. No persistence:
# Render free tier spins the instance down on idle anyway, which resets it
# too, and this is a coarse abuse backstop (not billing), so that's fine.

GLOBAL_DAILY_CAP = int(os.environ.get("GLOBAL_DAILY_CAP", "50"))

_daily_count = 0
_daily_date = date.today()


def check_daily_cap() -> bool:
    """Return True (and increment) if under the cap; False if exhausted."""
    global _daily_count, _daily_date
    today = date.today()
    if today != _daily_date:
        _daily_date = today
        _daily_count = 0
    if _daily_count >= GLOBAL_DAILY_CAP:
        return False
    _daily_count += 1
    return True


# --- slowapi rate limiting ---------------------------------------------------
# Render sits behind one proxy; slowapi's default key_func does not read
# X-Forwarded-For, which would key every request to the proxy's IP. Take the
# LAST entry (appended by Render's own proxy) rather than the first (client-
# supplied and trivially spoofable). Wiring the Limiter into the app
# (app.state.limiter, exception handler, per-route decorators) is Phase 6's
# job; the Limiter and key_func just need to exist here, ready to import.


def client_ip(request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[-1].strip() if fwd else (request.client.host if request.client else "unknown")


limiter = Limiter(key_func=client_ip, headers_enabled=True)
