"""Cheap, pre-LLM defenses: bad input, quota drain. Run BEFORE any LLM call.

`check_input` is the one entry point. It must never import app.llm — that's
what makes "profanity rejections spend zero Gemini calls" true by
construction, not by mocking. Also home to pieces wired up in Phase 6: the
greeting fast-path check, the global daily cap, and the rate limiter +
client IP key_func.
"""

from __future__ import annotations

import os
import random
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import date

from better_profanity import profanity
from limits import parse
from limits.storage import MemoryStorage
from limits.strategies import MovingWindowRateLimiter
from pygarble import EnsembleDetector

profanity.load_censor_words()  # once at import, not per request

MAX_MESSAGE_LENGTH = 2000
MAX_HISTORY = 10
MAX_RESPONSE_LENGTH = 2000  # output cap, applied to generated responses in Phase 6

CANNED_INVALID_RESPONSE = "Sorry, I couldn't understand that message. Could you rephrase?"
CANNED_PROFANITY_RESPONSE = (
    "I'd rather keep this conversation respectful — could you rephrase that?"
)
CANNED_GIBBERISH_RESPONSES = (
    "My keyboard-smash translator is out for lunch. Try a real question — like what Chaitanya actually builds.",
    "Bold linguistic experiment, but I only parse human. Ask me about Chaitanya's projects and we're back in business.",
    "I ran that through every decoder I have and got static. Try words — 'what's his stack?' works great.",
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


# --- gibberish gate -----------------------------------------------------
# Zero-LLM-cost keyboard-mash filter, checked after profanity. Tier 0
# (structural) is length-gated so short jargon like "C++?" or "k8s?" never
# gets close to the heuristics; Tier 1 (statistical) is a lenient backstop
# for longer mashes Tier 0 misses.

GIBBERISH_THRESHOLD = 0.7
GIBBERISH_MIN_LEN_FOR_STATISTICAL = 12
_VOWELS = set("aeiouy")
_ALLOWED_PUNCTUATION = set(".,?!'\"-+#/:()")

_garble_detector = EnsembleDetector(voting="average")  # built once at import


def _strip_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def _is_structural_gibberish(text: str) -> bool:
    # Repeated-char run: same char >= 5 times in a row, and it dominates a
    # short message (protects against "aaaaaaaaaaaaaa" without misfiring on
    # long legitimate messages that happen to be at the size cap).
    run_match = re.search(r"(.)\1{4,}", text)
    if run_match and len(text) <= 100:
        ch = run_match.group(1)
        non_space = [c for c in text if not c.isspace()]
        if non_space and non_space.count(ch) / len(non_space) > 0.5:
            return True

    # Symbol density: mostly punctuation/symbols outside the whitelist
    # (whitelist keeps "C++?" and "CI/CD?" safe), or no alnum chars at all
    # (catches pure-symbol spam like "!!!!!!@@@@####$$$$").
    if len(text) >= 6:
        symbol_count = sum(
            1 for c in text if not (c.isalnum() or c.isspace() or c in _ALLOWED_PUNCTUATION)
        )
        if symbol_count / len(text) > 0.5:
            return True
        if not any(c.isalnum() for c in text):
            return True

    # No-vowel words and single-token mash both operate on short messages
    # only — real keyboard-mash is short, and this keeps very long messages
    # (e.g. right at MAX_MESSAGE_LENGTH) from being penalized on vowel ratio
    # alone.
    if len(text) <= 100:
        # No-vowel words: multiple alphabetic tokens with almost no vowels.
        words = [w for w in re.findall(r"[^\s]+", text) if w.isalpha() and len(w) >= 4]
        if len(words) >= 2:
            vowelless = sum(1 for w in words if not any(c in _VOWELS for c in w.lower()))
            if vowelless / len(words) >= 0.8:
                return True

        # Single-token mash: one long alphabetic token with a very low vowel ratio.
        tokens = text.split()
        if len(tokens) == 1 and tokens[0].isalpha() and len(tokens[0]) >= 8:
            vowel_ratio = sum(1 for c in tokens[0].lower() if c in _VOWELS) / len(tokens[0])
            if vowel_ratio < 0.15:
                return True

    return False


GIBBERISH_MAX_LEN_FOR_STATISTICAL = 100  # real keyboard-mash is short; long
# messages that happen to score high (e.g. one char repeated to the size
# cap) are better left to a human/LLM than penalized just for length.


def _is_statistical_gibberish(text: str) -> bool:
    if not (GIBBERISH_MIN_LEN_FOR_STATISTICAL <= len(text) <= GIBBERISH_MAX_LEN_FOR_STATISTICAL):
        return False
    return _garble_detector.predict_proba(text) >= GIBBERISH_THRESHOLD


def _is_gibberish(text: str) -> bool:
    stripped = _strip_urls(text).strip()
    if not stripped:
        return False
    return _is_structural_gibberish(stripped) or _is_statistical_gibberish(stripped)


def check_input(messages: list[dict[str, str]]) -> GuardrailResult:
    """Validate and clean incoming chat history. Check order matters:
    shape -> length caps -> history truncation -> control-char strip ->
    empty/blank -> profanity -> gibberish. Blank, profanity, and gibberish
    rejection only look at the newest user message — stale profane, blank,
    or gibberish turns already sitting in client-held history must not
    reject an otherwise-clean new message."""
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

    # URLs can contain incidental substrings that better-profanity flags
    # (e.g. "foo/bar" inside a github.com path) — strip them before scanning,
    # same as the gibberish gate does.
    if profanity.contains_profanity(_strip_urls(newest["content"])):
        return _reject("profanity", CANNED_PROFANITY_RESPONSE)

    if _is_gibberish(newest["content"]):
        return _reject("gibberish", random.choice(CANNED_GIBBERISH_RESPONSES))

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


# --- rate limiting ------------------------------------------------------
# Render sits behind one proxy; the default key_func pattern does not read
# X-Forwarded-For, which would key every request to the proxy's IP. Take the
# LAST entry (appended by Render's own proxy) rather than the first (client-
# supplied and trivially spoofable).


def client_ip(request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[-1].strip() if fwd else (request.client.host if request.client else "unknown")


RATE_LIMITS = [parse("10/minute"), parse("30/day")]
_rate_storage = MemoryStorage()
_rate_limiter = MovingWindowRateLimiter(_rate_storage)


@dataclass
class RateLimitStatus:
    allowed: bool
    retry_after: int = 0


def check_rate_limit(key: str) -> RateLimitStatus:
    """Non-consuming check — does not count against the window."""
    for item in RATE_LIMITS:
        if not _rate_limiter.test(item, key):
            stats = _rate_limiter.get_window_stats(item, key)
            retry_after = max(1, int(stats.reset_time - time.time()) + 1)
            return RateLimitStatus(allowed=False, retry_after=retry_after)
    return RateLimitStatus(allowed=True)


def consume_rate_limit(key: str) -> None:
    for item in RATE_LIMITS:
        _rate_limiter.hit(item, key)


def reset_rate_limits() -> None:
    """Test hook: clears all rate-limit state."""
    _rate_storage.reset()
