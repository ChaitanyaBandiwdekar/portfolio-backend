"""Unit tests for app/guardrails.py — the pre-LLM defense layer.

No real APIs are involved; guardrails.py must not even import llm.py, so a
zero-LLM-calls assertion for the profanity path is just "no such import
exists" plus asserting the returned GuardrailResult is not ok.
"""

from __future__ import annotations

import datetime

import pytest

from app import guardrails


def _msg(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


# --- shape / empty -----------------------------------------------------


def test_empty_message_list_rejected():
    result = guardrails.check_input([])
    assert result.ok is False


def test_blank_content_rejected():
    result = guardrails.check_input([_msg("user", "   ")])
    assert result.ok is False


# --- length caps ---------------------------------------------------------


def test_oversize_message_rejected():
    oversize = "x" * (guardrails.MAX_MESSAGE_LENGTH + 1)
    result = guardrails.check_input([_msg("user", oversize)])
    assert result.ok is False
    assert result.reason == "oversize_message"


def test_message_at_cap_accepted():
    ok_length = "x" * guardrails.MAX_MESSAGE_LENGTH
    result = guardrails.check_input([_msg("user", ok_length)])
    assert result.ok is True


# --- history truncation ---------------------------------------------------


def test_oversize_history_truncated_keeps_newest():
    history = [_msg("user", f"turn {i}") for i in range(15)]
    result = guardrails.check_input(history)
    assert result.ok is True
    assert len(result.messages) == guardrails.MAX_HISTORY
    assert result.messages[-1]["content"] == "turn 14"
    assert result.messages[0]["content"] == "turn 5"


# --- control characters ---------------------------------------------------


def test_control_char_garbage_stripped():
    dirty = "hello\x00\x01world"
    result = guardrails.check_input([_msg("user", dirty)])
    assert result.ok is True
    assert "\x00" not in result.messages[0]["content"]
    assert "\x01" not in result.messages[0]["content"]


def test_pure_control_char_garbage_rejected():
    dirty = "\x00\x01\x02\x03"
    result = guardrails.check_input([_msg("user", dirty)])
    assert result.ok is False


def test_newline_and_tab_preserved():
    text = "line one\nline two\ttabbed"
    result = guardrails.check_input([_msg("user", text)])
    assert result.ok is True
    assert result.messages[0]["content"] == text


# --- profanity --------------------------------------------------------


def test_profanity_blocked_with_canned_response():
    result = guardrails.check_input([_msg("user", "you are a fucking idiot")])
    assert result.ok is False
    assert result.reason == "profanity"
    assert result.canned_response


def test_no_llm_import_in_guardrails_module():
    # guardrails.py must not import llm.py at all — profanity/rejection paths
    # spend zero Gemini calls by construction, not by mocking.
    assert "app.llm" not in guardrails.__dict__
    src = guardrails.__file__
    with open(src, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert not stripped.startswith("from app.llm") and not stripped.startswith(
                "import app.llm"
            )


def test_clean_message_passes():
    result = guardrails.check_input([_msg("user", "what did he work on at Acme?")])
    assert result.ok is True


def test_stale_profane_old_turn_clean_newest_passes():
    history = [
        _msg("user", "you are a fucking idiot"),
        _msg("assistant", "Let's keep things civil."),
        _msg("user", "what did he work on at Acme?"),
    ]
    result = guardrails.check_input(history)
    assert result.ok is True


def test_profane_newest_message_rejected():
    history = [
        _msg("user", "what did he work on at Acme?"),
        _msg("assistant", "He worked on several projects."),
        _msg("user", "you are a fucking idiot"),
    ]
    result = guardrails.check_input(history)
    assert result.ok is False
    assert result.reason == "profanity"


def test_blank_old_turn_clean_newest_passes():
    history = [
        _msg("user", "   "),
        _msg("assistant", "Could you rephrase that?"),
        _msg("user", "what did he work on at Acme?"),
    ]
    result = guardrails.check_input(history)
    assert result.ok is True


def test_blank_newest_message_rejected():
    history = [
        _msg("user", "what did he work on at Acme?"),
        _msg("assistant", "He worked on several projects."),
        _msg("user", "   "),
    ]
    result = guardrails.check_input(history)
    assert result.ok is False


# --- greeting fast path ---------------------------------------------------


@pytest.mark.parametrize("text", ["hi", "Hello", " hey ", "Hi!", "YO"])
def test_greeting_exact_match_positive(text):
    assert guardrails.is_greeting(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "hi, what did he work on at Acme?",
        "hello there, tell me about the project",
        "hey can you help",
        "",
        "random question",
    ],
)
def test_greeting_not_substring_matched(text):
    assert guardrails.is_greeting(text) is False


# --- global daily cap ---------------------------------------------------


def test_daily_cap_increments_and_blocks(monkeypatch):
    monkeypatch.setattr(guardrails, "_daily_count", 0)
    monkeypatch.setattr(guardrails, "_daily_date", datetime.date.today())
    monkeypatch.setattr(guardrails, "GLOBAL_DAILY_CAP", 2)

    assert guardrails.check_daily_cap() is True
    assert guardrails.check_daily_cap() is True
    assert guardrails.check_daily_cap() is False


def test_daily_cap_resets_on_date_change(monkeypatch):
    monkeypatch.setattr(guardrails, "_daily_count", 0)
    monkeypatch.setattr(guardrails, "_daily_date", datetime.date(2020, 1, 1))
    monkeypatch.setattr(guardrails, "GLOBAL_DAILY_CAP", 1)

    assert guardrails.check_daily_cap() is True  # uses up the cap on "old" date
    assert guardrails.check_daily_cap() is False  # still same in-process date

    class FakeDate(datetime.date):
        @classmethod
        def today(cls):
            return datetime.date(2020, 1, 2)

    monkeypatch.setattr(guardrails, "date", FakeDate)
    assert guardrails.check_daily_cap() is True  # new date, counter reset


# --- client_ip key_func ---------------------------------------------------


class _FakeClient:
    def __init__(self, host):
        self.host = host


class _FakeRequest:
    def __init__(self, headers, client_host="1.2.3.4"):
        self.headers = headers
        self.client = _FakeClient(client_host) if client_host else None


def test_client_ip_uses_last_xff_entry():
    req = _FakeRequest({"x-forwarded-for": "203.0.113.5, 10.0.0.1"})
    assert guardrails.client_ip(req) == "10.0.0.1"


def test_client_ip_spoofed_multi_entry_still_takes_last():
    # A client sending its own spoofed X-Forwarded-For can rotate fake IPs
    # for all but the last hop — the last entry is the one the trusted proxy
    # (Render) appended, so that's the one we key on.
    req = _FakeRequest({"x-forwarded-for": "9.9.9.9, 8.8.8.8, 172.16.0.7"})
    assert guardrails.client_ip(req) == "172.16.0.7"


def test_client_ip_missing_header_falls_back_to_client_host():
    req = _FakeRequest({}, client_host="5.6.7.8")
    assert guardrails.client_ip(req) == "5.6.7.8"


def test_client_ip_missing_everything_returns_unknown():
    req = _FakeRequest({}, client_host=None)
    assert guardrails.client_ip(req) == "unknown"


# --- rate limits configuration ---------------------------------------------


def test_rate_limits_configured_10_per_minute_30_per_day():
    per_minute, per_day = guardrails.RATE_LIMITS
    assert per_minute.amount == 10 and per_minute.GRANULARITY.name == "minute"
    assert per_day.amount == 30 and per_day.GRANULARITY.name == "day"


# --- rate limiter unit tests -------------------------------------------------


@pytest.fixture
def _small_rate_limits(monkeypatch):
    from limits import parse

    monkeypatch.setattr(guardrails, "RATE_LIMITS", [parse("2/minute"), parse("2/day")])
    guardrails.reset_rate_limits()
    yield
    guardrails.reset_rate_limits()


def test_rate_limit_allowed_when_fresh(_small_rate_limits):
    status = guardrails.check_rate_limit("1.2.3.4")
    assert status.allowed is True


def test_rate_limit_blocked_after_consuming_quota(_small_rate_limits):
    guardrails.consume_rate_limit("1.2.3.4")
    guardrails.consume_rate_limit("1.2.3.4")

    status = guardrails.check_rate_limit("1.2.3.4")

    assert status.allowed is False


def test_rate_limit_retry_after_is_positive(_small_rate_limits):
    guardrails.consume_rate_limit("1.2.3.4")
    guardrails.consume_rate_limit("1.2.3.4")

    status = guardrails.check_rate_limit("1.2.3.4")

    assert status.retry_after >= 1


def test_reset_rate_limits_restores_quota(_small_rate_limits):
    guardrails.consume_rate_limit("1.2.3.4")
    guardrails.consume_rate_limit("1.2.3.4")
    assert guardrails.check_rate_limit("1.2.3.4").allowed is False

    guardrails.reset_rate_limits()

    assert guardrails.check_rate_limit("1.2.3.4").allowed is True


# --- gibberish gate ----------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "asdfghjkl qweruiop",
        "aaaaaaaaaaaaaa",
        "!!!!!!@@@@####$$$$",
        "xjfkdls dkfjslkd fjsdlkf",
        "qwrtpsdfghjkl",
    ],
)
def test_gibberish_rejected_with_canned_response(text):
    result = guardrails.check_input([_msg("user", text)])
    assert result.ok is False
    assert result.reason == "gibberish"
    assert result.canned_response in guardrails.CANNED_GIBBERISH_RESPONSES


@pytest.mark.parametrize(
    "text",
    [
        "What stack?",
        "pgvector?",
        "C++?",
        "k8s?",
        "CI/CD experience?",
        "tell me about https://github.com/foo/bar",
        "What did Chaitanya build with FastAPI?",
    ],
)
def test_legit_jargon_not_flagged_as_gibberish(text):
    result = guardrails.check_input([_msg("user", text)])
    assert result.ok is True


def test_stale_gibberish_old_turn_clean_newest_passes():
    history = [
        _msg("user", "asdfghjkl qweruiop"),
        _msg("assistant", "Could you rephrase that?"),
        _msg("user", "what did he work on at Acme?"),
    ]
    result = guardrails.check_input(history)
    assert result.ok is True
