"""Model client: cache, free-tier limits, retries and the two provider styles, with no network."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from norsk_lov_rag.config import LLMConfig, load_settings
from norsk_lov_rag.llm import CacheMiss, MissingKey, ModelClient, ProviderError, QuotaExhausted
from norsk_lov_rag.llm.cache import cache_key
from norsk_lov_rag.llm.client import read_env_file
from norsk_lov_rag.llm.providers import build_request, parse_answer, parse_rate_limit
from norsk_lov_rag.llm.quota import Limits, QuotaTracker

MESSAGES = [
    {"role": "system", "content": "Du er en hjelpsom assistent."},
    {"role": "user", "content": "Hva er hovedstaden i Norge?"},
]


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds

    waits: list[float]


def config(limits: dict | None = None) -> LLMConfig:
    limits = limits or {"requests_per_minute": 2, "requests_per_day": 5,
                        "tokens_per_minute": 1000, "tokens_per_day": 3000}  # fmt: skip
    return LLMConfig(
        env_file=Path("does-not-exist.env"),
        max_retries=2,
        providers={
            "groq": {"style": "openai", "base_url": "https://groq.test/v1", "api_key_env": "K1"},
            "google": {"style": "gemini", "base_url": "https://g.test/v1beta", "api_key_env": "K2"},
        },  # fmt: skip
        models={
            "writer": {
                "provider": "groq",
                "model": "m1",
                "limits": limits,
                "params": {"reasoning_effort": "low"},
            },
            "reviewer": {"provider": "google", "model": "g1", "limits": limits},
        },  # fmt: skip
    )


def openai_ok(text: str = "Oslo", total: int = 50) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "m1",
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 40, "completion_tokens": total - 40, "total_tokens": total},
        },
    )


def make_client(tmp_path: Path, handler, clock: Clock | None = None, **kwargs) -> ModelClient:
    clock = clock or Clock()
    clock.waits = []
    return ModelClient(
        config(**kwargs),
        tmp_path,
        transport=httpx.MockTransport(handler),
        environ={"K1": "secret-1", "K2": "secret-2"},
        clock=clock,
        sleep=clock.sleep,
    )


def test_answers_are_cached_and_offline_mode_reads_only_the_cache(tmp_path: Path) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer secret-1"
        return openai_ok()

    client = make_client(tmp_path, handler)
    first = client.chat("writer", MESSAGES, "v1", max_tokens=100)
    second = client.chat("writer", MESSAGES, "v1", max_tokens=100)
    assert first.text == second.text == "Oslo"
    assert (first.cached, second.cached) == (False, True)
    assert len(calls) == 1 and calls[0]["reasoning_effort"] == "low"
    assert client.usage()["writer"]["requests"] == 1
    assert client.usage()["writer"]["tokens"] == 50

    offline = ModelClient(config(), tmp_path, offline=True, environ={})
    assert offline.chat("writer", MESSAGES, "v1", max_tokens=100).cached
    with pytest.raises(CacheMiss):
        offline.chat("writer", MESSAGES, "v2", max_tokens=100)


def test_cache_key_changes_with_prompt_version_model_and_settings() -> None:
    base = cache_key("groq:m1", "v1", MESSAGES, {"max_tokens": 10})
    assert base == cache_key("groq:m1", "v1", MESSAGES, {"max_tokens": 10})
    assert base != cache_key("groq:m1", "v2", MESSAGES, {"max_tokens": 10})
    assert base != cache_key("groq:m2", "v1", MESSAGES, {"max_tokens": 10})
    assert base != cache_key("groq:m1", "v1", MESSAGES, {"max_tokens": 11})


def test_minute_limit_waits_and_daily_limit_stops(tmp_path: Path) -> None:
    clock = Clock()
    client = make_client(tmp_path, lambda request: openai_ok(), clock=clock)
    for number in range(5):
        client.chat("writer", MESSAGES, f"v{number}", max_tokens=100)
    # Two requests per minute: the third and fifth calls had to wait for the next minute.
    assert len(clock.waits) == 2 and all(wait > 50 for wait in clock.waits)
    with pytest.raises(QuotaExhausted):
        client.chat("writer", MESSAGES, "v5", max_tokens=100)


def test_quota_counts_survive_a_new_run_and_unknown_models_are_refused(tmp_path: Path) -> None:
    clock = Clock()
    limits = {"m": Limits(requests_per_minute=10, requests_per_day=2, tokens_per_day=100)}
    tracker = QuotaTracker(tmp_path / "quota.json", limits, clock=clock)
    tracker.record("m", 60)
    again = QuotaTracker(tmp_path / "quota.json", limits, clock=clock)
    assert again.used_today("m") == {"requests": 1, "tokens": 60}
    with pytest.raises(QuotaExhausted):
        again.wait_seconds("m", 50)
    with pytest.raises(KeyError):
        again.wait_seconds("other", 1)
    clock.now += 24 * 60 * 60 + 1
    assert again.wait_seconds("m", 50) == 0.0


def test_provider_minute_limit_is_waited_out_but_daily_limit_stops(tmp_path: Path) -> None:
    replies = [
        httpx.Response(429, headers={"retry-after": "7"},
                       json={"error": {"message": "Rate limit reached on requests per minute"}}),
        openai_ok(),
        httpx.Response(429, json={"error": {"message": "Rate limit reached for model `m1` "
                                            "on tokens per day (TPD): Limit 200000"}}),
    ]  # fmt: skip
    clock = Clock()
    client = make_client(tmp_path, lambda request: replies.pop(0), clock=clock)
    assert client.chat("writer", MESSAGES, "v1", max_tokens=100).text == "Oslo"
    assert clock.waits == [7.5]
    with pytest.raises(QuotaExhausted):
        client.chat("writer", MESSAGES, "v2", max_tokens=100)


def test_server_errors_are_retried_then_reported(tmp_path: Path) -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(1)
        return httpx.Response(503, json={"error": {"message": "high demand"}})

    client = make_client(tmp_path, handler)
    with pytest.raises(ProviderError):
        client.chat("writer", MESSAGES, "v1", max_tokens=100)
    assert len(seen) == 3  # one call and two retries


def test_client_errors_are_not_retried_and_hide_the_key(tmp_path: Path) -> None:
    client = make_client(
        tmp_path, lambda request: httpx.Response(400, json={"error": {"message": "bad model"}})
    )
    with pytest.raises(ProviderError) as error:
        client.chat("writer", MESSAGES, "v1", max_tokens=100)
    assert "bad model" in str(error.value) and "secret" not in str(error.value)


def test_missing_key_is_named_without_a_value(tmp_path: Path) -> None:
    client = ModelClient(config(), tmp_path, environ={}, transport=httpx.MockTransport(None))
    with pytest.raises(MissingKey, match="K1"):
        client.chat("writer", MESSAGES, "v1", max_tokens=100)


def test_gemini_request_and_answer_are_translated(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models/g1:generateContent")
        assert request.headers["x-goog-api-key"] == "secret-2"
        body = json.loads(request.content)
        assert body["systemInstruction"]["parts"][0]["text"].startswith("Du er")
        assert [c["role"] for c in body["contents"]] == ["user"]
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        return httpx.Response(
            200,
            json={
                "modelVersion": "g1-001",
                "candidates": [{
                    "content": {"parts": [{"text": "tenker", "thought": True},
                                          {"text": '{"svar": "Oslo"}'}]},
                    "finishReason": "STOP",
                }],
                "usageMetadata": {"promptTokenCount": 20, "totalTokenCount": 90},
            },
        )  # fmt: skip

    client = make_client(tmp_path, handler)
    result = client.chat("reviewer", MESSAGES, "v1", max_tokens=200, json_mode=True)
    assert json.loads(result.text) == {"svar": "Oslo"}
    assert (result.model_version, result.total_tokens, result.finish_reason) == (
        "g1-001", 90, "stop")  # fmt: skip


def test_gemini_rate_limit_details_are_read() -> None:
    def body(quota_id: str) -> dict:
        return {"error": {"message": "quota", "details": [
            {"violations": [{"quotaId": quota_id}]}, {"retryDelay": "33s"}]}}  # fmt: skip

    minute = parse_rate_limit("gemini", body("GenerateRequestsPerMinutePerProject-FreeTier"), {})
    day = parse_rate_limit("gemini", body("GenerateRequestsPerDayPerProjectPerModel-FreeTier"), {})
    assert (minute.daily, minute.retry_after) == (False, 33.0)
    assert day.daily
    assert parse_rate_limit("openai", {}, {"retry-after": "1m30s"}).retry_after == 90.0


def test_openai_style_body_and_reported_limits() -> None:
    request = build_request("openai", "https://x/v1", "k", "m", MESSAGES, 300, 0.2, True,
                            {"reasoning_effort": "low"})  # fmt: skip
    assert request.body["max_completion_tokens"] == 300
    assert request.body["response_format"] == {"type": "json_object"}
    answer = parse_answer(
        "openai",
        {"choices": [{"message": {"content": None}, "finish_reason": "length"}], "usage": {}},
        {"x-ratelimit-limit-requests": "1000", "x-ratelimit-limit-tokens": "8000"},
    )
    assert (answer.text, answer.finish_reason) == ("", "length")
    assert answer.reported_limits == {"requests_per_day": 1000, "tokens_per_minute": 8000}


def test_env_file_is_read_and_default_config_names_free_models(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text('# keys\nA_KEY="abc"\nB_KEY=def\n\nnot a line\n', encoding="utf-8")
    assert read_env_file(env) == {"A_KEY": "abc", "B_KEY": "def"}
    llm = load_settings().llm
    assert set(llm.models) >= {"writer", "checker", "reviewer"}
    for model in llm.models.values():
        assert model.provider in llm.providers
        assert model.limits.requests_per_day > 0


def test_failed_calls_do_not_count_against_the_day(tmp_path: Path) -> None:
    replies = [httpx.Response(503, json={"error": {"message": "high demand"}}), openai_ok()]
    client = make_client(tmp_path, lambda request: replies.pop(0))
    client.chat("writer", MESSAGES, "v1", max_tokens=100)
    assert client.usage()["writer"]["requests"] == 1


def test_limits_that_refill_allow_steady_calls_and_stop_when_empty(tmp_path: Path) -> None:
    clock = Clock()
    limits = {
        "m": Limits(
            requests_per_minute=60,
            requests_per_day=1000,
            tokens_per_minute=600,
            tokens_per_day=1000,
            refill=True,
        )
    }
    tracker = QuotaTracker(tmp_path / "quota.json", limits, clock=clock)
    assert tracker.wait_seconds("m", 300) == 0.0
    tracker.record("m", 500)
    # 100 tokens are left of 600, and the limit refills by 10 tokens a second.
    assert 19 < tracker.wait_seconds("m", 300) < 21
    clock.now += 30
    assert tracker.wait_seconds("m", 300) == 0.0
    tracker.record("m", 480)
    with pytest.raises(QuotaExhausted):
        tracker.wait_seconds("m", 100)
    clock.now += 6 * 60 * 60  # a quarter of the day's tokens have come back
    assert tracker.wait_seconds("m", 100) == 0.0


def test_daily_count_can_start_at_midnight_in_a_time_zone(tmp_path: Path) -> None:
    clock = Clock()
    clock.now = datetime(2026, 11, 9, 7, 59, tzinfo=UTC).timestamp()  # 23:59 in California
    limits = {
        "g": Limits(
            requests_per_minute=100, requests_per_day=1, day_resets_tz="America/Los_Angeles"
        )
    }
    tracker = QuotaTracker(tmp_path / "quota.json", limits, clock=clock)
    tracker.record("g", 10)
    with pytest.raises(QuotaExhausted):
        tracker.wait_seconds("g", 10)
    clock.now += 120  # now past midnight there
    assert tracker.wait_seconds("g", 10) == 0.0


def test_failed_calls_count_where_the_provider_counts_them(tmp_path: Path) -> None:
    cfg = config({"requests_per_minute": 10, "requests_per_day": 5, "counts_failed": True})
    cfg.models["writer"].retry_base_seconds = 30
    replies = [httpx.Response(503, json={"error": {"message": "busy"}}), openai_ok()]
    clock = Clock()
    clock.waits = []
    client = ModelClient(
        cfg,
        tmp_path,
        transport=httpx.MockTransport(lambda request: replies.pop(0)),
        environ={"K1": "k"},
        clock=clock,
        sleep=clock.sleep,
    )
    client.chat("writer", MESSAGES, "v1", max_tokens=100)
    assert client.usage()["writer"]["requests"] == 2
    assert 15 <= clock.waits[0] <= 45
