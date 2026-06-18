"""The one way this project calls a language model.

Every call goes through ``ModelClient.chat``. It looks in the cache first. If the answer is not
there, it checks the free-tier limits, waits if a minute limit is reached, and stops if a daily
limit is reached. It never moves on to another model or service, and it never retries in a way
that could cost money: all keys belong to projects without billing.
"""

from __future__ import annotations

import logging
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx

from norsk_lov_rag.config import LLMConfig, ModelConfig
from norsk_lov_rag.llm.cache import ResponseCache, cache_key
from norsk_lov_rag.llm.providers import build_request, parse_answer, parse_rate_limit
from norsk_lov_rag.llm.quota import Limits, QuotaExhausted, QuotaTracker

log = logging.getLogger(__name__)

# A rough count of tokens from characters, used only to plan a call before it is sent.
CHARS_PER_TOKEN = 3
# How many times one call may wait out a minute limit before the run stops.
MAX_LIMITED_WAITS = 12


class MissingKey(RuntimeError):
    """The key for a provider is not in the environment or the .env file."""


class CacheMiss(RuntimeError):
    """Offline mode found no cached answer for a call."""


class ProviderError(RuntimeError):
    """The provider refused the call or failed after all retries."""


@dataclass
class ChatResult:
    text: str
    role: str
    model: str
    model_version: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str
    cached: bool
    key: str
    # Seconds the provider took to answer, measured when the call was made. Cached answers
    # keep the time of the original call.
    latency_seconds: float | None = None


def read_env_file(path: Path) -> dict[str, str]:
    """Read KEY=VALUE lines. Comments and blank lines are skipped."""
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        values[name.strip()] = value.strip().strip('"').strip("'")
    return values


def estimate_tokens(messages: list[dict]) -> int:
    return sum(len(m["content"]) for m in messages) // CHARS_PER_TOKEN + 1


class ModelClient:
    def __init__(
        self,
        cfg: LLMConfig,
        state_dir: Path,
        offline: bool = False,
        transport: httpx.BaseTransport | None = None,
        environ: dict[str, str] | None = None,
        clock=time.time,
        sleep=time.sleep,
    ) -> None:
        self.cfg = cfg
        self.offline = offline
        self.cache = ResponseCache(state_dir / "cache")
        limits = {
            self._quota_name(model): Limits(**model.limits.model_dump())
            for model in cfg.models.values()
        }
        self.quota = QuotaTracker(state_dir / "quota.json", limits, clock=clock)
        self.sleep = sleep
        self._environ = dict(os.environ) if environ is None else environ
        self._env_file = read_env_file(cfg.env_file)
        self._http = httpx.Client(timeout=cfg.timeout_seconds, transport=transport)
        self._warned: set[str] = set()

    @staticmethod
    def _quota_name(model: ModelConfig) -> str:
        return f"{model.provider}:{model.model}"

    def _api_key(self, name: str | None) -> str:
        if name is None:
            return ""
        key = self._environ.get(name) or self._env_file.get(name)
        if not key:
            raise MissingKey(f"{name} is not set in the environment or in the .env file")
        return key

    @contextmanager
    def cache_only(self):
        """Read answers from the cache only, for the rest of a stage after its quota is used."""
        before, self.offline = self.offline, True
        try:
            yield
        finally:
            self.offline = before

    def model_name(self, role: str) -> str:
        return self._quota_name(self.cfg.models[role])

    def chat(
        self,
        role: str,
        messages: list[dict],
        prompt_version: str,
        max_tokens: int,
        temperature: float = 0.0,
        json_mode: bool = False,
        schema: dict | None = None,
    ) -> ChatResult:
        if role not in self.cfg.models:
            raise KeyError(f"no model is configured for the role {role!r}")
        model = self.cfg.models[role]
        name = self._quota_name(model)
        settings = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "json_mode": json_mode,
            "params": model.params,
        }
        if schema is not None:
            settings["schema"] = schema
        key = cache_key(name, prompt_version, messages, settings)
        record = self.cache.get(key)
        if record is None:
            if self.offline:
                raise CacheMiss(f"{role}: no cached answer and offline mode is on")
            record = self._call(model, name, messages, max_tokens, temperature, json_mode, schema)
            self.cache.put(key, record)
            cached = False
        else:
            cached = True
        return ChatResult(
            text=record["text"],
            role=role,
            model=name,
            model_version=record["model_version"],
            prompt_tokens=record["prompt_tokens"],
            completion_tokens=record["completion_tokens"],
            total_tokens=record["total_tokens"],
            finish_reason=record["finish_reason"],
            cached=cached,
            key=key,
            latency_seconds=record.get("latency_seconds"),
        )

    def _call(
        self,
        model: ModelConfig,
        name: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        json_mode: bool,
        schema: dict | None = None,
    ) -> dict:
        provider = self.cfg.providers[model.provider]
        request = build_request(
            provider.style,
            provider.base_url,
            self._api_key(provider.api_key_env),
            model.model,
            messages,
            max_tokens,
            temperature,
            json_mode,
            model.params,
            schema,
        )
        # Groq reserves the full max_tokens against the minute budget, so plan with it.
        planned = estimate_tokens(messages) + max_tokens
        failures = limited = 0
        while True:
            wait = self.quota.wait_seconds(name, planned)
            if wait > 0:
                log.info("%s: minute limit reached, waiting %.0f s", name, wait)
                self.sleep(wait)
                continue
            started = time.perf_counter()
            try:
                response = self._http.post(
                    request.url,
                    headers=request.headers,
                    json=request.body,
                    timeout=provider.timeout_seconds or self.cfg.timeout_seconds,
                )
            except httpx.TransportError as exc:
                failures = self._failed(model, name, failures, type(exc).__name__)
                continue
            latency = time.perf_counter() - started
            if response.status_code == 200:
                answer = parse_answer(provider.style, response.json(), dict(response.headers))
                self.quota.record(name, answer.total_tokens)
                self._check_reported_limits(model, name, answer.reported_limits)
                return {
                    "text": answer.text,
                    "model_version": answer.model_version,
                    "prompt_tokens": answer.prompt_tokens,
                    "completion_tokens": answer.completion_tokens,
                    "total_tokens": answer.total_tokens,
                    "finish_reason": answer.finish_reason,
                    "latency_seconds": round(latency, 3),
                }
            if response.status_code == 429:
                limit = parse_rate_limit(provider.style, _json(response), dict(response.headers))
                if limit.daily:
                    raise QuotaExhausted(f"{name}: the provider says the daily limit is reached")
                wait = limit.retry_after if limit.retry_after is not None else 60.0
                if wait > self.cfg.max_wait_seconds:
                    raise QuotaExhausted(f"{name}: the provider asks for a wait of {wait:.0f} s")
                limited += 1
                if limited > MAX_LIMITED_WAITS:
                    raise QuotaExhausted(f"{name}: still limited after {limited - 1} waits")
                log.info("%s: limited by the provider, waiting %.0f s", name, wait)
                self.sleep(wait + 0.5)
                continue
            if response.status_code >= 500:
                failures = self._failed(model, name, failures, f"HTTP {response.status_code}")
                continue
            detail = str(_json(response).get("error", {}).get("message", ""))[:300]
            raise ProviderError(f"{name}: HTTP {response.status_code}: {detail}")

    def _failed(self, model: ModelConfig, name: str, failures: int, reason: str) -> int:
        """Count a failed call where the provider counts it, then wait before the next try."""
        if model.limits.counts_failed:
            self.quota.record(name, 0)
        failures += 1
        if failures > (model.max_retries or self.cfg.max_retries):
            raise ProviderError(f"{name}: {reason} after {failures - 1} retries")
        wait = min(900.0, model.retry_base_seconds * 2 ** (failures - 1)) * (0.5 + random.random())
        log.warning("%s: %s, retrying in %.0f s", name, reason, wait)
        self.sleep(wait)
        return failures

    def _check_reported_limits(self, model: ModelConfig, name: str, reported: dict) -> None:
        """Warn once when the provider reports lower limits than the configuration."""
        configured = model.limits.model_dump()
        for field, value in reported.items():
            mine = configured.get(field)
            if mine is not None and value < mine and (name, field) not in self._warned:
                self._warned.add((name, field))
                log.warning("%s: provider reports %s=%d, config says %d", name, field, value, mine)

    def usage(self) -> dict[str, dict]:
        """Requests and tokens used in the last 24 hours, with the daily limits."""
        report = {}
        for role, model in self.cfg.models.items():
            name = self._quota_name(model)
            report[role] = {
                "model": name,
                **self.quota.used_today(name),
                "requests_per_day": model.limits.requests_per_day,
                "tokens_per_day": model.limits.tokens_per_day,
            }
        return report

    def close(self) -> None:
        self._http.close()


def _json(response: httpx.Response) -> dict:
    try:
        data = response.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}
