"""Turn one chat request into the HTTP call of a provider, and its answer back into one shape.

Three styles are supported: the OpenAI-compatible chat API (used by Groq), Google's Gemini API
and a local Ollama server. All get the same input and give the same output, so the rest of the
code does not care which service answered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

STYLES = ("openai", "gemini", "ollama")


@dataclass
class Request:
    url: str
    headers: dict[str, str]
    body: dict[str, Any]


@dataclass
class Answer:
    text: str
    model_version: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str
    reported_limits: dict[str, int] = field(default_factory=dict)


@dataclass
class RateLimit:
    """What a 429 answer says: whether the day is used up, and how long to wait otherwise."""

    daily: bool
    retry_after: float | None
    message: str


def build_request(
    style: str,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    json_mode: bool,
    params: dict[str, Any],
    schema: dict | None = None,
) -> Request:
    if style == "ollama":
        options: dict[str, Any] = {"temperature": temperature, "num_predict": max_tokens}
        options.update(params.get("options", {}))
        body = {"model": model, "messages": messages, "stream": False, "options": options}
        if schema is not None:
            body["format"] = schema
        elif json_mode:
            body["format"] = "json"
        body.update({k: v for k, v in params.items() if k != "options"})
        return Request(url=f"{base_url}/api/chat", headers={}, body=body)
    if style == "openai":
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        body.update(params)
        return Request(
            url=f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            body=body,
        )
    if style == "gemini":
        system = [m["content"] for m in messages if m["role"] == "system"]
        contents = [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            }
            for m in messages
            if m["role"] != "system"
        ]
        generation: dict[str, Any] = {"temperature": temperature, "maxOutputTokens": max_tokens}
        if json_mode:
            generation["responseMimeType"] = "application/json"
        generation.update(params)
        body = {"contents": contents, "generationConfig": generation}
        if system:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system)}]}
        return Request(
            url=f"{base_url}/models/{model}:generateContent",
            headers={"x-goog-api-key": api_key},
            body=body,
        )
    raise ValueError(f"unknown provider style {style!r}, expected one of {STYLES}")


def _finish(reason: str | None) -> str:
    reason = (reason or "").lower()
    if reason in ("stop", "end_turn"):
        return "stop"
    if reason in ("length", "max_tokens"):
        return "length"
    return reason or "unknown"


def parse_answer(style: str, data: dict, headers: dict[str, str]) -> Answer:
    if style == "ollama":
        prompt = int(data.get("prompt_eval_count", 0))
        completion = int(data.get("eval_count", 0))
        return Answer(
            text=(data.get("message") or {}).get("content", ""),
            model_version=data.get("model", ""),
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            finish_reason=_finish(data.get("done_reason")),
        )
    if style == "openai":
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        limits = {}
        for header, name in (
            ("x-ratelimit-limit-requests", "requests_per_day"),
            ("x-ratelimit-limit-tokens", "tokens_per_minute"),
        ):
            if headers.get(header, "").isdigit():
                limits[name] = int(headers[header])
        return Answer(
            text=choice["message"].get("content") or "",
            model_version=data.get("model", ""),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
            finish_reason=_finish(choice.get("finish_reason")),
            reported_limits=limits,
        )
    if style == "gemini":
        usage = data.get("usageMetadata") or {}
        candidates = data.get("candidates") or []
        if not candidates:
            blocked = (data.get("promptFeedback") or {}).get("blockReason", "no answer")
            text, finish = "", f"blocked:{blocked}".lower()
        else:
            parts = (candidates[0].get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            finish = _finish(candidates[0].get("finishReason"))
        prompt = int(usage.get("promptTokenCount", 0))
        total = int(usage.get("totalTokenCount", 0))
        return Answer(
            text=text,
            model_version=data.get("modelVersion", ""),
            prompt_tokens=prompt,
            completion_tokens=max(0, total - prompt),
            total_tokens=total,
            finish_reason=finish,
        )
    raise ValueError(f"unknown provider style {style!r}, expected one of {STYLES}")


def _seconds(text: str) -> float | None:
    """Read waits such as '33s', '1.5s', '7m12.5s' or a plain number of seconds."""
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    match = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", text)
    if not match or not any(match.groups()):
        return None
    hours, minutes, seconds = (float(g) if g else 0.0 for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def parse_rate_limit(style: str, data: dict, headers: dict[str, str]) -> RateLimit:
    retry_after = _seconds(headers.get("retry-after", ""))
    error = data.get("error") or {}
    message = str(error.get("message", ""))[:300]
    if style == "gemini":
        daily = False
        for detail in error.get("details") or []:
            for violation in detail.get("violations") or []:
                if "perday" in str(violation.get("quotaId", "")).lower():
                    daily = True
            if "retryDelay" in detail:
                retry_after = _seconds(str(detail["retryDelay"])) or retry_after
        return RateLimit(daily=daily, retry_after=retry_after, message=message)
    lowered = message.lower()
    daily = "per day" in lowered or "(tpd)" in lowered or "(rpd)" in lowered
    return RateLimit(daily=daily, retry_after=retry_after, message=message)
