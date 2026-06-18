"""Keep model calls inside the free-tier limits.

Every model has limits per minute and per day, for requests and for tokens. The tracker follows
the rules of each provider: Groq's limits refill steadily, and Google's free daily count starts
again at midnight in California. When a minute limit is reached the tracker says how long to
wait. When a day limit is reached it raises, and the run stops. Nothing here ever switches to
another model or service.

The counts are kept in a small file under the ignored data folder, so that a new run the same
day continues from what was already used.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

MINUTE = 60.0
DAY = 24 * 60 * 60.0


class QuotaExhausted(RuntimeError):
    """The free quota for a model is used up for now. The run must stop, not pay."""


@dataclass(frozen=True)
class Limits:
    requests_per_minute: int
    requests_per_day: int
    tokens_per_minute: int | None = None
    tokens_per_day: int | None = None
    # True when the limits refill steadily. False counts what was used inside the window.
    refill: bool = False
    # The time zone whose midnight starts a new day. None means a rolling 24 hours.
    day_resets_tz: str | None = None
    # True when the provider also counts calls that failed.
    counts_failed: bool = False


def _one(_tokens: float) -> float:
    return 1.0


def _same(tokens: float) -> float:
    return tokens


class QuotaTracker:
    def __init__(self, state_path: Path, limits: dict[str, Limits], clock=time.time) -> None:
        self.state_path = state_path
        self.limits = limits
        self.clock = clock
        self.events: dict[str, list[list[float]]] = {}
        if state_path.exists():
            self.events = json.loads(state_path.read_text(encoding="utf-8"))

    def _recent(self, model: str, window: float) -> tuple[int, int, float]:
        """Requests, tokens and the oldest timestamp inside the window."""
        cutoff = self.clock() - window
        events = [event for event in self.events.get(model, []) if event[0] > cutoff]
        oldest = min((event[0] for event in events), default=self.clock())
        return len(events), int(sum(event[1] for event in events)), oldest

    def _room(
        self, model: str, capacity: int, window: float, cost: Callable[[float], float]
    ) -> float:
        """What is left of a limit that refills evenly over the window."""
        rate = capacity / window
        level, previous = float(capacity), None
        for when, tokens in sorted(self.events.get(model, [])):
            if previous is not None:
                level = min(capacity, level + (when - previous) * rate)
            level -= cost(tokens)
            previous = when
        if previous is not None:
            level = min(capacity, level + (self.clock() - previous) * rate)
        return level

    def _wait_with_refill(self, model: str, tokens: int, limits: Limits) -> float:
        if self._room(model, limits.requests_per_day, DAY, _one) < 1:
            raise QuotaExhausted(f"{model}: the daily request limit is used up")
        if (
            limits.tokens_per_day is not None
            and self._room(model, limits.tokens_per_day, DAY, _same) < tokens
        ):
            raise QuotaExhausted(f"{model}: the daily token limit is used up")
        waits = [0.0]
        room = self._room(model, limits.requests_per_minute, MINUTE, _one)
        if room < 1:
            waits.append((1 - room) * MINUTE / limits.requests_per_minute + 0.05)
        if limits.tokens_per_minute is not None:
            room = self._room(model, limits.tokens_per_minute, MINUTE, _same)
            if room < tokens:
                waits.append((tokens - room) * MINUTE / limits.tokens_per_minute + 0.05)
        return max(waits)

    def _day_window(self, limits: Limits) -> float:
        if limits.day_resets_tz is None:
            return DAY
        now = datetime.fromtimestamp(self.clock(), ZoneInfo(limits.day_resets_tz))
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return (now - midnight).total_seconds()

    def wait_seconds(self, model: str, tokens: int) -> float:
        """Seconds to wait before a call of this size may go out. Raises when the day is used up."""
        if model not in self.limits:
            raise KeyError(f"no free-tier limits are configured for {model}, so it is not called")
        limits = self.limits[model]
        if limits.tokens_per_minute is not None and tokens > limits.tokens_per_minute:
            raise ValueError(f"{model}: one call of {tokens} tokens is above the minute limit")
        if limits.refill:
            return self._wait_with_refill(model, tokens, limits)
        day_requests, day_tokens, _ = self._recent(model, self._day_window(limits))
        if day_requests + 1 > limits.requests_per_day:
            raise QuotaExhausted(f"{model}: the daily request limit is used up")
        if limits.tokens_per_day is not None and day_tokens + tokens > limits.tokens_per_day:
            raise QuotaExhausted(f"{model}: the daily token limit is used up")
        requests, minute_tokens, oldest = self._recent(model, MINUTE)
        over_requests = requests + 1 > limits.requests_per_minute
        over_tokens = (
            limits.tokens_per_minute is not None
            and minute_tokens + tokens > limits.tokens_per_minute
        )
        if over_requests or over_tokens:
            return max(0.0, oldest + MINUTE - self.clock()) + 0.5
        return 0.0

    def record(self, model: str, tokens: int) -> None:
        """Count one finished call with the tokens the provider reported."""
        now = self.clock()
        events = [event for event in self.events.get(model, []) if event[0] > now - DAY]
        events.append([now, float(tokens)])
        # Other processes may call models at the same time, the same model included. Keep what
        # they wrote and add this call, so that no call is lost from the count.
        if self.state_path.exists():
            on_disk = json.loads(self.state_path.read_text(encoding="utf-8") or "{}")
            self.events.update({name: e for name, e in on_disk.items() if name != model})
            seen = {tuple(event) for event in events}
            events += [
                e for e in on_disk.get(model, []) if tuple(e) not in seen and e[0] > now - DAY
            ]
            events.sort()
        self.events[model] = events
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.events), encoding="utf-8")

    def used_today(self, model: str) -> dict[str, int]:
        requests, tokens, _ = self._recent(model, DAY)
        return {"requests": requests, "tokens": tokens}
