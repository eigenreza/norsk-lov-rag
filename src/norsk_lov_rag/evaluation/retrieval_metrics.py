"""Retrieval metrics over ranked units, with bootstrap confidence intervals over questions.

A ranking is a list of unit ids (sections, or loose paragraphs) without repeats. Gold is the set of
gold sections of a question. Relevance is binary.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

KS = (1, 5, 10)
NDCG_K = 10
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 0


def recall_at(ranking: Sequence[str], gold: set[str], k: int) -> float:
    """Share of the gold units found in the first ``k`` places."""
    return len(gold.intersection(ranking[:k])) / len(gold)


def reciprocal_rank(ranking: Sequence[str], gold: set[str]) -> float:
    for place, unit in enumerate(ranking, start=1):
        if unit in gold:
            return 1.0 / place
    return 0.0


def ndcg_at(ranking: Sequence[str], gold: set[str], k: int = NDCG_K) -> float:
    gain = sum(1 / math.log2(place + 1) for place, u in enumerate(ranking[:k], 1) if u in gold)
    ideal = sum(1 / math.log2(place + 1) for place in range(1, min(len(gold), k) + 1))
    return gain / ideal


def question_scores(ranking: Sequence[str], gold: set[str]) -> dict[str, float]:
    scores = {f"recall@{k}": recall_at(ranking, gold, k) for k in KS}
    scores["mrr"] = reciprocal_rank(ranking, gold)
    scores[f"ndcg@{NDCG_K}"] = ndcg_at(ranking, gold)
    return scores


def dedupe(units: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(units))


def bootstrap(values: Sequence[float], samples: int = BOOTSTRAP_SAMPLES) -> tuple[float, float]:
    """A 95 percent percentile interval for the mean, from resampled questions."""
    data = np.asarray(values, dtype=float)
    if len(data) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = data[rng.integers(0, len(data), size=(samples, len(data)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return (float(low), float(high))


def paired_difference(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    """Mean of a minus b over the same questions, with a bootstrap interval for that mean.

    Resampling the per-question differences keeps the pairing: both systems are scored on the
    same resampled questions.
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    low, high = bootstrap(diff)
    return {"mean": float(diff.mean()) if len(diff) else math.nan, "low": low, "high": high}


def summarise(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean and interval of each metric over the given questions."""
    out = {}
    for metric in rows[0] if rows else []:
        values = [row[metric] for row in rows]
        low, high = bootstrap(values)
        out[metric] = {"mean": float(np.mean(values)), "low": low, "high": high}
    return out
