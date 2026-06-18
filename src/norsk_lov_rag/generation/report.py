"""Measures for every retrieval system and generator, with bootstrap intervals over questions.

Correctness counts a refusal on an answerable question as wrong. Citation recall is taken over
all answerable questions, so a refusal gives zero recall. Citation precision is taken over the
answers that cite something. A citation that names no known section is wrong for precision and
unsupported for support.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from norsk_lov_rag.evaluation.retrieval_metrics import bootstrap, paired_difference
from norsk_lov_rag.generation.run import read_runs
from norsk_lov_rag.generation.sources import SourceBook, find_mentions
from norsk_lov_rag.retrieval.search import load_split

SCORE = {"riktig": 1.0, "delvis": 0.5, "feil": 0.0}


def _mean(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "low": None, "high": None}
    low, high = bootstrap(values)
    return {"n": len(values), "mean": float(np.mean(values)), "low": low, "high": high}


def _share(flags: list[bool]) -> dict:
    return _mean([1.0 if f else 0.0 for f in flags])


def row_scores(
    row: dict, question: dict, verdict: dict | None, mentioned: list[str | None] | None = None
) -> dict:
    """Per-question values; None where a measure does not apply to this question.

    ``mentioned`` holds the sections named anywhere in the answer, with or without brackets,
    for the lenient citation count.
    """
    answerable = question["question_type"] != "unanswerable"
    out: dict = {"answerable": answerable, "abstained": row["abstained"]}
    label = None
    if answerable:
        if row["abstained"]:
            label = "avstått"
        elif verdict and verdict.get("correctness") in SCORE:
            label = verdict["correctness"]
    out["label"] = label
    out["score"] = 0.0 if label == "avstått" else SCORE.get(label) if label else None
    out["strict"] = None if out["score"] is None else float(label == "riktig")
    gold = set(question.get("gold_section_ids") or [])
    cites = row["citations"]
    if answerable:
        out["recall"] = len(row["cited_gold"]) / len(gold) if not row["abstained"] else 0.0
        out["precision"] = (
            len(row["cited_gold"]) / len(cites) if cites and not row["abstained"] else None
        )
    else:
        out["recall"] = out["precision"] = None
    out["lenient_recall"] = out["lenient_precision"] = None
    if answerable and mentioned is not None:
        found = {unit for unit in mentioned if unit is not None}
        out["lenient_recall"] = len(gold & found) / len(gold) if not row["abstained"] else 0.0
        if mentioned and not row["abstained"]:
            out["lenient_precision"] = len(gold & found) / len(set(mentioned))
    support = []
    for c in cites if not row["abstained"] else []:
        if c["unit"] is None:
            support.append(False)
        elif verdict and verdict.get("support", {}).get(c["citation"]) is not None:
            support.append(bool(verdict["support"][c["citation"]]))
    out["support"] = sum(support) / len(support) if support else None
    out["outside_sources"] = (
        sum(not c["in_sources"] for c in cites) / len(cites) if cites and not row["abstained"]
        else None
    )  # fmt: skip
    out["unresolved"] = (
        sum(c["unit"] is None for c in cites) / len(cites) if cites and not row["abstained"]
        else None
    )  # fmt: skip
    if row["abstained"] or not row["sentences"]:
        out["all_cited"] = None
    else:
        out["all_cited"] = float(row["sentences_with_citation"] == row["sentences"])
    return out


def summarise_run(
    rows: list[dict], questions: dict[str, dict], verdicts: dict, book: SourceBook | None = None
) -> dict:
    scores = [
        row_scores(r, questions[r["item_id"]], verdicts.get(r["item_id"]), _mentioned(r, book))
        for r in rows
    ]

    def pick(field: str, rows_scores=scores) -> list[float]:
        return [s[field] for s in rows_scores if s[field] is not None]

    answerable = [s for s in scores if s["answerable"]]
    unanswerable = [s for s in scores if not s["answerable"]]
    latencies = [r["latency_seconds"] for r in rows if r.get("latency_seconds") is not None]
    out = {
        "questions": len(rows),
        "correctness": _mean(pick("score")),
        "correct_strict": _mean(pick("strict")),
        "labels": dict(Counter(s["label"] for s in answerable)),
        "unjudged": sum(1 for s in answerable if s["score"] is None),
        "citation_precision": _mean(pick("precision")),
        "citation_recall": _mean(pick("recall")),
        "lenient_citation_precision": _mean(pick("lenient_precision")),
        "lenient_citation_recall": _mean(pick("lenient_recall")),
        "support": _mean(pick("support")),
        "outside_sources": _mean(pick("outside_sources")),
        "unresolved": _mean(pick("unresolved")),
        "all_sentences_cited": _mean(pick("all_cited")),
        "correct_refusals": _share([s["abstained"] for s in unanswerable]),
        "false_refusals": _share([s["abstained"] for s in answerable]),
        "language": dict(Counter(r["language"] for r in rows)),
        "language_match": _share([r["language"] != "mismatch" for r in rows]),
        "valid_output": _share([r["valid_output"] for r in rows]),
        "truncated": sum(1 for r in rows if r["finish_reason"] == "length"),
        "prompt_tokens": float(np.mean([r["prompt_tokens"] for r in rows])) if rows else None,
        "completion_tokens": float(np.mean([r["completion_tokens"] for r in rows]))
        if rows
        else None,
        "latency_median": statistics.median(latencies) if latencies else None,
        "latency_p95": float(np.percentile(latencies, 95)) if latencies else None,
    }
    for field, name in (("question_type", "by_type"), ("written_standard", "by_standard")):
        groups: dict[str, list] = defaultdict(list)
        for row, s in zip(rows, scores, strict=True):
            groups[row[field]].append(s)
        out[name] = {
            key: {
                "n": len(group),
                "correctness": _mean(pick("score", group)),
                "refusals": _share([s["abstained"] for s in group]),
            }
            for key, group in sorted(groups.items())
        }
    return out


def _mentioned(row: dict, book: SourceBook | None) -> list[str | None] | None:
    if book is None:
        return None
    return [book.resolve(m) for m in dict.fromkeys(find_mentions(row["answer"]))]


def load_verdicts(path: Path) -> tuple[dict[str, dict[str, dict]], list[dict]]:
    """First verdicts by run and question, and the rows that were judged twice."""
    by_run: dict[str, dict[str, dict]] = defaultdict(dict)
    repeated = []
    if not path.exists():
        return by_run, repeated
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        if row["first"]:
            by_run[row["run"]][row["item_id"]] = row["first"]
        if row["first"] and row["second"]:
            repeated.append(row)
    return by_run, repeated


def consistency(repeated: list[dict]) -> dict:
    """Agreement between the two verdicts on the items judged twice."""
    pairs = [
        (r["first"]["correctness"], r["second"]["correctness"])
        for r in repeated
        if r["first"]["correctness"] and r["second"]["correctness"]
    ]
    support = [
        (a, r["second"]["support"].get(c))
        for r in repeated
        for c, a in r["first"]["support"].items()
        if a is not None and r["second"]["support"].get(c) is not None
    ]
    agree = sum(a == b for a, b in pairs)
    kappa = None
    if pairs:
        observed = agree / len(pairs)
        first, second = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
        expected = sum(first[k] * second[k] for k in first) / len(pairs) ** 2
        kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    return {
        "items": len(repeated),
        "correctness_pairs": len(pairs),
        "correctness_agreement": agree / len(pairs) if pairs else None,
        "correctness_kappa": kappa,
        "support_pairs": len(support),
        "support_agreement": sum(a == b for a, b in support) / len(support) if support else None,
    }


def compare(pairs: list[list[str]], runs: dict, questions: dict, verdicts: dict) -> list[dict]:
    """Paired differences in correctness on answerable questions judged in both runs."""
    out = []
    for a, b in pairs:
        if a not in runs or b not in runs:
            continue
        sa = {r["item_id"]: row_scores(r, questions[r["item_id"]], verdicts[a].get(r["item_id"]))
              for r in runs[a]}  # fmt: skip
        sb = {r["item_id"]: row_scores(r, questions[r["item_id"]], verdicts[b].get(r["item_id"]))
              for r in runs[b]}  # fmt: skip
        both = [
            i for i in sa if i in sb and sa[i]["score"] is not None and sb[i]["score"] is not None
        ]
        out.append({
            "a": a, "b": b, "n": len(both),
            "correctness": paired_difference([sa[i]["score"] for i in both],
                                             [sb[i]["score"] for i in both]),
        })  # fmt: skip
    return out


def examples(runs: dict, questions: dict, verdicts: dict, per_kind: int = 3) -> dict:
    """A few failures of each kind, picked in a fixed order."""
    kinds: dict[str, list] = defaultdict(list)
    for run, rows in sorted(runs.items()):
        for row in rows:
            q = questions[row["item_id"]]
            v = verdicts[run].get(row["item_id"]) or {}
            answerable = q["question_type"] != "unanswerable"
            entry = {"run": run, "item_id": row["item_id"], "question": q["question"],
                     "answer": row["answer"], "reason": v.get("reason")}  # fmt: skip
            if answerable and not row["abstained"] and v.get("correctness") == "feil":
                kinds["wrong_answer"].append(entry)
            if answerable and row["abstained"]:
                kinds["false_refusal"].append(entry)
            if not answerable and not row["abstained"]:
                kinds["missed_refusal"].append(entry)
            if any(s is False for s in (v.get("support") or {}).values()):
                kinds["unsupported_citation"].append(entry)
            if any(not c["in_sources"] for c in row["citations"]) and not row["abstained"]:
                kinds["citation_outside_sources"].append(entry)
            if row["language"] == "mismatch":
                kinds["wrong_written_standard"].append(entry)
            if not row["valid_output"]:
                kinds["invalid_output"].append(entry)
    return {kind: {"count": len(items), "examples": items[:per_kind]}
            for kind, items in sorted(kinds.items())}  # fmt: skip


def report(settings, split: str, frozen: Path, runs: Path, judged: Path) -> dict:
    questions = {q["item_id"]: q for q in load_split(frozen, split)}
    scopes = {path.name.split("-", 1)[0] for path in (runs / split).glob("*.jsonl")}
    books = {
        scope: SourceBook(settings.paths.data_dir, settings.paths.processed, scope)
        for scope in scopes
    }
    generated = read_runs(runs, split, questions, books)
    verdicts, repeated = load_verdicts(judged / f"{split}.jsonl")
    summaries = {
        name: summarise_run(rows, questions, verdicts[name], books[name.split("-", 1)[0]])
        for name, rows in generated.items()
    }
    return {
        "split": split,
        "runs": summaries,
        "judge_consistency": consistency(repeated),
        "comparisons": compare(settings.generation.comparisons, generated, questions, verdicts),
        "examples": examples(generated, questions, verdicts),
    }
