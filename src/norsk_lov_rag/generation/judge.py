"""Judge the answers with a stronger model, blind to the system that wrote them.

Every answer that is not a refusal goes to the judge: answerable questions for correctness, and
every answer with citations for support. A citation that names no known section is not sent; it
counts as unsupported. Items from all systems are mixed in a fixed random order and sent in
batches, with a short id in place of any system name. Items the judge leaves out of its reply
are sent once more in new batches.

A random sample is judged a second time in a new order and new batches, to measure how stable
the judge is. The free daily limit of the judge may stop a run. The next run then starts again
from the cached batches.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from pathlib import Path

from norsk_lov_rag.config import Settings
from norsk_lov_rag.generation.prompts import JUDGE_PROMPT_VERSION, judge_messages
from norsk_lov_rag.generation.run import read_runs
from norsk_lov_rag.generation.sources import SourceBook
from norsk_lov_rag.llm import ModelClient, QuotaExhausted
from norsk_lov_rag.retrieval.search import load_split

log = logging.getLogger(__name__)

LABELS = ("riktig", "delvis", "feil")
MAX_SOURCE_WORDS = 600
ITEM_START = re.compile(r'\{\s*"id"')


def item_id(run: str, question_id: str) -> str:
    return hashlib.sha256(f"{run}/{question_id}".encode()).hexdigest()[:10]


def judge_items(runs: dict[str, list[dict]], questions: dict[str, dict], books) -> list[dict]:
    items = []
    for run, rows in runs.items():
        for row in rows:
            if row["abstained"]:
                continue
            question = questions[row["item_id"]]
            answerable = question["question_type"] != "unanswerable"
            scope = row["system"].split("-", 1)[0]
            cites = [
                {
                    "citation": c["citation"],
                    "claims": c["claims"] or [row["answer"]],
                    "text": " ".join(books[scope].text(c["unit"]).split()[:MAX_SOURCE_WORDS]),
                }
                for c in row["citations"]
                if c["unit"] is not None
            ]
            if not answerable and not cites:
                continue
            items.append(
                {
                    "id": item_id(run, row["item_id"]),
                    "run": run,
                    "item_id": row["item_id"],
                    "question": question["question"],
                    "reference": question.get("reference_answer") if answerable else None,
                    "answer": row["answer"],
                    "citations": cites,
                }
            )
    return items


def _batches(items: list[dict], size: int, seed: int) -> list[list[dict]]:
    order = sorted(items, key=lambda item: item["id"])
    random.Random(seed).shuffle(order)
    return [order[i : i + size] for i in range(0, len(order), size)]


def _salvage(text: str) -> dict:
    """Read the complete item objects from a reply that was cut off or is not valid JSON."""
    decoder = json.JSONDecoder()
    entries = []
    starts = [m.start() for m in ITEM_START.finditer(text)]
    end = 0
    for start in starts:
        if start < end:
            continue
        try:
            entry, end = decoder.raw_decode(text, start)
        except ValueError:
            end = start + 1
        else:
            entries.append(entry)
    return {"items": entries}


def _parse(text: str, batch: list[dict]) -> dict[str, dict]:
    try:
        data = json.loads(text.strip().removeprefix("```json").removesuffix("```"))
    except ValueError:
        data = _salvage(text)
    found = {}
    wanted = {item["id"]: item for item in batch}
    for entry in data.get("items", []) if isinstance(data, dict) else []:
        if not isinstance(entry, dict) or entry.get("id") not in wanted:
            continue
        item = wanted[entry["id"]]
        label = entry.get("correctness")
        if item["reference"] is not None and label not in LABELS:
            continue
        support = {
            s.get("citation"): s.get("supported")
            for s in entry.get("support") or []
            if isinstance(s, dict) and isinstance(s.get("supported"), bool)
        }
        found[entry["id"]] = {
            "correctness": label if item["reference"] is not None else None,
            "support": {c["citation"]: support.get(c["citation"]) for c in item["citations"]},
            "reason": str(entry.get("reason", ""))[:500],
        }
    return found


def _judge_pass(client, cfg, items: list[dict], seed: int, label: str) -> tuple[dict, bool]:
    results: dict[str, dict] = {}
    batches = _batches(items, cfg.judge_batch_size, seed)
    for number, batch in enumerate(batches, start=1):
        payload = [
            {k: item[k] for k in ("id", "question", "reference", "answer", "citations")}
            for item in batch
        ]
        try:
            result = client.chat(
                cfg.judge,
                judge_messages(payload),
                JUDGE_PROMPT_VERSION,
                cfg.judge_max_tokens,
                json_mode=True,
            )
        except QuotaExhausted as exc:
            log.warning("%s: stopped at batch %d of %d: %s", label, number, len(batches), exc)
            return results, False
        found = _parse(result.text, batch)
        for item in batch:
            if item["id"] in found:
                results[item["id"]] = {**found[item["id"]], "model_version": result.model_version}
        log.info("%s: batch %d of %d, %d of %d items read", label, number, len(batches),
                 len(found), len(batch))  # fmt: skip
    return results, True


def judge(settings: Settings, split: str, frozen: Path, runs: Path, out: Path) -> dict:
    cfg = settings.generation
    questions = {q["item_id"]: q for q in load_split(frozen, split)}
    scopes = {path.name.split("-", 1)[0] for path in (runs / split).glob("*.jsonl")}
    books = {
        scope: SourceBook(settings.paths.data_dir, settings.paths.processed, scope)
        for scope in scopes
    }
    generated = read_runs(runs, split, questions, books)
    items = judge_items(generated, questions, books)
    client = ModelClient(settings.llm, settings.paths.llm)
    try:
        first, done = _judge_pass(client, cfg, items, cfg.seed, "judge")
        missing = [item for item in items if item["id"] not in first]
        if done and missing:
            # The judge sometimes leaves items out of its reply. Send those once more, in new
            # batches, so that a gap in one reply does not leave them without a verdict.
            filled, done = _judge_pass(client, cfg, missing, cfg.seed + 3, "fill")
            first.update(filled)
        second: dict[str, dict] = {}
        if done:
            sample = sorted(items, key=lambda item: item["id"])
            random.Random(cfg.seed + 1).shuffle(sample)
            sample = sample[: cfg.consistency_sample]
            second, done = _judge_pass(client, cfg, sample, cfg.seed + 2, "repeat")
    finally:
        client.close()
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        {"id": item["id"], "run": item["run"], "item_id": item["item_id"],
         "first": first.get(item["id"]), "second": second.get(item["id"])}
        for item in items
    ]  # fmt: skip
    with (out / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"items": len(items), "judged": len(first), "repeated": len(second), "complete": done}
