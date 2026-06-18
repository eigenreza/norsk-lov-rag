"""Build the test set: sample, write, filter, check, review and split.

Every model call is cached, so the whole pipeline runs again from the start each time and only
new calls cost quota. When a model's free quota is used up, its stage stops, and the items it
did not reach stay pending until the next run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from norsk_lov_rag.config import Settings
from norsk_lov_rag.corpus.language import marker_counts
from norsk_lov_rag.corpus.select import load_corpus
from norsk_lov_rag.llm import ModelClient
from norsk_lov_rag.testset import prompts as p
from norsk_lov_rag.testset.sample import Counts, build_tasks
from norsk_lov_rag.testset.search import BM25, SEARCH_VERSION
from norsk_lov_rag.testset.stages import (
    REVIEW_HIT_CHARS,
    REVIEW_HITS,
    SEARCH_HITS,
    check_block,
    group_of,
    run_checks,
    search_block,
    write_tasks,
)

log = logging.getLogger(__name__)

MAX_COPIED_WORDS = 5
DUPLICATE_SIMILARITY = 0.6
DEV_SHARE = 0.3
WORD = re.compile(r"[0-9a-zæøåéü]+")
# Words that point to text the reader of a question cannot see, such as "første ledd".
UNSEEN_CONTEXT = re.compile(
    r"§|\bparagraf|\b(?:første|andre|annet|tredje|fjerde|femte|sjette|siste) ledd\b"
    r"|\bdenne (?:lov|loven|forskrift|forskriften|paragraf|bestemmelse|bestemmelsen)\b"
    r"|\bdenne (?:lova|forskrifta|føresegna)\b|\b(?:lova|loven) i\b",
    re.IGNORECASE,
)
# Answers that name the texts the writer saw, such as "tekst A" or "B fastsetter".
TEXT_LABELS = re.compile(
    r"\btekst [AB]\b|\b[AB] (?:fastsetter|fastslår|tillater|sier|angir|gir|inkluderer|krever)\b",
    re.IGNORECASE,
)
TYPE_NAMES = {
    "direct": "direkte",
    "everyday": "hverdag",
    "cross_reference": "krysshenvisning",
    "unanswerable": "ubesvarbar",
}
NO_ANSWER = "Teksten svarer ikke på spørsmålet."
PUBLIC_FIELDS = (
    "item_id",
    "question_type",
    "written_standard",
    "question",
    "reference_answer",
    "gold_provision_ids",
    "gold_section_ids",
    "doc_key",
    "kind",
    "unanswerable_in",
)


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def longest_copied_run(question: str, texts: list[str]) -> int:
    """The longest run of words the question shares with any of the texts."""
    q = words(question)
    best = 0
    for text in texts:
        t = words(text)
        previous = [0] * (len(t) + 1)
        for qi in range(1, len(q) + 1):
            current = [0] * (len(t) + 1)
            for ti in range(1, len(t) + 1):
                if q[qi - 1] == t[ti - 1]:
                    current[ti] = previous[ti - 1] + 1
                    best = max(best, current[ti])
            previous = current
    return best


def shingles(text: str) -> set[str]:
    joined = " ".join(words(text))
    return {joined[i : i + 4] for i in range(max(1, len(joined) - 3))}


def similarity(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def live(items: list[dict]) -> list[dict]:
    return [item for item in items if "removed_by" not in item and "pending" not in item]


def apply_filters(items: list[dict]) -> None:
    """Written standard, unseen context, copied word runs and near-duplicates, without a model."""
    for item in live(items):
        unseen = UNSEEN_CONTEXT.search(item["question"])
        if unseen:
            item.update(removed_by="unseen_context", removed_reason=f"mentions {unseen.group()!r}")
            continue
        label = TEXT_LABELS.search(f"{item['question']} {item.get('reference_answer') or ''}")
        if label:
            item.update(removed_by="unseen_context", removed_reason=f"mentions {label.group()!r}")
            continue
        text = f"{item['question']} {item.get('reference_answer') or ''}"
        bokmal, nynorsk = marker_counts(text)
        wanted = item["written_standard"]
        if (wanted == "nn" and bokmal > nynorsk) or (wanted == "nb" and nynorsk > bokmal):
            item.update(
                removed_by="written_standard",
                removed_reason=f"markers bokmål {bokmal}, nynorsk {nynorsk}",
            )
            continue
        if item["question_type"] != "unanswerable":
            run = longest_copied_run(item["question"], item["gold_texts"])
            item["copied_words"] = run
            if run > MAX_COPIED_WORDS:
                item.update(
                    removed_by="overlap", removed_reason=f"{run} words in a row from the source"
                )
    kept: list[tuple[dict, set[str]]] = []
    for item in live(items):
        mine = shingles(item["question"])
        for other, theirs in kept:
            score = similarity(mine, theirs)
            if score >= DUPLICATE_SIMILARITY:
                item.update(
                    removed_by="duplicate",
                    duplicate_of=other["item_id"],
                    removed_reason=f"similarity {score:.2f}",
                )
                break
        else:
            kept.append((item, mine))


def _with_model(verdict: dict, result) -> dict:
    return {**verdict, "model": result.model, "model_version": result.model_version}


def check_outcome(item: dict, verdict: dict, result) -> None:
    item["check"] = _with_model(verdict, result)
    if verdict.get("kan_besvares") is not True:
        item.update(removed_by="check", removed_reason="not answerable from the gold text")
    elif verdict.get("svar_stemmer") is not True:
        item.update(removed_by="check", removed_reason="reference answer does not match")
    elif item["question_type"] == "cross_reference" and verdict.get("trenger_begge") is not True:
        item.update(removed_by="check", removed_reason="one of the texts is enough")


def search_outcome(item: dict, verdict: dict, result) -> None:
    item["check"] = _with_model(verdict, result)
    if verdict.get("besvart") is not False:
        item.update(removed_by="search_check", removed_reason="a corpus text answers it")


def review_outcome(item: dict, verdict: dict, result) -> None:
    label = str(verdict.get("vurdering", "")).strip().lower()
    item["review"] = _with_model(verdict, result)
    if label != "korrekt":
        item.update(
            removed_by="review",
            removed_reason=f"{label or 'no label'}: {verdict.get('begrunnelse', '')}",
        )


def review_block(item: dict) -> str:
    lines = [
        f"Element {item['item_id']}",
        f"Type: {TYPE_NAMES[item['question_type']]}",
        f"Målform: {p.STANDARD_NAMES[item['written_standard']]}",
        f"Spørsmål: {item['question']}",
    ]
    if item["question_type"] == "unanswerable":
        lines.append(f"Referansesvar: {NO_ANSWER}")
        lines.append("Tekster fra søket:")
        lines += [
            f"- {hit['heading']}: {hit['text'][:REVIEW_HIT_CHARS]}"
            for hit in item["search_hits"][:REVIEW_HITS]
        ]
    else:
        lines.append(f"Referansesvar: {item['reference_answer']}")
        cross = item["question_type"] == "cross_reference"
        for number, (title, text) in enumerate(
            zip(item["gold_headings"], item["gold_texts"], strict=True)
        ):
            label = ("Tekst A" if number == 0 else "Tekst B") if cross else "Kildetekst"
            lines.append(f"{label} ({title}): {text}")
    return "\n".join(lines)


class Searcher:
    """BM25 over the paragraphs of a corpus, with hits remembered on disk between runs."""

    def __init__(self, settings: Settings, cache_path: Path) -> None:
        self.settings = settings
        self.cache_path = cache_path
        self.cache: dict[str, list[dict]] = {}
        if cache_path.exists():
            self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
        self.indexes: dict[str, tuple[BM25, dict[str, dict]]] = {}

    def _index(self, scope: str) -> tuple[BM25, dict[str, dict]]:
        if scope not in self.indexes:
            log.info("building the search index for the %s corpus", scope)
            tables = load_corpus(self.settings.paths.processed, self.settings.corpus, scope)
            documents = tables["documents"].set_index("doc_key")
            provisions = tables["provisions"]
            labels = tables["sections"].set_index("section_id")["section_label"]
            info = {}
            for row in provisions.itertuples(index=False):
                document = documents.loc[row.doc_key]
                label = labels.get(row.section_id, "") if isinstance(row.section_id, str) else ""
                info[row.provision_id] = {
                    "provision_id": row.provision_id,
                    "heading": f"{document['short_title'] or document['title']} {label}".strip(),
                    "text": row.text,
                }
            index = BM25(list(provisions["provision_id"]), list(provisions["text"]))
            self.indexes[scope] = (index, info)
        return self.indexes[scope]

    def _search(self, question: str, scope: str, k: int) -> list[dict]:
        index, info = self._index(scope)
        return [{**info[pid], "score": round(score, 3)} for pid, score in index.search(question, k)]

    def hits(self, question: str, scope: str) -> list[dict]:
        """Top paragraphs for a question. For the full corpus, half come from the tax corpus
        first, since that is where an answer to a tax question is most likely to be."""
        key = f"{SEARCH_VERSION}|tax-first|{SEARCH_HITS}|{scope}|{question}"
        if key not in self.cache:
            if scope == "full":
                found = self._search(question, "tax", SEARCH_HITS // 2)
                seen = {hit["provision_id"] for hit in found}
                for hit in self._search(question, "full", SEARCH_HITS):
                    if len(found) < SEARCH_HITS and hit["provision_id"] not in seen:
                        found.append(hit)
            else:
                found = self._search(question, scope, SEARCH_HITS)
            self.cache[key] = found
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False), "utf-8")
        return self.cache[key]


def split(items: list[dict]) -> None:
    """Put a fixed share of each type and written standard in the development set."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for item in live(items):
        groups.setdefault((item["question_type"], item["written_standard"]), []).append(item)
    for members in groups.values():
        members.sort(key=lambda item: hashlib.sha256(item["item_id"].encode()).hexdigest())
        cut = round(len(members) * DEV_SHARE)
        for position, item in enumerate(members):
            item["split"] = "dev" if position < cut else "test"


def report(tasks: list[dict], items: list[dict], client: ModelClient, stopped: dict) -> dict:
    def count(rows, key=lambda item: item["question_type"]) -> dict:
        return dict(sorted(Counter(key(row) for row in rows).items()))

    removed = [item for item in items if "removed_by" in item]
    final = live(items)
    versions: dict[str, set[str]] = {}
    for item in items:
        for record in (item, item.get("check") or {}, item.get("review") or {}):
            if record.get("model"):
                versions.setdefault(record["model"], set()).add(record.get("model_version", ""))
    return {
        "tasks": count(tasks, lambda t: f"{t['question_type']}-{t['written_standard']}"),
        "items": count(items, lambda i: f"{i['question_type']}-{i['written_standard']}"),
        "pending": count([i for i in items if "pending" in i], lambda i: i["pending"]),
        "removed": count(removed, lambda i: f"{i['removed_by']}: {i['question_type']}"),
        "final": count(final, lambda i: f"{i['question_type']}-{i['written_standard']}"),
        "final_split": count(final, lambda i: i.get("split", "")),
        "stopped_by_quota": stopped,
        "model_versions": {model: sorted(v) for model, v in versions.items()},
        "usage_last_24h": client.usage(),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_testset(
    settings: Settings, pilot: int | None = None, offline: bool = False, final: bool = False
) -> dict:
    """Run all stages. With ``final``, questions not written yet are left out for good, so the
    last batch of each type no longer has to wait for them."""
    out = settings.paths.data_dir / "testset" / ("pilot" if pilot else "")
    tables = load_corpus(settings.paths.processed, settings.corpus, "tax")
    references = pd.read_parquet(settings.paths.processed / "references.parquet")
    tasks = build_tasks(tables, references, Counts())
    if pilot:
        tasks = [t for t in tasks if int(t["task_id"].rsplit("-", 1)[1]) <= pilot]
    write_jsonl(out / "tasks.jsonl", tasks)

    client = ModelClient(settings.llm, settings.paths.llm, offline=offline)
    stopped = {}
    try:
        items, stopped["writer"] = write_tasks(client, tasks)
        apply_filters(items)
        still_written = {group_of(i) for i in items if i.get("pending") == "write"}
        open_groups = frozenset() if final else frozenset(still_written)

        answerable = [i for i in live(items) if i["question_type"] != "unanswerable"]
        stopped["checker"] = run_checks(
            client,
            answerable,
            "checker",
            p.CHECKER_SYSTEM,
            p.CHECKER_VERSION,
            check_block,
            check_outcome,
            open_groups,
        )
        searcher = Searcher(settings, settings.paths.data_dir / "testset" / "search_hits.json")
        unanswerable = [i for i in live(items) if i["question_type"] == "unanswerable"]
        for item in unanswerable:
            scope = "full" if "full" in item["unanswerable_in"] else "tax"
            item["search_hits"] = searcher.hits(item["question"], scope)
        search_stopped = run_checks(
            client,
            unanswerable,
            "checker",
            p.SEARCH_CHECK_SYSTEM,
            p.SEARCH_CHECK_VERSION,
            search_block,
            search_outcome,
            open_groups,
        )
        stopped["checker"] = stopped["checker"] or search_stopped

        stopped["reviewer"] = run_checks(
            client,
            live(items),
            "reviewer",
            p.REVIEWER_SYSTEM,
            p.REVIEWER_VERSION,
            review_block,
            review_outcome,
            open_groups,
        )
        split(items)
        summary = report(tasks, items, client, stopped)
    finally:
        client.close()

    write_jsonl(out / "items.jsonl", items)
    for name in ("dev", "test"):
        rows = [
            {key: item.get(key) for key in PUBLIC_FIELDS}
            for item in live(items)
            if item.get("split") == name
        ]
        write_jsonl(out / f"{name}.jsonl", rows)
    (out / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary["output"] = str(out)
    return summary


FROZEN_FILES = ("dev.jsonl", "test.jsonl")


def freeze_testset(source: Path, target: Path, corpus_hash: str, models: dict[str, str]) -> dict:
    """Copy the development and test sets to a tracked folder, with a manifest of their hashes.

    A frozen set is never replaced: the call fails if the target already holds one.
    """
    if (target / "manifest.json").exists():
        raise FileExistsError(f"{target} already holds a frozen test set")
    summary = json.loads((source / "report.json").read_text(encoding="utf-8"))
    target.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in FROZEN_FILES:
        data = (source / name).read_bytes()
        (target / name).write_bytes(data)
        files[name] = {"rows": data.count(b"\n"), "sha256": hashlib.sha256(data).hexdigest()}
    manifest = {
        "corpus_hash": corpus_hash,
        "counts": summary["final"],
        "split": summary["final_split"],
        "dev_share": DEV_SHARE,
        "models": models,
        "model_versions": summary["model_versions"],
        "prompt_versions": {
            "writer": p.WRITER_VERSION,
            "cross_reference_writer": p.CROSS_WRITER_VERSION,
            "unanswerable_writer": p.UNANSWERABLE_VERSION,
            "checker": p.CHECKER_VERSION,
            "search_check": p.SEARCH_CHECK_VERSION,
            "reviewer": p.REVIEWER_VERSION,
        },
        "search_version": SEARCH_VERSION,
        "files": files,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest
