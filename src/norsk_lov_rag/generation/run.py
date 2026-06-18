"""Write answers for every question with each retrieval system and generator.

Each answer is saved with its sources, tokens, time and the checks that need no model. Calls go
through the cached model client, so a stopped run goes on where it stopped, and a finished run
can be read again offline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from norsk_lov_rag.config import Settings
from norsk_lov_rag.generation.answers import analyse, parse_output
from norsk_lov_rag.generation.prompts import ANSWER_PROMPT_VERSION, answer_messages
from norsk_lov_rag.generation.sources import SourceBook
from norsk_lov_rag.llm import ModelClient, ProviderError, QuotaExhausted
from norsk_lov_rag.retrieval.search import Searcher, load_split

log = logging.getLogger(__name__)


def run_name(system: str, generator: str) -> str:
    """File name for one retrieval system and generator, such as tax-structure__bm25__gemma."""
    setting, name = system.split("/", 1)
    return f"{setting}__{name.replace(':', '-')}__{generator}"


def generate(
    settings: Settings,
    split: str,
    frozen: Path,
    runs: Path,
    systems: list[str] | None = None,
    generators: list[str] | None = None,
    item_ids: set[str] | None = None,
) -> dict[str, int]:
    cfg = settings.generation
    questions = load_split(frozen, split)
    if item_ids is not None:
        questions = [q for q in questions if q["item_id"] in item_ids]
    client = ModelClient(settings.llm, settings.paths.llm)
    out_dir = runs / split
    out_dir.mkdir(parents=True, exist_ok=True)
    books: dict[str, SourceBook] = {}
    written = {}
    try:
        for system in systems or cfg.retrieval_systems:
            setting, name = system.split("/", 1)
            scope = setting.split("-", 1)[0]
            ranking = Searcher(settings, setting, split).rank(
                name, [q["question"] for q in questions]
            )
            if scope not in books:
                books[scope] = SourceBook(settings.paths.data_dir, settings.paths.processed, scope)
            book = books[scope]
            for role in generators or cfg.generators:
                path = out_dir / f"{run_name(system, role)}.jsonl"
                rows = []
                try:
                    for number, (question, units) in enumerate(
                        zip(questions, ranking, strict=True)
                    ):
                        try:
                            rows.append(_answer(client, cfg, book, system, role, question, units))
                        except ProviderError as exc:
                            # Leave the question out; a new run fills it in from the cache.
                            log.warning("%s: %s skipped: %s", path.stem, question["item_id"], exc)
                        if (number + 1) % 20 == 0:
                            log.info("%s: %d of %d answers", path.stem, number + 1, len(questions))
                finally:
                    _write(path, rows)
                    written[path.stem] = len(rows)
    except QuotaExhausted as exc:
        log.warning("stopped: %s", exc)
    finally:
        client.close()
    return written


def _answer(client, cfg, book, system, role, question, units) -> dict:
    sources = book.sources(units, cfg.top_k, cfg.max_words_per_source, cfg.max_context_words)
    messages = answer_messages(question["question"], sources)
    result = client.chat(role, messages, ANSWER_PROMPT_VERSION, cfg.max_tokens)
    parsed = parse_output(result.text)
    shown = [s.unit for s in sources]
    return {
        "item_id": question["item_id"],
        "question_type": question["question_type"],
        "written_standard": question["written_standard"],
        "system": system,
        "generator": role,
        "model": result.model,
        "model_version": result.model_version,
        "prompt_version": ANSWER_PROMPT_VERSION,
        "sources": shown,
        "context_words": sum(len(s.text.split()) for s in sources),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "latency_seconds": result.latency_seconds,
        "finish_reason": result.finish_reason,
        "cache_key": result.key,
        "raw": result.text,
        **parsed,
        **analyse(parsed, question, shown, book),
    }


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_runs(
    runs: Path, split: str, questions: dict[str, dict] | None = None, books: dict | None = None
) -> dict[str, list[dict]]:
    """Saved answers by run. With questions and source books, the checks that need no model
    are done again from the saved text, so every answer is read by the same rules."""
    out = {}
    for path in sorted((runs / split).glob("*.jsonl")):
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        if questions is not None and books is not None:
            for row in rows:
                book = books[row["system"].split("-", 1)[0]]
                parsed = parse_output(row["raw"])
                row.update(parsed)
                row.update(analyse(parsed, questions[row["item_id"]], row["sources"], book))
        out[path.stem] = rows
    return out
