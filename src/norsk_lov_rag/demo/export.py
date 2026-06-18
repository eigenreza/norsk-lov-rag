"""Write the small set of files the demo needs, so that it runs without the data folder.

The demo searches the tax corpus with BM25 and shows the full text of each section. It needs
the chunks, the labels and texts of the sections, a few recorded answers, and the result
summaries. Everything goes to ``demo_data/`` as plain JSON, small enough for a free web plan.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from norsk_lov_rag.chunking.build import chunk_path
from norsk_lov_rag.config import Settings
from norsk_lov_rag.generation.prompts import ANSWER_PROMPT_VERSION
from norsk_lov_rag.generation.report import load_verdicts
from norsk_lov_rag.generation.run import read_runs
from norsk_lov_rag.generation.sources import SourceBook
from norsk_lov_rag.retrieval.search import load_split

SCOPE = "tax"
STRATEGY = "structure"
DEMO_SYSTEM = "bm25"
EXAMPLE_RUN = "tax-structure__hybrid+rerank__gemma"
ATTRIBUTION = (
    "Inneholder data under Norsk lisens for offentlige data (NLOD) tilgjengeliggjort av "
    "Stiftelsen Lovdata."
)
# How many recorded answers of each kind the examples page shows.
EXAMPLE_PLAN = (
    ("direct", "nb", 2),
    ("everyday", "nb", 2),
    ("cross_reference", "nb", 2),
    ("everyday", "nn", 1),
    ("direct", "nn", 1),
    ("unanswerable", "nb", 1),
    ("unanswerable", "nn", 1),
)


def _order(item_id: str) -> str:
    return hashlib.sha256(item_id.encode()).hexdigest()


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def book_data(book: SourceBook) -> dict:
    """The parts of a source book the demo needs to show and resolve citations."""
    return {
        "labels": book.labels,
        "titles": book.titles,
        "texts": book.texts,
        "lookup": [[name, number, unit] for (name, number), unit in book._lookup.items()],
    }


def _good_answer(row: dict, verdict: dict | None) -> bool:
    if row["abstained"] or not row["valid_output"] or row["finish_reason"] == "length":
        return False
    if not row["citations"] or any(not c["in_sources"] for c in row["citations"]):
        return False
    if row["sentences_with_citation"] != row["sentences"] or row["language"] == "mismatch":
        return False
    if verdict is not None and verdict.get("correctness") not in (None, "riktig"):
        return False
    return bool(row["cited_gold"])


def pick_examples(rows: list[dict], questions: dict, verdicts: dict) -> list[dict]:
    """A fixed, varied set: good answers of each answerable type and correct refusals."""
    picked: list[dict] = []
    ordered = sorted(rows, key=lambda row: _order(row["item_id"]))
    for kind, standard, count in EXAMPLE_PLAN:
        found = 0
        for row in ordered:
            question = questions[row["item_id"]]
            if question["question_type"] != kind or question["written_standard"] != standard:
                continue
            if row in picked:
                continue
            verdict = verdicts.get(row["item_id"])
            fits = row["abstained"] if kind == "unanswerable" else _good_answer(row, verdict)
            if fits:
                picked.append(row)
                found += 1
            if found == count:
                break
    return picked


def example_entry(row: dict, question: dict, book: SourceBook, verdict: dict | None) -> dict:
    return {
        "item_id": row["item_id"],
        "question": question["question"],
        "question_type": question["question_type"],
        "written_standard": question["written_standard"],
        "reference_answer": question.get("reference_answer"),
        "answer": row["answer"],
        "abstained": row["abstained"],
        "model": row["model"],
        "model_version": row["model_version"],
        "system": row["system"],
        "citations": [
            {"citation": c["citation"], "unit": c["unit"], "in_sources": c["in_sources"]}
            for c in row["citations"]
        ],
        "sources": [{"unit": u, "label": book.labels.get(u, u)} for u in row["sources"]],
        "correctness": (verdict or {}).get("correctness"),
    }


def export_demo(
    settings: Settings,
    out: Path,
    split: str = "test",
    frozen: Path = Path("testsett"),
    runs: Path = Path("runs/generation"),
    judged: Path = Path("runs/judge"),
    results: Path = Path("results"),
    figures: Path = Path("figures"),
    example_run: str = EXAMPLE_RUN,
) -> dict:
    """Write demo_data and return a short summary of what was written."""
    out.mkdir(parents=True, exist_ok=True)
    data_dir, processed = settings.paths.data_dir, settings.paths.processed
    book = SourceBook(data_dir, processed, SCOPE)
    chunks = pd.read_parquet(chunk_path(data_dir, SCOPE, STRATEGY))
    _write_json(
        out / "chunks.json",
        {"texts": chunks["text"].tolist(), "units": [list(u) for u in chunks["units"]]},
    )
    _write_json(out / "sections.json", book_data(book))

    questions = {q["item_id"]: q for q in load_split(frozen, split)}
    examples: list[dict] = []
    run_path = runs / split / f"{example_run}.jsonl"
    if run_path.exists():
        rows = read_runs(runs, split, questions, {SCOPE: book}).get(example_run, [])
        verdicts = load_verdicts(judged / f"{split}.jsonl")[0].get(example_run, {})
        examples = [
            example_entry(row, questions[row["item_id"]], book, verdicts.get(row["item_id"]))
            for row in pick_examples(rows, questions, verdicts)
        ]
    _write_json(out / "examples.json", {"run": example_run, "split": split, "items": examples})

    copied = []
    for name in ("retrieval", "generation"):
        source = results / name / f"{split}.json"
        if source.exists():
            (out / "results").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, out / "results" / f"{name}.json")
            copied.append(name)
    figure_files = []
    if figures.exists():
        for path in sorted(figures.glob("*.png")):
            target = out / "figures" / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            figure_files.append(path.name)

    manifest_path = processed / "manifest.json"
    corpus_hash = None
    if manifest_path.exists():
        corpus_hash = json.loads(manifest_path.read_text(encoding="utf-8")).get("corpus_hash")
    gen = settings.generation
    answer_model = settings.llm.models.get("gemma")
    manifest = {
        "source": "Lovdata, open datasets under NLOD 2.0",
        "attribution": ATTRIBUTION,
        "corpus_hash": corpus_hash,
        "scope": SCOPE,
        "strategy": STRATEGY,
        "system": DEMO_SYSTEM,
        "chunks": len(chunks),
        "sections": len(book.labels),
        "answer_model": answer_model.model if answer_model else None,
        "prompt_version": ANSWER_PROMPT_VERSION,
        "top_k": gen.top_k,
        "max_words_per_source": gen.max_words_per_source,
        "max_context_words": gen.max_context_words,
        "max_tokens": gen.max_tokens,
        "results": copied,
        "figures": figure_files,
    }
    _write_json(out / "manifest.json", manifest)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    return {"examples": len(examples), "results": copied, "figures": len(figure_files),
            "bytes": size}  # fmt: skip
