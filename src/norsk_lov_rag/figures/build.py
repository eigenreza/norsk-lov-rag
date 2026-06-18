"""Build every figure from the saved result files, in Norwegian and in English.

Norwegian figures go to ``<out>/`` and English ones to ``<out>/en/``. A figure whose result
file is missing is skipped with a message, so the command can run before every stage is done.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from norsk_lov_rag.figures import answers, corpus, retrieval
from norsk_lov_rag.figures.style import LANGS

log = logging.getLogger(__name__)


def _read(path: Path) -> dict | None:
    if not path.exists():
        log.info("figures: %s is missing, its figures are skipped", path)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def refresh_corpus_results(settings, results: Path) -> None:
    """Copy the corpus counts and the reference summary to ``results/corpus/``.

    Runs only when the processed tables exist. Without them, the saved files are used as
    they are.
    """
    processed = settings.paths.processed
    if not (processed / "provisions.parquet").exists():
        log.info("figures: no processed tables, using the saved corpus results")
        return
    _write(results / "corpus" / "overview.json", corpus.corpus_summary(settings))
    summary = processed / "references_summary.json"
    if summary.exists():
        _write(results / "corpus" / "references.json",
               json.loads(summary.read_text(encoding="utf-8")))  # fmt: skip


def build_figures(results: Path, out: Path, langs: list[str], split: str = "test") -> list[Path]:
    """Every figure whose input exists, for each language. Returns the files written."""
    inputs = [
        (results / "corpus" / "overview.json", [corpus.corpus_overview]),
        (results / "corpus" / "references.json", [corpus.reference_graph]),
        (results / "retrieval" / f"{split}.json", retrieval.FIGURES),
        (results / "generation" / f"{split}.json", answers.FIGURES),
    ]
    written: list[Path] = []
    for path, figures in inputs:
        data = _read(path)
        if data is None:
            continue
        for lang in langs:
            if lang not in LANGS:
                raise ValueError(f"unknown language {lang!r}, expected one of {LANGS}")
            target = out if lang == "nb" else out / lang
            for make in figures:
                written.extend(make(data, target, lang))
    return written
