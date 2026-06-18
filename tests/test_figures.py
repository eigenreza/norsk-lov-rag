"""Figures are written in both languages from small fake result files, without dates."""

from __future__ import annotations

import json

from norsk_lov_rag.figures.build import build_figures
from norsk_lov_rag.figures.style import run_label


def _ci(mean: float) -> dict:
    return {"mean": mean, "low": max(0.0, mean - 0.05), "high": min(1.0, mean + 0.05)}


def _scores(base: float) -> dict:
    return {m: _ci(base) for m in ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10")}


def _system(base: float) -> dict:
    return {
        "n": 10,
        "all": _scores(base),
        "by_type": {k: {"n": 3, **_scores(base)} for k in ("direct", "everyday",
                                                          "cross_reference")},
        "by_standard": {k: {"n": 5, **_scores(base)} for k in ("nb", "nn")},
    }  # fmt: skip


def _retrieval() -> dict:
    tax = ["bm25", "bm25-compound", "dense:e5-base", "dense:nb-sbert-v2-large", "hybrid",
           "hybrid+rerank", "hybrid+expansion", "hybrid+rerank+expansion"]  # fmt: skip
    return {
        "split": "test",
        "questions": 10,
        "settings": {
            "tax-structure": {"systems": {s: _system(0.8) for s in tax}},
            "tax-naive": {"systems": {s: _system(0.6) for s in tax[:5]}},
            "full-structure": {
                "systems": {s: _system(0.7) for s in ["bm25", "hybrid", "hybrid+rerank"]}
            },
            "full-naive": {"systems": {"bm25": _system(0.5)}},
        },  # fmt: skip
        "comparisons": [],
    }


def _run(base: float) -> dict:
    return {
        "questions": 10,
        "correctness": _ci(base),
        "citation_precision": _ci(base),
        "citation_recall": _ci(base - 0.1),
        "support": _ci(0.9),
        "correct_refusals": _ci(0.7),
        "false_refusals": _ci(0.1),
        "language_match": _ci(0.95),
        "prompt_tokens": 1500.0,
        "completion_tokens": 200.0,
        "latency_median": 12.0,
        "latency_p95": 30.0,
        "by_type": {k: {"n": 3, "correctness": _ci(base), "refusals": _ci(0.1)}
                    for k in ("direct", "everyday", "cross_reference")},
        "by_standard": {k: {"n": 5, "correctness": _ci(base), "refusals": _ci(0.1)}
                        for k in ("nb", "nn")},
    }  # fmt: skip


def _generation() -> dict:
    runs = {
        "tax-structure__bm25__gemma": _run(0.7),
        "tax-structure__hybrid+rerank__gemma": _run(0.8),
        "tax-structure__bm25__normistral": _run(0.4),
        "tax-structure__hybrid+rerank__normistral": _run(0.5),
    }
    return {"split": "test", "runs": runs, "judge_consistency": {}, "comparisons": [],
            "examples": {}}  # fmt: skip


def _corpus() -> tuple[dict, dict]:
    overview = {
        "corpus_hash": "abc",
        "tax_documents": [
            {"doc_key": "lov/a", "name": "Skatteloven", "sections": 344, "provisions": 1100,
             "written_standard": "nb"},
            {"doc_key": "forskrift/b", "name": "FSFIN", "sections": 516, "provisions": 1100,
             "written_standard": "nb"},
        ],
        "full_folders": {
            "nl": {"documents": 100, "nynorsk_documents": 13, "provisions": 1000,
                   "nynorsk_provisions": 100},
            "sf": {"documents": 300, "nynorsk_documents": 10, "provisions": 3000,
                   "nynorsk_provisions": 90},
        },
    }  # fmt: skip
    scope = {
        "resolved_share": 0.9,
        "agreement_with_lovdata_links": {"recall_against_links": 0.88,
                                         "precision_against_links": 0.91},
        "unresolved_reasons": {"section_not_found": 50, "unknown_act_name": 20},
    }  # fmt: skip
    return overview, {"tax_corpus": scope, "full_corpus": scope}


def _write(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_all_figures_in_both_languages(tmp_path):
    results = tmp_path / "results"
    overview, references = _corpus()
    _write(results / "corpus" / "overview.json", overview)
    _write(results / "corpus" / "references.json", references)
    _write(results / "retrieval" / "test.json", _retrieval())
    _write(results / "generation" / "test.json", _generation())
    out = tmp_path / "figures"
    written = build_figures(results, out, ["nb", "en"])
    names = {p.name for p in written}
    assert "sok_recall_at_k.svg" in names and "svar_kilder.png" in names
    # 2 corpus, 6 retrieval and 5 answer figures, in SVG and PNG, in two languages.
    assert len(written) == (2 + 6 + 5) * 2 * 2
    assert (out / "en" / "svar_avslag.svg").exists()
    for path in written:
        assert path.stat().st_size > 0
        if path.suffix == ".svg":
            text = path.read_text(encoding="utf-8")
            assert "<dc:date>" not in text
            assert chr(0x2014) not in text


def test_missing_results_are_skipped(tmp_path):
    results = tmp_path / "results"
    _write(results / "retrieval" / "test.json", _retrieval())
    written = build_figures(results, tmp_path / "figures", ["nb"])
    assert written and all(p.name.startswith("sok_") for p in written)


def test_svg_is_stable(tmp_path):
    results = tmp_path / "results"
    _write(results / "retrieval" / "test.json", _retrieval())
    first = build_figures(results, tmp_path / "a", ["nb"])
    second = build_figures(results, tmp_path / "b", ["nb"])
    for a, b in zip(first, second, strict=True):
        if a.suffix == ".svg":
            assert a.read_bytes() == b.read_bytes()


def test_run_label():
    assert run_label("tax-structure__hybrid+rerank__gemma", "nb") == (
        "Hybrid med omrangering + Gemma 4 31B"
    )
    assert run_label("tax-structure__bm25__normistral", "en") == "BM25 + NorMistral 7B"
