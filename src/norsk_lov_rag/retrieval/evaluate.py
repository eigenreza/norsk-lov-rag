"""Run every retrieval system on one split of the frozen test set and score it.

Systems: BM25 with and without compound splitting, one dense system per encoder, a hybrid of the
lexical and dense systems named in the settings (reciprocal rank fusion), the hybrid reranked by a
cross-encoder, and both of these with reference expansion. Unanswerable questions have no gold
sections and are left out here.

The hybrid is chosen on the development set. On the test set it must already be named in the
settings, so that nothing is chosen by looking at test results.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from norsk_lov_rag.chunking.build import chunk_path
from norsk_lov_rag.config import Settings
from norsk_lov_rag.evaluation.retrieval_metrics import (
    NDCG_K,
    dedupe,
    paired_difference,
    question_scores,
    summarise,
)
from norsk_lov_rag.retrieval.dense import (
    DenseIndex,
    encode,
    index_dir,
    load_encoder,
    load_reranker,
    rerank,
)
from norsk_lov_rag.retrieval.lexical import BM25Index, CompoundSplitter

log = logging.getLogger(__name__)

KEEP_UNITS = 20
LEXICAL = ("bm25", "bm25-compound")


def load_questions(folder: Path, split: str) -> list[dict]:
    rows = [json.loads(line) for line in (folder / f"{split}.jsonl").open(encoding="utf-8")]
    return [row for row in rows if row["question_type"] != "unanswerable"]


def rrf(rankings: list[list[int]], k: int) -> list[int]:
    """Reciprocal rank fusion: each list adds 1 / (k + place) to the items it holds."""
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for place, item in enumerate(ranking, start=1):
            scores[item] += 1.0 / (k + place)
    return sorted(scores, key=lambda item: (-scores[item], item))


def units_of(ranking: list[int], chunk_units: list[list[str]]) -> list[str]:
    return dedupe([unit for row in ranking for unit in chunk_units[row]])


def reference_map(processed: Path, known: set[str]) -> dict[str, list[str]]:
    """For each section, the other sections it refers to, in the order they are cited."""
    refs = pd.read_parquet(
        processed / "references.parquet",
        columns=["source_section_id", "target_section_id", "resolved"],
    )
    refs = refs[refs["resolved"] & refs["target_section_id"].notna()]
    refs = refs[refs["source_section_id"] != refs["target_section_id"]]
    refs = refs[refs["source_section_id"].isin(known) & refs["target_section_id"].isin(known)]
    refs = refs.drop_duplicates(["source_section_id", "target_section_id"])
    return refs.groupby("source_section_id", sort=False)["target_section_id"].agg(list).to_dict()


def expand(units: list[str], refs: dict[str, list[str]], top: int) -> list[str]:
    """The first ``top`` units, then the sections they refer to, then the rest."""
    head = units[:top]
    added = [target for unit in head for target in refs.get(unit, [])]
    return dedupe(head + added + units[top:])


def _hash(texts: list[str]) -> str:
    return hashlib.sha256("\n".join(texts).encode()).hexdigest()[:16]


def _query_vectors(settings: Settings, name: str, questions: list[str], cache: Path) -> np.ndarray:
    spec = settings.retrieval.encoders[name]
    path = cache / f"{name}-{_hash(questions)}.npy"
    if path.exists():
        return np.load(path)
    model = load_encoder(spec, settings.retrieval.model_dir)
    vectors = encode(model, questions, spec.query_prefix, spec.batch_size)
    cache.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors


def _per_question(rankings: dict[str, list[str]], questions: list[dict]) -> dict[str, dict]:
    return {
        q["item_id"]: question_scores(rankings[q["item_id"]], set(q["gold_section_ids"]))
        for q in questions
    }


def _score(per_question: dict[str, dict], questions: list[dict]) -> dict:
    rows = [(question, per_question[question["item_id"]]) for question in questions]
    out = {"n": len(rows), "all": summarise([s for _, s in rows])}
    for field, name in (("question_type", "by_type"), ("written_standard", "by_standard")):
        groups: dict[str, list] = defaultdict(list)
        for question, scores in rows:
            groups[question[field]].append(scores)
        out[name] = {key: {"n": len(v), **summarise(v)} for key, v in sorted(groups.items())}
    return out


def evaluate_setting(
    settings: Settings, split: str, scope: str, strategy: str, questions: list[dict], runs: Path
) -> dict:
    cfg = settings.retrieval
    chunks = pd.read_parquet(chunk_path(settings.paths.data_dir, scope, strategy))
    texts = chunks["text"].tolist()
    chunk_units = [list(units) for units in chunks["units"]]
    known = {unit for units in chunk_units for unit in units}
    missing = sum(1 for q in questions for g in q["gold_section_ids"] if g not in known)
    queries = [q["question"] for q in questions]
    folder = index_dir(settings.paths.data_dir, scope, strategy)
    chunk_rankings: dict[str, list[list[int]]] = {}

    splitter = CompoundSplitter.from_texts(texts)
    for name, use in (("bm25", None), ("bm25-compound", splitter)):
        index = BM25Index(texts, splitter=use)
        chunk_rankings[name] = [[i for i, _ in index.search(q, cfg.depth)] for q in queries]
        log.info("%s %s-%s: %s done", split, scope, strategy, name)

    index_meta = {}
    for name in cfg.encoders:
        if not (folder / f"{name}.npy").exists():
            continue
        index_meta[name] = json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))
        dense = DenseIndex.load(folder, name)
        vectors = _query_vectors(settings, name, queries, folder / "queries")
        chunk_rankings[f"dense:{name}"] = [
            [i for i, _ in dense.search(v, cfg.depth)] for v in vectors
        ]
        log.info("%s %s-%s: dense %s done", split, scope, strategy, name)

    unit_rankings = {
        system: [units_of(r, chunk_units) for r in ranks]
        for system, ranks in chunk_rankings.items()
    }
    hybrid = list(cfg.hybrid.get(scope, []))
    if not hybrid:
        if split == "test":
            raise ValueError("name the hybrid systems in the settings before running the test set")
        hybrid = _choose_hybrid(unit_rankings, questions)
    hybrid = [system for system in hybrid if system in chunk_rankings]
    if len(hybrid) == 2:
        fused = [
            rrf([chunk_rankings[s][n] for s in hybrid], cfg.rrf_k) for n in range(len(queries))
        ]
        chunk_rankings["hybrid"] = fused
        unit_rankings["hybrid"] = [units_of(r, chunk_units) for r in fused]
        if cfg.reranker is not None and strategy == "structure":
            reranked = _rerank(settings, queries, fused, texts, folder, split)
            chunk_rankings["hybrid+rerank"] = reranked
            unit_rankings["hybrid+rerank"] = [units_of(r, chunk_units) for r in reranked]
        refs = reference_map(settings.paths.processed, known)
        for system in ("hybrid", "hybrid+rerank"):
            if system in unit_rankings:
                unit_rankings[f"{system}+expansion"] = [
                    expand(u, refs, cfg.expansion_top) for u in unit_rankings[system]
                ]

    out_dir = runs / split / f"{scope}-{strategy}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    per_question = {}
    for system, rankings in unit_rankings.items():
        by_id = {q["item_id"]: ranking for q, ranking in zip(questions, rankings, strict=True)}
        with (out_dir / f"{system.replace(':', '-')}.jsonl").open("w", encoding="utf-8") as handle:
            for q in questions:
                row = {"item_id": q["item_id"], "units": by_id[q["item_id"]][:KEEP_UNITS]}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        per_question[system] = _per_question(by_id, questions)
        results[system] = _score(per_question[system], questions)
    return {
        "chunks": len(chunks),
        "gold_units_missing": missing,
        "hybrid": hybrid,
        "index": index_meta,
        "systems": results,
        "per_question": per_question,
    }


COMPARE_METRICS = (f"ndcg@{NDCG_K}", "recall@10")


def compare(
    pairs: list[list[str]], per_question: dict[str, dict[str, dict]], questions: list[dict]
) -> list[dict]:
    """Paired differences for the planned comparisons; a pair with a missing system is skipped."""
    out = []
    for a, b, *rest in pairs:
        kind = rest[0] if rest else None
        if a not in per_question or b not in per_question:
            log.warning("comparison %s vs %s skipped: system not run", a, b)
            continue
        chosen = [q["item_id"] for q in questions if kind is None or q["question_type"] == kind]
        row = {"a": a, "b": b, "question_type": kind or "all", "n": len(chosen)}
        for metric in COMPARE_METRICS:
            row[metric] = paired_difference(
                [per_question[a][i][metric] for i in chosen],
                [per_question[b][i][metric] for i in chosen],
            )
        out.append(row)
    return out


def _choose_hybrid(unit_rankings: dict[str, list[list[str]]], questions: list[dict]) -> list[str]:
    """The best lexical and the best dense system by nDCG on this (development) split."""

    def ndcg(system: str) -> float:
        by_id = {q["item_id"]: r for q, r in zip(questions, unit_rankings[system], strict=True)}
        return _score(_per_question(by_id, questions), questions)["all"][f"ndcg@{NDCG_K}"]["mean"]

    lexical = max((s for s in unit_rankings if s in LEXICAL), key=ndcg)
    dense = [s for s in unit_rankings if s.startswith("dense:")]
    return [lexical, max(dense, key=ndcg)] if dense else [lexical]


def _rerank(
    settings: Settings,
    queries: list[str],
    fused: list[list[int]],
    texts: list[str],
    folder: Path,
    split: str,
) -> list[list[int]]:
    spec = settings.retrieval.reranker
    cache_path = folder / f"rerank-{split}.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    model = None
    out = []
    for query, ranking in zip(queries, fused, strict=True):
        head, tail = ranking[: spec.depth], ranking[spec.depth :]
        key = _hash([query])
        stored = cache.get(key, {})
        todo = [i for i in head if str(i) not in stored]
        if todo:
            model = model or load_reranker(spec, settings.retrieval.model_dir)
            for i, score in zip(
                todo, rerank(model, query, [texts[i] for i in todo], spec.batch_size), strict=True
            ):
                stored[str(i)] = score
            cache[key] = stored
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
        out.append(sorted(head, key=lambda i: -stored[str(i)]) + tail)
    return out


def evaluate_retrieval(settings: Settings, split: str, frozen: Path, runs: Path) -> dict:
    questions = load_questions(frozen, split)
    report = {"split": split, "questions": len(questions), "settings": {}}
    per_question = {}
    for scope in settings.retrieval.scopes:
        for strategy in settings.retrieval.strategies:
            if not chunk_path(settings.paths.data_dir, scope, strategy).exists():
                continue
            setting = f"{scope}-{strategy}"
            result = evaluate_setting(settings, split, scope, strategy, questions, runs)
            for system, scores in result.pop("per_question").items():
                per_question[f"{setting}/{system}"] = scores
            report["settings"][setting] = result
    report["comparisons"] = compare(settings.retrieval.comparisons, per_question, questions)
    return report
