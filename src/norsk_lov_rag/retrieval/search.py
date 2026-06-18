"""Rank sections for any list of questions with one of the retrieval systems.

The generation stage needs rankings for every question, also the unanswerable ones that the
retrieval evaluation leaves out. This module builds the same systems from the same parts, so a
question gets the same ranking here as in the retrieval evaluation.
"""

from __future__ import annotations

import json

import pandas as pd

from norsk_lov_rag.chunking.build import chunk_path
from norsk_lov_rag.config import Settings
from norsk_lov_rag.retrieval.dense import DenseIndex, index_dir
from norsk_lov_rag.retrieval.evaluate import (
    _query_vectors,
    _rerank,
    expand,
    reference_map,
    rrf,
    units_of,
)
from norsk_lov_rag.retrieval.lexical import BM25Index, CompoundSplitter


class Searcher:
    """The retrieval systems of one corpus and chunking, built when first asked for."""

    def __init__(self, settings: Settings, setting: str, split: str) -> None:
        self.settings = settings
        self.scope, self.strategy = setting.split("-", 1)
        self.split = split
        chunks = pd.read_parquet(chunk_path(settings.paths.data_dir, self.scope, self.strategy))
        self.texts = chunks["text"].tolist()
        self.chunk_units = [list(units) for units in chunks["units"]]
        self.folder = index_dir(settings.paths.data_dir, self.scope, self.strategy)
        self._lexical: dict[str, BM25Index] = {}

    def _chunks(self, system: str, queries: list[str]) -> list[list[int]]:
        cfg = self.settings.retrieval
        if system in ("bm25", "bm25-compound"):
            if system not in self._lexical:
                splitter = CompoundSplitter.from_texts(self.texts) if system != "bm25" else None
                self._lexical[system] = BM25Index(self.texts, splitter=splitter)
            index = self._lexical[system]
            return [[i for i, _ in index.search(q, cfg.depth)] for q in queries]
        if system.startswith("dense:"):
            name = system.split(":", 1)[1]
            dense = DenseIndex.load(self.folder, name)
            vectors = _query_vectors(self.settings, name, queries, self.folder / "queries")
            return [[i for i, _ in dense.search(v, cfg.depth)] for v in vectors]
        if system in ("hybrid", "hybrid+rerank"):
            parts = [self._chunks(s, queries) for s in cfg.hybrid[self.scope]]
            fused = [rrf([p[n] for p in parts], cfg.rrf_k) for n in range(len(queries))]
            if system == "hybrid":
                return fused
            return _rerank(self.settings, queries, fused, self.texts, self.folder, self.split)
        raise ValueError(f"unknown retrieval system {system!r}")

    def rank(self, system: str, queries: list[str]) -> list[list[str]]:
        """Section ids (or loose-text ids) in ranked order for each query."""
        base = system.removesuffix("+expansion")
        units = [units_of(r, self.chunk_units) for r in self._chunks(base, queries)]
        if system.endswith("+expansion"):
            known = {unit for chunk in self.chunk_units for unit in chunk}
            refs = reference_map(self.settings.paths.processed, known)
            top = self.settings.retrieval.expansion_top
            units = [expand(u, refs, top) for u in units]
        return units


def load_split(frozen, split: str) -> list[dict]:
    """Every question of a split, the unanswerable ones included."""
    return [json.loads(line) for line in (frozen / f"{split}.jsonl").open(encoding="utf-8")]
