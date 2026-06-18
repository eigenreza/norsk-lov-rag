from __future__ import annotations

import math
from collections import Counter

import pandas as pd
import pytest

from norsk_lov_rag.chunking.build import (
    naive_chunks,
    pack,
    prefix,
    structure_chunks,
    window_units,
)
from norsk_lov_rag.evaluation.retrieval_metrics import (
    bootstrap,
    ndcg_at,
    paired_difference,
    question_scores,
    recall_at,
    reciprocal_rank,
)
from norsk_lov_rag.retrieval.evaluate import compare, expand, rrf, units_of
from norsk_lov_rag.retrieval.lexical import BM25Index, CompoundSplitter, analyze


def test_metrics_match_hand_computed_values() -> None:
    ranking = ["a", "g1", "b", "c", "d", "e", "f", "h", "i", "g2"]
    gold = {"g1", "g2"}
    assert recall_at(ranking, gold, 1) == 0.0
    assert recall_at(ranking, gold, 5) == 0.5
    assert recall_at(ranking, gold, 10) == 1.0
    assert reciprocal_rank(ranking, gold) == 0.5
    # Gains at places 2 and 10, ideal gains at places 1 and 2.
    expected = (1 / math.log2(3) + 1 / math.log2(11)) / (1 + 1 / math.log2(3))
    assert ndcg_at(ranking, gold) == pytest.approx(expected)
    assert question_scores(["x"], {"g"})["mrr"] == 0.0


def test_bootstrap_interval_holds_the_mean() -> None:
    low, high = bootstrap([0.0, 1.0] * 50)
    assert low < 0.5 < high
    assert bootstrap([1.0, 1.0, 1.0]) == (1.0, 1.0)


def test_paired_difference_and_planned_comparisons() -> None:
    diff = paired_difference([1.0, 1.0, 0.0], [0.0, 1.0, 0.0])
    assert diff["mean"] == pytest.approx(1 / 3)
    assert diff["low"] <= diff["mean"] <= diff["high"]
    questions = [
        {"item_id": "q1", "question_type": "direct"},
        {"item_id": "q2", "question_type": "cross_reference"},
    ]
    scores = {
        "s/a": {"q1": {"ndcg@10": 1.0, "recall@10": 1.0}, "q2": {"ndcg@10": 0.5, "recall@10": 1.0}},
        "s/b": {"q1": {"ndcg@10": 0.0, "recall@10": 0.0}, "q2": {"ndcg@10": 0.5, "recall@10": 0.5}},
    }
    rows = compare(
        [["s/a", "s/b"], ["s/a", "s/b", "cross_reference"], ["s/a", "s/x"]], scores, questions
    )
    assert [(r["question_type"], r["n"]) for r in rows] == [("all", 2), ("cross_reference", 1)]
    assert rows[1]["ndcg@10"]["mean"] == 0.0
    assert rows[1]["recall@10"]["mean"] == 0.5


def test_rrf_and_units() -> None:
    assert rrf([[1, 2, 3], [3, 1]], k=60) == [1, 3, 2]
    chunk_units = [["s1"], ["s1"], ["s2", "s3"], ["s3"]]
    assert units_of([1, 0, 3, 2], chunk_units) == ["s1", "s3", "s2"]


def test_expansion_puts_references_after_the_top_results() -> None:
    refs = {"a": ["x", "b"], "b": ["y"]}
    assert expand(["a", "b", "c", "d"], refs, top=2) == ["a", "b", "x", "y", "c", "d"]
    assert expand(["c", "a"], refs, top=1) == ["c", "a"]


def test_compound_splitter_finds_parts_with_linking_letter() -> None:
    counts = Counter({"merverdiavgift": 9, "registeret": 9, "merverdi": 9, "avgift": 9})
    splitter = CompoundSplitter(counts)
    assert splitter.split("merverdiavgiftsregisteret") == ("merverdiavgift", "registeret")
    assert splitter.split("register") == ()
    terms = analyze("Merverdiavgiftsregisteret", splitter)
    assert len(terms) == 3


def test_bm25_ranks_the_matching_text_first() -> None:
    texts = [
        "Skattyter kan kreve fradrag for reisekostnader mellom hjem og arbeidssted.",
        "Merverdiavgift skal beregnes ved omsetning av varer.",
        "Formuesskatt beregnes av nettoformuen.",
    ]
    index = BM25Index(texts)
    assert index.search("fradrag for reise til arbeidssted", k=3)[0][0] == 0
    assert index.search("ukjent ord", k=3) == []


def corpus() -> dict[str, pd.DataFrame]:
    documents = pd.DataFrame(
        [{"doc_key": "lov/x", "short_name": "Testloven", "title": "Lov om test"}]
    )
    sections = pd.DataFrame(
        [
            {
                "section_id": "lov/x/§1",
                "doc_key": "lov/x",
                "section_label": "§ 1",
                "section_title": "Formål",
                "chapter_path": "Kapittel 1. Innledning",
                "text": "En to tre.",
                "is_empty": False,
                "is_repealed": False,
                "written_standard": "nb",
            },
            {
                "section_id": "lov/x/§2",
                "doc_key": "lov/x",
                "section_label": "§ 2",
                "section_title": None,
                "chapter_path": None,
                "text": "",
                "is_empty": True,
                "is_repealed": True,
                "written_standard": "nb",
            },
        ]
    )
    provisions = pd.DataFrame(
        [
            {"provision_id": "lov/x/§1/ledd/1", "section_id": "lov/x/§1", "doc_key": "lov/x",
             "text": "En to tre.", "order": 1, "written_standard": "nb"},
            {"provision_id": "lov/x/§2/ledd/1", "section_id": "lov/x/§2", "doc_key": "lov/x",
             "text": "Opphevet.", "order": 2, "written_standard": "nb"},
            {"provision_id": "lov/x/#ledd-1", "section_id": None, "doc_key": "lov/x",
             "text": "Vedlegg fire fem.", "order": 3, "written_standard": "nb"},
        ]
    )  # fmt: skip
    return {"documents": documents, "sections": sections, "provisions": provisions}


def test_structure_chunks_have_heading_and_skip_repealed_sections() -> None:
    chunks = structure_chunks(corpus())
    assert chunks["chunk_id"].tolist() == ["lov/x/§1", "lov/x#0"]
    assert chunks["text"].iloc[0].startswith("Testloven. Kapittel 1. Innledning. § 1 Formål\n")
    assert chunks["units"].iloc[1] == ["lov/x#0"]


def test_naive_chunks_cover_text_without_heading() -> None:
    chunks = naive_chunks(corpus())
    assert len(chunks) == 1
    assert chunks["text"].iloc[0] == "En to tre. Vedlegg fire fem."
    assert chunks["units"].iloc[0] == ["lov/x/§1", "lov/x##ledd-1"]


def test_window_units_put_the_largest_unit_first() -> None:
    assert window_units(["a", "b", "b", "c"]) == ["b", "a", "c"]
    assert window_units(["a", "a", "b", "b"]) == ["a", "b"]


def test_pack_splits_long_paragraphs_and_keeps_short_ones_together() -> None:
    assert pack(["a b", "c d"], max_words=4) == ["a b\nc d"]
    assert pack(["a b c", "d e f g h"], max_words=4) == ["a b c", "d e f g", "h"]
    assert prefix("Lov", None, "§ 3", None) == "Lov. § 3"
