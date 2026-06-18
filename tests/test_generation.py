from __future__ import annotations

import json

import pandas as pd
import pytest

from norsk_lov_rag.chunking.build import chunk_path, structure_chunks
from norsk_lov_rag.generation.answers import analyse, language_check, parse_output, sentences
from norsk_lov_rag.generation.judge import _batches, _parse
from norsk_lov_rag.generation.prompts import ABSTAIN, answer_messages, judge_messages
from norsk_lov_rag.generation.report import consistency, row_scores
from norsk_lov_rag.generation.sources import (
    SourceBook,
    citation_name,
    find_citations,
    find_mentions,
    normalise_number,
)
from norsk_lov_rag.llm.providers import build_request, parse_answer
from norsk_lov_rag.llm.quota import Limits, QuotaTracker


def test_section_numbers_and_act_names_normalise() -> None:
    assert normalise_number("§ 6-44 første ledd") == "6-44"
    assert normalise_number("§6 \u2013 44") == "6-44"
    assert normalise_number("§ 14 g") == "14g"
    assert normalise_number("§ 7A-24") == "7a-24"
    assert normalise_number("ingen tall") is None
    assert citation_name("Skatteloven") == "skatteloven"
    assert citation_name("FSFIN Forskrift til skatteloven") == "FSFIN"


def corpus() -> dict[str, pd.DataFrame]:
    documents = pd.DataFrame(
        [
            {"doc_key": "lov/s", "short_name": "Skatteloven", "title": "Lov om skatt"},
            {"doc_key": "forskrift/f", "short_name": "FSFIN Forskrift til skatteloven",
             "title": "Forskrift"},
        ]
    )  # fmt: skip
    sections = pd.DataFrame(
        [
            {"section_id": "lov/s/§6-44", "doc_key": "lov/s", "section_label": "§ 6-44",
             "section_title": "Reise", "chapter_path": None, "text": "Fradrag for reise.",
             "is_empty": False, "is_repealed": False, "written_standard": "nb"},
            {"section_id": "forskrift/f/§6-47-12", "doc_key": "forskrift/f",
             "section_label": "§ 6-47-12", "section_title": "Pensjon", "chapter_path": None,
             "text": "Pensjon i minst ti år.", "is_empty": False, "is_repealed": False,
             "written_standard": "nb"},
        ]
    )  # fmt: skip
    provisions = pd.DataFrame(
        [
            {"provision_id": "lov/s/§6-44/ledd/1", "section_id": "lov/s/§6-44", "doc_key": "lov/s",
             "text": "Fradrag for reise mellom hjem og arbeid.", "order": 1,
             "written_standard": "nb"},
            {"provision_id": "forskrift/f/§6-47-12/ledd/1", "section_id": "forskrift/f/§6-47-12",
             "doc_key": "forskrift/f", "text": "Pensjon skal betales i minst ti år.", "order": 1,
             "written_standard": "nb"},
        ]
    )  # fmt: skip
    return {"documents": documents, "sections": sections, "provisions": provisions}


@pytest.fixture
def book(tmp_path) -> SourceBook:
    data = corpus()
    processed = tmp_path / "processed"
    processed.mkdir()
    data["documents"].to_parquet(processed / "documents.parquet")
    data["sections"].to_parquet(processed / "sections.parquet")
    path = chunk_path(tmp_path, "tax", "structure")
    path.parent.mkdir(parents=True)
    structure_chunks(data).to_parquet(path, index=False)
    return SourceBook(tmp_path, processed, "tax")


def test_source_book_labels_and_resolves_citations(book: SourceBook) -> None:
    assert book.labels["lov/s/§6-44"] == "skatteloven § 6-44"
    assert book.labels["forskrift/f/§6-47-12"] == "FSFIN § 6-47-12"
    assert book.resolve("Skatteloven §6\u201344 første ledd") == "lov/s/§6-44"
    assert book.resolve("FSFIN § 6-47-12") == "forskrift/f/§6-47-12"
    assert book.resolve("Forskrift til skatteloven § 6-47-12") == "forskrift/f/§6-47-12"
    assert book.resolve("skatteloven § 99-1") is None
    assert book.resolve("skattelova § 6-44") == "lov/s/§6-44"
    assert find_mentions("Etter skatteloven § 6-44 og [FSFIN § 6-47-12] gjeld dette.") == [
        "skatteloven § 6-44", "FSFIN § 6-47-12"
    ]  # fmt: skip
    assert book.text("lov/s/§6-44") == "Fradrag for reise mellom hjem og arbeid."


def test_sources_respect_the_word_budgets(book: SourceBook) -> None:
    units = ["lov/s/§6-44", "forskrift/f/§6-47-12"]
    sources = book.sources(units, top_k=5, max_words_per_source=3, max_context_words=100)
    assert [s.text for s in sources] == ["Fradrag for reise", "Pensjon skal betales"]
    assert len(book.sources(units, top_k=1, max_words_per_source=50, max_context_words=100)) == 1
    messages = answer_messages("Hva gjelder?", sources)
    assert "[skatteloven § 6-44] Reise\nFradrag for reise" in messages[0]["content"]


def test_parse_output_builds_the_record_from_plain_text() -> None:
    good = parse_output("Ja [x § 1]. Også dette [y § 2] og [x § 1].")
    assert good == {"answer": "Ja [x § 1]. Også dette [y § 2] og [x § 1].",
                    "citations": ["x § 1", "y § 2"], "abstain": False,
                    "valid_output": True}  # fmt: skip
    assert parse_output(f"«{ABSTAIN['nb']}»")["abstain"] is True
    assert parse_output("<im_start> tekst")["valid_output"] is False
    assert parse_output('{"answer": "x"}')["valid_output"] is False
    assert parse_output("")["valid_output"] is False
    assert sentences("Første [a § 1]. Andre setning. 3 kroner.") == [
        "Første [a § 1].", "Andre setning.", "3 kroner."
    ]  # fmt: skip
    assert find_citations("Ja [skatteloven § 6-44].") == ["skatteloven § 6-44"]


def test_analyse_checks_citations_claims_and_language(book: SourceBook) -> None:
    question = {"written_standard": "nb", "gold_section_ids": ["lov/s/§6-44"]}
    parsed = parse_output(
        "Du kan få fradrag for reisen [skatteloven § 6-44]. Det er ikke noe mer "
        "[FSFIN § 6-47-12]. Dette er også en regel [skatteloven § 1-1]."
    )
    result = analyse(parsed, question, ["lov/s/§6-44"], book)
    by_citation = {c["citation"]: c for c in result["citations"]}
    assert by_citation["skatteloven § 6-44"]["in_sources"] is True
    assert by_citation["FSFIN § 6-47-12"]["in_sources"] is False
    assert by_citation["skatteloven § 1-1"]["unit"] is None
    assert result["cited_gold"] == ["lov/s/§6-44"]
    assert result["sentences_with_citation"] == result["sentences"] == 3
    result = analyse(parse_output(ABSTAIN["nn"]), {"written_standard": "nb"}, [], book)
    assert result["abstained"] and result["language"] == "mismatch"
    assert language_check("Eg kan ikkje sjå noko her.", "nn") == "match"


def test_judge_batches_are_fixed_and_bad_entries_are_dropped() -> None:
    items = [{"id": f"i{n}"} for n in range(7)]
    assert _batches(items, 3, seed=0) == _batches(list(reversed(items)), 3, seed=0)
    assert [len(b) for b in _batches(items, 3, seed=0)] == [3, 3, 1]
    batch = [
        {"id": "a", "reference": "Ja.", "citations": [{"citation": "x § 1"}]},
        {"id": "b", "reference": None, "citations": [{"citation": "y § 2"}]},
        {"id": "c", "reference": "Nei.", "citations": []},
    ]
    text = json.dumps({"items": [
        {"id": "a", "correctness": "riktig", "support": [{"citation": "x § 1", "supported": True}]},
        {"id": "b", "correctness": "feil", "support": [{"citation": "y § 2", "supported": False}]},
        {"id": "c", "correctness": "kanskje", "support": []},
    ]})  # fmt: skip
    found = _parse(text, batch)
    assert found["a"]["correctness"] == "riktig" and found["a"]["support"] == {"x § 1": True}
    assert found["b"]["correctness"] is None and found["b"]["support"] == {"y § 2": False}
    assert "c" not in found
    cut = json.dumps(json.loads(text), indent=2)
    cut = cut[: cut.index('"b"') + 20]
    assert set(_parse(cut, batch)) == {"a"}
    prompt = judge_messages([{"id": "a", "question": "Q?", "reference": None, "answer": "A.",
                              "citations": []}])[1]["content"]  # fmt: skip
    assert "Referansesvar" not in prompt


def test_row_scores_count_refusals_and_unknown_citations() -> None:
    question = {"question_type": "direct", "gold_section_ids": ["s1", "s2"]}
    refused = {"abstained": True, "citations": [], "cited_gold": [], "sentences": 0,
               "sentences_with_citation": 0}  # fmt: skip
    scores = row_scores(refused, question, None)
    assert scores["score"] == 0.0 and scores["recall"] == 0.0 and scores["precision"] is None
    answered = {
        "abstained": False,
        "citations": [{"citation": "a", "unit": "s1", "in_sources": True},
                      {"citation": "b", "unit": None, "in_sources": False}],
        "cited_gold": ["s1"], "sentences": 2, "sentences_with_citation": 1,
    }  # fmt: skip
    verdict = {"correctness": "delvis", "support": {"a": True}}
    scores = row_scores(answered, question, verdict)
    assert scores["score"] == 0.5 and scores["strict"] == 0.0
    assert scores["recall"] == 0.5 and scores["precision"] == 0.5
    assert scores["support"] == 0.5 and scores["unresolved"] == 0.5
    assert scores["all_cited"] == 0.0
    lenient = row_scores(answered, question, verdict, mentioned=["s1", "s2", None])
    assert lenient["lenient_recall"] == 1.0
    assert lenient["lenient_precision"] == pytest.approx(2 / 3)
    unanswerable = row_scores(answered, {"question_type": "unanswerable"}, None)
    assert unanswerable["score"] is None and unanswerable["recall"] is None


def test_consistency_reports_agreement_and_kappa() -> None:
    def row(a: str, b: str) -> dict:
        return {"first": {"correctness": a, "support": {"x": True}},
                "second": {"correctness": b, "support": {"x": True}}}  # fmt: skip

    result = consistency([row("riktig", "riktig"), row("feil", "feil"), row("riktig", "delvis")])
    assert result["correctness_agreement"] == pytest.approx(2 / 3)
    assert 0 < result["correctness_kappa"] < 1
    assert result["support_agreement"] == 1.0


def test_ollama_requests_and_answers() -> None:
    request = build_request(
        "ollama", "http://localhost:11434", "", "m", [{"role": "user", "content": "Hei"}],
        100, 0.0, True, {"options": {"temperature": 0.3, "seed": 0}, "keep_alive": "5m"},
        schema={"type": "object"},
    )  # fmt: skip
    assert request.url == "http://localhost:11434/api/chat"
    assert request.body["options"] == {"temperature": 0.3, "num_predict": 100, "seed": 0}
    assert request.body["format"] == {"type": "object"} and request.body["keep_alive"] == "5m"
    answer = parse_answer("ollama", {"model": "m", "message": {"content": "{}"},
                                     "prompt_eval_count": 10, "eval_count": 5,
                                     "done_reason": "length"}, {})  # fmt: skip
    assert (answer.total_tokens, answer.finish_reason) == (15, "length")


def test_quota_keeps_other_processes_models(tmp_path) -> None:
    path = tmp_path / "quota.json"
    limits = {"a": Limits(requests_per_minute=5, requests_per_day=9),
              "b": Limits(requests_per_minute=5, requests_per_day=9)}  # fmt: skip
    first = QuotaTracker(path, limits, clock=lambda: 1000.0)
    second = QuotaTracker(path, limits, clock=lambda: 1000.0)
    first.record("a", 10)
    second.record("b", 20)
    stored = json.loads(path.read_text())
    assert set(stored) == {"a", "b"}
