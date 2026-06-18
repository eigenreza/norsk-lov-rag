"""The demo logic and pages, with a tiny data folder and no network calls."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from norsk_lov_rag.config import Settings
from norsk_lov_rag.demo import core
from norsk_lov_rag.demo.core import (
    MAX_QUESTION_CHARS,
    DemoData,
    NullCache,
    RateLimiter,
    answer_question,
    check_question,
    closest_example,
    log_event,
)
from norsk_lov_rag.llm import ChatResult, QuotaExhausted

ROOT = Path(__file__).resolve().parents[1]
UNIT_A = "lov/1999-03-26-14/§6-44"
UNIT_B = "lov/1999-03-26-14/§6-1"


def write_demo_data(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    texts = [
        "Skatteloven. § 6-44 Reise mellom hjem og arbeid\n(1) Det gis fradrag for reise mellom "
        "hjem og fast arbeidssted.",
        "Skatteloven. § 6-1 Fradrag for kostnader\n(1) Det gis fradrag for kostnader som er "
        "pådratt for å erverve skattepliktig inntekt.",
    ]
    files = {
        "chunks.json": {"texts": texts, "units": [[UNIT_A], [UNIT_B]]},
        "sections.json": {
            "labels": {UNIT_A: "skatteloven § 6-44", UNIT_B: "skatteloven § 6-1"},
            "titles": {UNIT_A: "Reise mellom hjem og arbeid", UNIT_B: "Fradrag for kostnader"},
            "texts": {UNIT_A: [texts[0].split("\n", 1)[1]], UNIT_B: [texts[1].split("\n", 1)[1]]},
            "lookup": [["skatteloven", "6-44", UNIT_A], ["skatteloven", "6-1", UNIT_B]],
        },
        "examples.json": {
            "run": "x",
            "split": "test",
            "items": [
                {
                    "item_id": "everyday-nb-001",
                    "question": "Kan jeg trekke fra reise mellom hjem og arbeid?",
                    "question_type": "everyday",
                    "written_standard": "nb",
                    "reference_answer": "Ja.",
                    "answer": "Ja, det gis fradrag [skatteloven § 6-44].",
                    "abstained": False,
                    "model": "google:gemma-4-31b-it",
                    "model_version": "gemma-4-31b-it",
                    "system": "tax-structure/hybrid+rerank",
                    "citations": [
                        {"citation": "skatteloven § 6-44", "unit": UNIT_A, "in_sources": True}
                    ],
                    "sources": [{"unit": UNIT_A, "label": "skatteloven § 6-44"}],
                    "correctness": "riktig",
                }
            ],
        },
        "manifest.json": {"attribution": "Inneholder data under NLOD.", "sections": 2},
    }
    for name, content in files.items():
        (folder / name).write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
    return folder


@pytest.fixture(scope="module")
def demo_folder(tmp_path_factory) -> Path:
    return write_demo_data(tmp_path_factory.mktemp("demo_data"))


@pytest.fixture
def demo(demo_folder) -> DemoData:
    return DemoData.load(demo_folder)


class FakeClient:
    def __init__(self, text: str | None = None, error: Exception | None = None):
        self.text, self.error, self.calls = text, error, 0

    def chat(self, role, messages, prompt_version, max_tokens, **_):
        self.calls += 1
        if self.error:
            raise self.error
        return ChatResult(self.text, role, "google:gemma", "gemma-v", 100, 20, 120, "STOP",
                          False, "k", 1.5)  # fmt: skip


def test_search_ranks_sections_with_labels(demo):
    hits = demo.search("reise mellom hjem og arbeid", "bm25", k=5)
    assert hits[0]["unit"] == UNIT_A
    assert hits[0]["label"] == "skatteloven § 6-44"
    assert demo.search("reise hjem", "bm25-compound", k=5)[0]["unit"] == UNIT_A


def test_live_answer_resolves_citations(demo):
    hits = demo.search("reise mellom hjem og arbeid", "bm25")
    client = FakeClient("Det gis fradrag for reise mellom hjem og arbeid [skatteloven § 6-44].")
    answer = answer_question(demo, Settings(), client, "Får jeg fradrag for reise?", hits)
    assert answer.outcome == "answer" and not answer.abstained
    cite = answer.citations[0]
    assert cite["unit"] == UNIT_A and cite["in_sources"]
    assert "fast arbeidssted" in cite["text"]
    assert answer.prompt_tokens == 100


def test_abstention_and_quota_fall_back(demo):
    hits = demo.search("satsen for trygdeavgift", "bm25")
    refusal = FakeClient("Jeg finner ikke svaret i lovtekstene jeg har fått.")
    answer = answer_question(demo, Settings(), refusal, "Hva er satsen?", hits)
    assert answer.outcome == "abstain" and answer.abstained
    quota = FakeClient(error=QuotaExhausted("daily limit"))
    assert answer_question(demo, Settings(), quota, "Hva er satsen?", hits).outcome == (
        "replay_quota"
    )


def test_question_checks():
    assert check_question("   ") == "empty"
    assert check_question("x" * (MAX_QUESTION_CHARS + 1)) == "too_long"
    assert check_question("Kan jeg trekke fra reise?") is None


def test_rate_limiter_counts_per_visitor_and_resets_each_day():
    now = [0.0]
    limiter = RateLimiter(per_ip=2, total=3, clock=lambda: now[0])
    assert limiter.allow("1.2.3.4") and limiter.allow("1.2.3.4")
    assert not limiter.allow("1.2.3.4")
    assert limiter.allow("5.6.7.8")
    assert not limiter.allow("9.9.9.9")  # the total for the day is used
    now[0] += 86400
    assert limiter.allow("1.2.3.4")
    assert "1.2.3.4" not in repr(limiter._counts)


def test_log_keeps_no_question_text(caplog):
    question = "Kan Ola Nordmann trekke fra reise?"
    with caplog.at_level(logging.INFO, logger="norsk_lov_rag.demo"):
        record = log_event(question, "answer", 1.2, 100, 20)
    assert "Ola" not in caplog.text and "reise" not in caplog.text
    assert record["length"] == len(question) and len(record["sha256"]) == 64
    assert set(record) == {"event", "sha256", "length", "seconds", "prompt_tokens",
                           "completion_tokens", "outcome"}  # fmt: skip


def test_client_stores_no_answers(tmp_path):
    settings = Settings(
        llm={
            "providers": {"g": {"style": "gemini", "base_url": "https://example.invalid",
                                "api_key_env": "X"}},
            "models": {"demo": {"provider": "g", "model": "m", "max_retries": 8,
                                 "limits": {"requests_per_minute": 1, "requests_per_day": 1}}},
        }
    )  # fmt: skip
    client = core.make_client(settings, tmp_path)
    assert isinstance(client.cache, NullCache)
    assert client.cache.get("k") is None
    assert settings.llm.models["demo"].max_retries == core.DEMO_MAX_RETRIES
    client.close()


def test_closest_example(demo):
    assert closest_example("reise mellom hjem og arbeid", demo.examples)["item_id"] == (
        "everyday-nb-001"
    )
    assert closest_example("xyz", demo.examples) is None


def test_pages_render_in_replay_mode(demo_folder, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("DEMO_DATA_DIR", str(demo_folder))
    monkeypatch.setenv("DEMO_REPLAY", "1")
    at = testing.AppTest.from_file(str(ROOT / "app" / "Home.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert any("ikke juridisk råd" in w.value for w in at.warning)
    at.chat_input[0].set_value("Kan jeg trekke fra reise mellom hjem og arbeid?").run()
    assert not at.exception
    assert any("slått av" in w.value for w in at.warning)
    assert at.dataframe[0].value.shape[0] >= 1
    assert any(m.name == "user" for m in at.chat_message)
    for page in ("examples", "compare", "results", "about"):
        at.switch_page(f"pages/{page}.py")
        at.run()
        assert not at.exception, page
        assert any("ikke juridisk råd" in w.value for w in at.warning), page
        assert any("NLOD" in c.value for c in at.caption), page


def test_instruction_words_are_left_out_of_the_search():
    from norsk_lov_rag.demo.core import search_text

    assert (
        search_text("Kan jeg trekke fra reiseutgifter? Kort svar")
        == "Kan jeg trekke fra reiseutgifter?"
    )
    assert search_text("Forklar enkelt") == "Forklar enkelt"
    assert search_text("Hva gjelder for skattekort?") == "Hva gjelder for skattekort?"


def test_english_questions_are_searched_with_the_terms_only(demo):
    assert core.is_english("Will I pay less tax if I have children?")
    assert not core.is_english("Vil jeg betale mindre skatt hvis jeg har barn?")
    plain = demo.search("Can I deduct travel to work?", "bm25", k=5)
    with_terms = demo.search("Can I deduct travel to work?", "bm25", k=5, terms="reise hjem")
    assert with_terms[0]["unit"] == UNIT_A
    assert [h["unit"] for h in plain] != [h["unit"] for h in with_terms] or plain[0][
        "unit"
    ] == UNIT_A


def test_model_picks_sections_and_keeps_the_rest(demo):
    candidates = [{"unit": u, "label": demo.book.labels[u], "title": demo.book.titles.get(u, "")}
                  for u in demo.book.labels]  # fmt: skip
    assert len(candidates) >= 2
    picked = core.select_sections(FakeClient("2, 1, 99"), demo, "Reise?", candidates, k=1)
    assert picked[0]["unit"] == candidates[1]["unit"]
    assert {c["unit"] for c in picked} == {c["unit"] for c in candidates}
    failing = FakeClient(error=QuotaExhausted("daily limit"))
    assert core.select_sections(failing, demo, "Reise?", candidates, k=1) == candidates


def test_answer_prompt_allows_partial_answers(demo):
    seen = {}

    class Recorder(FakeClient):
        def chat(self, role, messages, prompt_version, max_tokens, **kw):
            seen["prompt"] = messages[-1]["content"]
            return super().chat(role, messages, prompt_version, max_tokens, **kw)

    hits = demo.search("reise mellom hjem og arbeid", "bm25")
    answer_question(demo, Settings(), Recorder("Ja [skatteloven § 6-44]."), "Reise?", hits)
    assert "særregel" in seen["prompt"] and "[skatteloven § 6-44]" in seen["prompt"]


def test_round_bracket_citations_are_read(demo):
    hits = demo.search("reise mellom hjem og arbeid", "bm25")
    client = FakeClient("Det gis fradrag for reise mellom hjem og arbeid (skatteloven § 6-44).")
    answer = answer_question(demo, Settings(), client, "Får jeg fradrag for reise?", hits)
    assert answer.citations and answer.citations[0]["unit"] == UNIT_A
    text = "se (skatteloven § 6-44) og (to år)"
    assert core.ROUND_CITATION.sub(r"[\1]", text) == "se [skatteloven § 6-44] og (to år)"


def test_core_sections_join_the_candidates(demo):
    core_units = core.core_candidates(demo, set())
    assert all(c["label"] in core.CORE_SECTIONS for c in core_units)
    present = {c["unit"] for c in core_units}
    assert core.core_candidates(demo, present) == []


def test_routing_reads_the_model_reply():
    reply = '{"type": "about", "question": "Hvem laget demoen?", "terms": ""}'
    route = core.route_message(FakeClient(reply), "hvem lagde dette?", [])
    assert (route.kind, route.question) == ("about", "Hvem laget demoen?")
    broken = core.route_message(FakeClient("not json"), "Er barnetrygd skattefri?", [])
    assert broken.kind == "tax" and broken.question == "Er barnetrygd skattefri?"
