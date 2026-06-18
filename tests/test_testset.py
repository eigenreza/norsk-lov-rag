"""Test-set stage: sampling, filters, batching, JSON reading, model stages and split, offline."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pandas as pd

from norsk_lov_rag.config import LLMConfig
from norsk_lov_rag.llm import ModelClient
from norsk_lov_rag.testset.pipeline import (
    apply_filters,
    check_outcome,
    freeze_testset,
    longest_copied_run,
    review_outcome,
    split,
    write_jsonl,
)
from norsk_lov_rag.testset.sample import Counts, allocate, build_tasks
from norsk_lov_rag.testset.search import BM25
from norsk_lov_rag.testset.stages import (
    _check_batches,
    batches,
    parse_json,
    run_checks,
    task_block,
    write_tasks,
)

FILLER = "Skattyter kan kreve fradrag for kostnader som har sammenheng med inntekten. " * 3


def corpus() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    documents = pd.DataFrame(
        [
            {"doc_key": "lov/1", "short_title": "Testloven", "title": "Lov om test"},
            {"doc_key": "forskrift/2", "short_title": "Testforskriften", "title": "Forskrift"},
        ]
    )
    sections, provisions = [], []
    order = 0
    for doc in ("lov/1", "forskrift/2"):
        for chapter in (1, 2, 3):
            for number in (1, 2, 3, 4):
                sid = f"{doc}/§{chapter}-{number}"
                title = "Ikrafttredelse" if (chapter, number) == (3, 4) else f"Regel {number}"
                sections.append(
                    {
                        "section_id": sid,
                        "doc_key": doc,
                        "section_label": f"§ {chapter}-{number}",
                        "section_title": title,
                        "chapter_path": f"Kapittel {chapter}",
                        "text": FILLER * 2,
                    }
                )
                for paragraph in (1, 2):
                    order += 1
                    provisions.append(
                        {
                            "provision_id": f"{sid}/ledd/{paragraph}",
                            "section_id": sid,
                            "doc_key": doc,
                            "chapter_path": f"Kapittel {chapter}",
                            "text": f"({paragraph}) {FILLER}",
                            "order": order,
                        }
                    )
    references = pd.DataFrame(
        [
            {
                "source_provision_id": "lov/1/§1-1/ledd/1",
                "source_section_id": "lov/1/§1-1",
                "source_doc_key": "lov/1",
                "raw": "§ 2-1",
                "resolved": True,
                "scope": "same_document",
                "reason": "same_document",
                "target_section_id": "lov/1/§2-1",
                "target_provision_id": None,
            },
            {
                "source_provision_id": "forskrift/2/§1-1/ledd/2",
                "source_section_id": "forskrift/2/§1-1",
                "source_doc_key": "forskrift/2",
                "raw": "§ 1-1",
                "resolved": True,
                "scope": "other_document",
                "reason": "parent_law_fallback",
                "target_section_id": "lov/1/§1-1",
                "target_provision_id": None,
            },
        ]
    )
    tables = {
        "documents": documents,
        "sections": pd.DataFrame(sections),
        "provisions": pd.DataFrame(provisions),
    }
    return tables, references


def test_allocation_follows_square_roots_and_never_exceeds_a_size() -> None:
    shares = allocate({"big": 100, "small": 4, "tiny": 1}, 20)
    assert sum(shares.values()) == 20
    assert shares["big"] > shares["small"] >= shares["tiny"] == 1
    assert allocate({"a": 2, "b": 1}, 10) == {"a": 2, "b": 1}


def test_tasks_use_each_section_once_and_skip_transition_rules() -> None:
    tables, references = corpus()
    counts = Counts(direct_nb=6, everyday_nb=6, direct_nn=2, everyday_nn=2, cross_reference_nb=2)
    tasks = build_tasks(tables, references, counts)
    sections = [task["section_id"] for task in tasks if "section_id" in task]
    assert len(sections) == len(set(sections))
    assert not any(section.endswith("§3-4") for section in sections)
    assert tasks == build_tasks(tables, references, counts)
    only_cross = Counts(
        direct_nb=0, everyday_nb=0, direct_nn=0, everyday_nn=0, cross_reference_nb=2
    )
    tasks = build_tasks(tables, references, only_cross)
    cross = [task for task in tasks if task["question_type"] == "cross_reference"]
    # Only the directly resolved link is used, not the parent-act fallback.
    assert [task["target_provision_id"] for task in cross] == ["lov/1/§1-1/ledd/1"]
    assert {part["side"] for part in cross[0]["parts"]} == {"A", "B"}
    assert "Testloven" in task_block(cross[0]) and "Tekst B" in task_block(cross[0])


def test_batches_respect_size_and_length() -> None:
    entries = [({"id": n}, "x" * 40) for n in range(7)]
    assert [len(b) for b in batches(entries, 3, 1000)] == [3, 3, 1]
    assert [len(b) for b in batches(entries, 3, 90)] == [2, 2, 2, 1]


def test_json_is_read_from_fenced_or_wrapped_text() -> None:
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Her er svaret: {"a": [1, 2]} takk') == {"a": [1, 2]}
    assert parse_json("ikke json") is None


def item(item_id: str, question: str, standard: str = "nb", kind: str = "direct") -> dict:
    return {
        "item_id": item_id,
        "task_id": f"{kind}-{standard}-{item_id}",
        "question_type": kind,
        "written_standard": standard,
        "question": question,
        "reference_answer": "Ja.",
        "gold_texts": [
            "Skattyter kan kreve fradrag for kostnader til reise mellom hjem og arbeidssted."
        ],
    }


def test_filters_remove_copies_duplicates_and_wrong_written_standard() -> None:
    items = [
        item("a", "Kan jeg få fradrag for kostnader til reise mellom hjem og arbeidssted?"),
        item("b", "Når har en lønnsmottaker rett til fradrag for pendling?"),
        item("c", "Når har en lønnsmottaker rett til fradrag for pendling til jobb?"),
        item("d", "Kva kan ein trekkje frå når ein ikkje bur heime?", standard="nb"),
        item("e", "Kva kan ein trekkje frå når ein ikkje bur heime?", standard="nn"),
    ]
    apply_filters(items)
    removed = {i["item_id"]: i.get("removed_by") for i in items}
    assert removed == {
        "a": "overlap",
        "b": None,
        "c": "duplicate",
        "d": "written_standard",
        "e": None,
    }
    assert longest_copied_run("reise mellom hjem og jobb", ["reise mellom hjem og arbeid"]) == 4


def test_questions_that_point_to_unseen_text_are_removed() -> None:
    items = [
        item("a", "Kva skatteplikt omfattar lova i første ledd?", standard="nn"),
        item("b", "Hva gjelder etter § 6-44?"),
        item("c", "Hvilke skatter gjelder for personer bosatt i Norge?"),
    ]
    apply_filters(items)
    assert [i.get("removed_by") for i in items] == ["unseen_context", "unseen_context", None]


def test_check_and_review_outcomes() -> None:
    cross = item("x", "Spørsmål?", kind="cross_reference")
    check_outcome(
        cross,
        {"kan_besvares": True, "svar_stemmer": True, "trenger_begge": False},
        type("R", (), {"model": "m", "model_version": "v"}),
    )
    assert cross["removed_by"] == "check"
    plain = item("y", "Spørsmål?")
    review_outcome(
        plain,
        {"vurdering": "Uklar", "begrunnelse": "to tolkninger"},
        type("R", (), {"model": "m", "model_version": "v"}),
    )
    assert plain["removed_by"] == "review" and "uklar" in plain["removed_reason"]


def test_items_left_out_by_the_model_get_one_retry(tmp_path: Path) -> None:
    contents = []

    def handler(request: httpx.Request) -> httpx.Response:
        content = json.loads(request.content)["messages"][1]["content"]
        contents.append(content)
        ids = [i for i in ("a", "b", "c") if f"Element {i}" in content]
        answered = ids[:1] if len(contents) == 1 else ids  # the first answer skips items
        verdicts = [{"id": i, "vurdering": "korrekt", "begrunnelse": "ok"} for i in answered]
        return httpx.Response(
            200,
            json={
                "model": "m1",
                "choices": [
                    {
                        "message": {"content": json.dumps({"vurderinger": verdicts})},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 5},
            },
        )

    cfg = LLMConfig(
        providers={"groq": {"style": "openai", "base_url": "https://g.test", "api_key_env": "K"}},
        models={
            "reviewer": {
                "provider": "groq",
                "model": "m1",
                "limits": {"requests_per_minute": 10, "requests_per_day": 10},
            }
        },
    )
    client = ModelClient(cfg, tmp_path, transport=httpx.MockTransport(handler), environ={"K": "k"})
    items = [item(i, "Spørsmål?") for i in ("a", "b", "c")]
    run_checks(
        client,
        items,
        "reviewer",
        "system",
        "v1",
        lambda i: f"Element {i['item_id']}",
        review_outcome,
    )
    assert len(contents) == 2 and "id-ene: b, c" in contents[1]
    assert all(i["review"]["vurdering"] == "korrekt" for i in items)
    assert not any("removed_by" in i or "missing_reason" in i for i in items)


def test_split_keeps_a_fixed_share_per_type() -> None:
    items = [item(f"d{n}", "q") for n in range(10)] + [item(f"n{n}", "q", "nn") for n in range(4)]
    split(items)
    assert sum(i["split"] == "dev" for i in items if i["written_standard"] == "nb") == 3
    assert sum(i["split"] == "dev" for i in items if i["written_standard"] == "nn") == 1


def test_bm25_finds_the_text_with_the_query_words() -> None:
    index = BM25(
        ["a", "b", "c"],
        [
            "Merverdiavgift ved innførsel av varer.",
            "Fradrag for reiseutgifter mellom hjem og arbeid.",
            "Formuesskatt på bolig.",
        ],
    )
    assert index.search("reiseutgiftene til arbeidet", k=1)[0][0] == "b"


def test_writer_stage_maps_answers_to_tasks_and_gold_provisions(tmp_path: Path) -> None:
    tables, references = corpus()
    counts = Counts(direct_nb=2, everyday_nb=0, direct_nn=0, everyday_nn=0, cross_reference_nb=0)
    tasks = [t for t in build_tasks(tables, references, counts) if t["question_type"] == "direct"]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["response_format"] == {"type": "json_object"}
        answer = {
            "oppgaver": [
                {
                    "id": tasks[0]["task_id"],
                    "sporsmal": "Når gis fradrag?",
                    "svar": "Når kostnaden har sammenheng med inntekten.",
                    "deler": [2],
                }
            ]
        }
        return httpx.Response(
            200,
            json={
                "model": "m1",
                "choices": [{"message": {"content": json.dumps(answer)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            },
        )

    cfg = LLMConfig(
        providers={"groq": {"style": "openai", "base_url": "https://g.test", "api_key_env": "K"}},
        models={
            "writer": {
                "provider": "groq",
                "model": "m1",
                "limits": {"requests_per_minute": 10, "requests_per_day": 10},
            }
        },
    )
    client = ModelClient(cfg, tmp_path, transport=httpx.MockTransport(handler), environ={"K": "k"})
    items, stopped = write_tasks(client, tasks)
    assert not stopped
    first, second = items
    target = tasks[0]["target_provision_id"]
    assert first["question"] == "Når gis fradrag?" and target in first["gold_provision_ids"]
    assert first["gold_section_ids"] == [tasks[0]["section_id"]]
    assert second["removed_by"] == "format" and second["removed_reason"] == "no answer"


def test_after_a_quota_stop_cached_batches_are_still_read(tmp_path: Path) -> None:
    tables, references = corpus()
    counts = Counts(direct_nb=1, everyday_nb=1, direct_nn=0, everyday_nn=0, cross_reference_nb=0)
    tasks = [t for t in build_tasks(tables, references, counts) if "section_id" in t]

    def handler(request: httpx.Request) -> httpx.Response:
        content = json.loads(request.content)["messages"][1]["content"]
        answers = [
            {"id": t["task_id"], "sporsmal": "Når gis fradrag?", "svar": "Svar.", "deler": [1]}
            for t in tasks
            if t["task_id"] in content
        ]
        return httpx.Response(
            200,
            json={
                "model": "m1",
                "choices": [
                    {
                        "message": {"content": json.dumps({"oppgaver": answers})},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 5},
            },
        )

    cfg = LLMConfig(
        providers={"groq": {"style": "openai", "base_url": "https://g.test", "api_key_env": "K"}},
        models={
            "writer": {
                "provider": "groq",
                "model": "m1",
                "limits": {"requests_per_minute": 10, "requests_per_day": 1},
            }
        },
    )

    def client() -> ModelClient:
        return ModelClient(
            cfg, tmp_path, transport=httpx.MockTransport(handler), environ={"K": "k"}
        )

    everyday = [t for t in tasks if t["question_type"] == "everyday"]
    write_tasks(client(), everyday)  # uses the one request of the day and fills the cache
    items, stopped = write_tasks(client(), tasks)
    by_type = {i["question_type"]: i for i in items}
    assert stopped
    assert by_type["direct"]["pending"] == "write"
    assert by_type["everyday"]["question"] == "Når gis fradrag?"
    assert [i["question_type"] for i in items] == ["direct", "everyday"]


def test_batches_of_one_type_do_not_move_when_another_type_grows(tmp_path: Path) -> None:
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        content = json.loads(request.content)["messages"][1]["content"]
        sent.append(content)
        ids = content.split("id-ene: ")[1].split(".")[0].split(", ")
        verdicts = [{"id": i, "vurdering": "korrekt", "begrunnelse": "ok"} for i in ids]
        return httpx.Response(
            200,
            json={
                "model": "m1",
                "choices": [
                    {
                        "message": {"content": json.dumps({"vurderinger": verdicts})},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 5},
            },
        )

    cfg = LLMConfig(
        providers={"groq": {"style": "openai", "base_url": "https://g.test", "api_key_env": "K"}},
        models={
            "checker": {
                "provider": "groq",
                "model": "m1",
                "limits": {"requests_per_minute": 100, "requests_per_day": 100},
            }
        },
    )
    client = ModelClient(cfg, tmp_path, transport=httpx.MockTransport(handler), environ={"K": "k"})

    def run(direct: int, everyday: int, open_groups: frozenset[str]) -> list[dict]:
        items = [item(f"d{n}", "q") for n in range(direct)]
        items += [item(f"e{n}", "q", kind="everyday") for n in range(everyday)]
        run_checks(
            client,
            items,
            "checker",
            "system",
            "v1",
            lambda i: i["item_id"],
            lambda i, v, r: i.update(done=True),
            open_groups,
        )
        return items

    # The last batch of a type still being written waits.
    items = run(6, 4, frozenset({"direct-nb"}))
    assert len(sent) == 2
    assert [i.get("pending") for i in items[:6]] == [None] * 4 + ["checker"] * 2
    # Five more direct questions leave the batches of the everyday type as they were.
    first_calls = len(sent)
    items = run(11, 4, frozenset())
    assert len(sent) == first_calls + 2  # only the two new batches of the direct type
    assert all(i.get("done") for i in items)


def test_reviewer_gets_full_batches_first() -> None:
    entries = [(item(f"d{n}", "q"), "x") for n in range(25)]
    entries += [(item(f"e{n}", "q", kind="everyday"), "x") for n in range(3)]
    planned, _ = _check_batches(entries, "reviewer", 1, frozenset())
    assert [len(batch) for batch in planned] == [20, 5, 3]


def test_freeze_copies_the_sets_once(tmp_path: Path) -> None:
    source = tmp_path / "data"
    source.mkdir()
    write_jsonl(source / "dev.jsonl", [{"item_id": "a"}])
    write_jsonl(source / "test.jsonl", [{"item_id": "b"}, {"item_id": "c"}])
    report = {"final": {"direct-nb": 3}, "final_split": {"dev": 1, "test": 2}}
    report["model_versions"] = {}
    (source / "report.json").write_text(json.dumps(report), encoding="utf-8")
    target = tmp_path / "testsett"
    manifest = freeze_testset(source, target, "abc", {"reviewer": "google:m"})
    assert manifest["files"]["test.jsonl"]["rows"] == 2
    assert (target / "test.jsonl").read_bytes() == (source / "test.jsonl").read_bytes()
    try:
        freeze_testset(source, target, "abc", {})
    except FileExistsError:
        pass
    else:
        raise AssertionError("a frozen set must not be replaced")
