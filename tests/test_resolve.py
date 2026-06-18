"""The resolver, the graph and the quality checks, on small made-up tables."""

from __future__ import annotations

import pandas as pd
import pytest

from norsk_lov_rag.corpus.build import corpus_hash
from norsk_lov_rag.corpus.quality import run_checks
from norsk_lov_rag.references.graph import (
    build_edges,
    compare_with_links,
    link_section_target,
    summarise,
)
from norsk_lov_rag.references.parser import find_references
from norsk_lov_rag.references.resolve import Resolver, name_forms

LAW = "lov/2001-02-03-4"
OTHER = "lov/1999-03-26-14"
REGULATION = "forskrift/2002-01-01-10"


def tables() -> dict[str, pd.DataFrame]:
    documents = pd.DataFrame(
        [
            {"doc_key": LAW, "legacy_id": "LOV-2001-02-03-4", "folder": "nl", "kind": "lov",
             "short_name": "Prøveloven", "abbreviation": "prl", "based_on": [],
             "written_standard": "nb"},
            {"doc_key": OTHER, "legacy_id": "LOV-1999-03-26-14", "folder": "nl", "kind": "lov",
             "short_name": "Skatteloven", "abbreviation": "sktl", "based_on": [],
             "written_standard": "nb"},
            {"doc_key": REGULATION, "legacy_id": "FOR-2002-01-01-10", "folder": "sf",
             "kind": "forskrift", "short_name": "Prøveforskriften", "abbreviation": "",
             "based_on": [LAW], "written_standard": "nb"},
        ]
    )  # fmt: skip
    section_rows, provision_rows = [], []
    layout = {
        LAW: ["1-1", "1-2", "1-3", "1-4"],
        OTHER: ["6-44", "14-4"],
        REGULATION: ["1-1-1", "1-1-2"],
    }
    for doc_key, numbers in layout.items():
        for number in numbers:
            section_id = f"{doc_key}/§{number}"
            section_rows.append(
                {"section_id": section_id, "doc_key": doc_key, "section_number": number,
                 "is_repealed": False, "is_empty": False}
            )  # fmt: skip
            for paragraph in (1, 2):
                provision_rows.append(
                    {"provision_id": f"{section_id}/ledd/{paragraph}", "section_id": section_id,
                     "doc_key": doc_key, "paragraph": paragraph, "text": "tekst"}
                )  # fmt: skip
    return {
        "documents": documents,
        "sections": pd.DataFrame(section_rows),
        "provisions": pd.DataFrame(provision_rows),
    }


@pytest.fixture(scope="module")
def resolver() -> Resolver:
    t = tables()
    return Resolver(t["documents"], t["sections"], t["provisions"])


def resolve(resolver: Resolver, text: str, source=LAW, section="1-1", last_named=None):
    reference = find_references(text)[0]
    return resolver.resolve(reference, source, f"{source}/§{section}", last_named)


def test_bare_reference_points_into_the_same_document(resolver: Resolver) -> None:
    target = resolve(resolver, "jf. § 1-2")
    assert target.resolved and target.section_id == f"{LAW}/§1-2" and target.level == "section"


def test_paragraph_level_and_fallback_to_the_section(resolver: Resolver) -> None:
    target = resolve(resolver, "jf. § 1-2 annet ledd")
    assert target.provision_id == f"{LAW}/§1-2/ledd/2" and target.level == "paragraph"
    missing = resolve(resolver, "jf. § 1-2 femte ledd")
    assert missing.resolved and missing.provision_id is None
    assert missing.reason == "paragraph_not_found_section_used"


@pytest.mark.parametrize(
    "text",
    [
        "etter skatteloven § 6-44",
        "etter skattelova § 6-44",
        "etter sktl. § 6-44",
        "etter lov 26. mars 1999 nr. 14 om skatt (skatteloven) § 6-44",
    ],
)
def test_named_act_in_any_form(resolver: Resolver, text: str) -> None:
    target = resolve(resolver, text)
    assert target.resolved and target.section_id == f"{OTHER}/§6-44"


def test_unknown_names_and_missing_sections_stay_unresolved(resolver: Resolver) -> None:
    assert resolve(resolver, "etter fantasiloven § 1").reason == "unknown_act_name"
    assert resolve(resolver, "etter lov 1. januar 1950 nr. 9 § 1").reason == (
        "cited_document_not_in_corpus"
    )
    assert resolve(resolver, "jf. § 99").reason == "section_not_found"


def test_the_law_in_a_regulation_is_the_act_it_is_based_on(resolver: Resolver) -> None:
    target = resolve(resolver, "etter loven § 1-3", source=REGULATION, section="1-1-1")
    assert target.section_id == f"{LAW}/§1-3" and target.reason == "parent_law"
    own = resolve(resolver, "etter forskriften § 1-1-2", source=REGULATION, section="1-1-1")
    assert own.section_id == f"{REGULATION}/§1-1-2"
    assert resolve(resolver, "etter forskriften § 3").reason == "regulation_not_named"


def test_bare_number_missing_in_a_regulation_falls_back_to_its_act(resolver: Resolver) -> None:
    target = resolve(resolver, "jf. § 1-4", source=REGULATION, section="1-1-1")
    assert target.section_id == f"{LAW}/§1-4" and target.reason == "parent_law_fallback"


def test_this_regulation_never_falls_back_to_its_act(resolver: Resolver) -> None:
    target = resolve(resolver, "etter denne forskrift § 1-4", source=REGULATION, section="1-1-1")
    assert not target.resolved and target.reason == "section_not_found"


def test_bare_number_can_follow_the_act_named_earlier(resolver: Resolver) -> None:
    target = resolve(resolver, "og § 14-4", last_named=OTHER)
    assert target.section_id == f"{OTHER}/§14-4" and target.reason == "last_named_act"


def test_same_section_paragraph(resolver: Resolver) -> None:
    target = resolve(resolver, "som nevnt i annet ledd", section="1-3")
    assert target.provision_id == f"{LAW}/§1-3/ledd/2" and target.reason == "same_section"
    assert not resolve(resolver, "som nevnt i niende ledd").resolved


def test_sections_between_two_ends_of_a_range(resolver: Resolver) -> None:
    assert resolver.between(f"{LAW}/§1-1", f"{LAW}/§1-4") == [f"{LAW}/§1-2", f"{LAW}/§1-3"]
    assert resolver.between(f"{LAW}/§1-4", f"{LAW}/§1-1") == []
    assert resolver.between(f"{LAW}/§1-1", f"{OTHER}/§6-44") == []


def test_graph_edges_ranges_and_summary(resolver: Resolver) -> None:
    provisions = pd.DataFrame(
        [
            {"provision_id": f"{LAW}/§1-1/ledd/1", "section_id": f"{LAW}/§1-1", "doc_key": LAW,
             "text": "Se §§ 1-2 til 1-4 og skatteloven § 6-44, § 14-4 og første ledd."},
            {"provision_id": f"{LAW}/§1-2/ledd/1", "section_id": f"{LAW}/§1-2", "doc_key": LAW,
             "text": "Ingen henvisning."},
        ]
    )  # fmt: skip
    edges = build_edges(provisions, resolver)
    targets = list(edges["target_section_id"])
    assert targets == [
        f"{LAW}/§1-2",
        f"{LAW}/§1-3",
        f"{LAW}/§1-4",
        f"{OTHER}/§6-44",
        f"{OTHER}/§14-4",
        f"{LAW}/§1-1",
    ]
    assert list(edges["reason"]) == [
        "same_document", "inside_range", "same_document", "named", "named", "same_section"
    ]  # fmt: skip
    assert list(edges["scope"])[3:] == ["other_document", "other_document", "same_section"]
    summary = summarise(edges, "test")
    assert summary["references_to_sections"] == 4 and summary["resolved_share"] == 1.0
    assert summary["same_section_references"] == 1 and summary["edges_added_inside_ranges"] == 1

    links = pd.DataFrame(
        [
            {"source_provision_id": f"{LAW}/§1-1/ledd/1", "href": "lov/2001-02-03-4/§1-2"},
            {"source_provision_id": f"{LAW}/§1-1/ledd/1", "href": "lov/1999-03-26-14/§6-44/ledd/1"},
            {"source_provision_id": f"{LAW}/§1-1/ledd/1", "href": "lov/1999-03-26-14/§14-4"},
            {"source_provision_id": f"{LAW}/§1-1/ledd/1", "href": "lov/1999-03-26-14"},
            {"source_provision_id": f"{LAW}/§1-2/ledd/1", "href": "lov/2001-02-03-4/§1-1"},
        ]
    )
    agreement = compare_with_links(edges, links, tables()["sections"])
    assert agreement["pairs_in_lovdata_links"] == 4 and agreement["pairs_in_graph"] == 4
    assert agreement["pairs_in_both"] == 3
    assert agreement["recall_against_links"] == 0.75


def test_link_targets_and_name_forms() -> None:
    assert link_section_target("lov/1999-03-26-14/§6-13") == "lov/1999-03-26-14/§6-13"
    assert link_section_target("lov/1917-12-14-17/§10/ledd/1") == "lov/1917-12-14-17/§10"
    assert link_section_target("lov/1994-06-24-39/kap16") is None
    assert link_section_target("eu/31993l0013") is None
    assert name_forms("Skatteloven") == {"skatteloven", "skattelova"}
    assert name_forms("Prøveforskrifta") == {"prøveforskrifta", "prøveforskriften"}


def test_quality_checks_pass_on_clean_tables_and_flag_duplicates() -> None:
    clean = tables()
    assert all(check["passed"] for check in run_checks(clean))
    broken = tables()
    broken["provisions"] = pd.concat([broken["provisions"], broken["provisions"].head(1)])
    failed = [check["check"] for check in run_checks(broken) if not check["passed"]]
    assert failed == ["provision_ids_unique"]


def test_corpus_hash_depends_on_the_archive_hashes_only() -> None:
    first = corpus_hash({"a.tar.bz2": {"sha256": "11", "bytes": 1}, "b.tar.bz2": {"sha256": "22"}})
    same = corpus_hash({"b.tar.bz2": {"sha256": "22"}, "a.tar.bz2": {"sha256": "11", "bytes": 9}})
    other = corpus_hash({"a.tar.bz2": {"sha256": "11"}, "b.tar.bz2": {"sha256": "23"}})
    assert first == same and first != other and len(first) == 64
