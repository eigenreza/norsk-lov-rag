"""The reference parser on the citation forms Norwegian law uses."""

from __future__ import annotations

import pytest

from norsk_lov_rag.references import parser
from norsk_lov_rag.references.parser import find_references, normalise_number

EN_DASH = chr(0x2013)


def one(text: str):
    found = find_references(text)
    assert len(found) == 1, found
    return found[0]


def test_section_with_paragraph_and_letter() -> None:
    ref = one("Se § 6-44 første ledd bokstav a om reise.")
    assert (ref.section, ref.paragraph, ref.letter) == ("6-44", 1, "a")
    assert ref.act_kind == parser.ACT_NONE
    assert ref.raw == "§ 6-44 første ledd bokstav a"


@pytest.mark.parametrize(
    ("text", "section"),
    [
        ("jf. § 5-1", "5-1"),
        ("jf. § 3", "3"),
        ("etter § 3 a annet ledd", "3a"),
        ("etter § 12-2 a", "12-2a"),
        ("etter § 11 A-4", "11A-4"),
        ("etter § 6-44-2 i dag", "6-44-2"),
        ("etter § 4-11\nd. neste punkt", "4-11"),
    ],
)
def test_section_number_forms(text: str, section: str) -> None:
    assert one(text).section == section


def test_a_preposition_after_the_number_is_not_a_letter() -> None:
    ref = one("Reglene i § 3 i skatteloven gjelder.")
    assert ref.section == "3"
    assert (ref.act_kind, ref.act) == (parser.ACT_NAMED, "skatteloven")


@pytest.mark.parametrize(
    ("text", "kind", "act"),
    [
        ("etter skatteloven § 14-4", parser.ACT_NAMED, "skatteloven"),
        ("etter skattelovens § 14-4", parser.ACT_NAMED, "skatteloven"),
        ("etter skattelova § 14-4", parser.ACT_NAMED, "skattelova"),
        ("etter sktl. § 2-1", parser.ACT_ABBREVIATION, "sktl"),
        ("etter denne lov § 2-1", parser.ACT_SELF, None),
        ("etter loven her § 2-1", parser.ACT_SELF, None),
        ("etter denne forskrifta § 2", parser.ACT_SELF, None),
        ("etter § 2-1 i denne lov", parser.ACT_SELF, None),
        ("etter loven § 2-1", parser.ACT_THE_LAW, None),
        ("etter lova § 2-1", parser.ACT_THE_LAW, None),
        ("etter forskriften § 3", parser.ACT_THE_REGULATION, None),
        ("jf. § 5-1", parser.ACT_NONE, None),
        ("mv. § 5-1", parser.ACT_NONE, None),
    ],
)
def test_how_the_act_is_named(text: str, kind: str, act: str | None) -> None:
    ref = one(text)
    assert (ref.act_kind, ref.act) == (kind, act)


def test_full_citation_gives_the_document_key() -> None:
    ref = one("lov 26. mars 1999 nr. 14 om skatt av formue og inntekt (skatteloven) § 6-44")
    assert ref.act_kind == parser.ACT_CITATION
    assert ref.act_key == "lov/1999-03-26-14" and ref.section == "6-44"
    regulation = one("forskrift 19. november 1999 nr. 1158 § 5-15-1")
    assert regulation.act_key == "forskrift/1999-11-19-1158" and regulation.section == "5-15-1"


def test_list_of_sections() -> None:
    found = find_references("etter §§ 5-10, 5-11 og 5-12 annet ledd")
    assert [r.section for r in found] == ["5-10", "5-11", "5-12"]
    assert [r.paragraph for r in found] == [None, None, 2]
    assert not any(r.range_end for r in found)


@pytest.mark.parametrize("text", ["§§ 5-10 til 5-15", f"§§ 5-10{EN_DASH}5-15", "§ 5-10 til § 5-15"])
def test_range_of_sections(text: str) -> None:
    found = find_references(text)
    assert [r.section for r in found] == ["5-10", "5-15"]
    assert [r.range_end for r in found] == [False, True]


def test_act_name_carries_over_to_the_next_sections() -> None:
    found = find_references("omfattes av skatteloven § 2-3, § 14-81 og § 14-82, men ikke § 9.")
    assert [(r.section, r.act) for r in found[:3]] == [
        ("2-3", "skatteloven"),
        ("14-81", "skatteloven"),
        ("14-82", "skatteloven"),
    ]
    assert found[3].act_kind == parser.ACT_NONE
    carried = find_references("etter vareførselsforskriften § 5-9-7, jf. § 5-9-10")
    assert [r.act for r in carried] == ["vareførselsforskriften", "vareførselsforskriften"]


def test_several_paragraphs_of_one_section() -> None:
    found = find_references("§ 3-4 første til tredje ledd")
    assert [(r.section, r.paragraph) for r in found] == [("3-4", 1), ("3-4", 2), ("3-4", 3)]
    both = find_references("§ 3-4 første og annet ledd")
    assert [r.paragraph for r in both] == [1, 2]


def test_paragraph_reference_without_a_section_points_inside_the_same_section() -> None:
    found = find_references(
        "Beløpsgrensene i første ledd gjelder tilsvarende, jf. annet ledd bokstav b."
    )
    assert [(r.act_kind, r.paragraph, r.letter) for r in found] == [
        (parser.ACT_SAME_SECTION, 1, None),
        (parser.ACT_SAME_SECTION, 2, "b"),
    ]


def test_nynorsk_ordinals() -> None:
    ref = one("etter lova § 3 fyrste ledd")
    assert (ref.act_kind, ref.section, ref.paragraph) == (parser.ACT_THE_LAW, "3", 1)
    assert one("etter § 7 sjuande ledd").paragraph == 7


def test_number_item_and_sentence_qualifiers() -> None:
    ref = one("§ 5-1 første ledd annet punktum nr. 3")
    assert (ref.section, ref.paragraph, ref.number) == ("5-1", 1, 3)


def test_text_without_references() -> None:
    assert find_references("Ingen henvisninger her, bare et beløp på 100 kroner.") == []
    assert find_references("Tegnet § alene gir ingen henvisning.") == []


def test_normalise_number() -> None:
    assert normalise_number("6 - 44") == "6-44" and normalise_number("3 a") == "3a"
