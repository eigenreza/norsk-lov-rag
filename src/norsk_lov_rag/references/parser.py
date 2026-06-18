"""Find references to legal provisions in Norwegian text.

The parser reads the forms Norwegian law uses: ``§ 6-44 første ledd bokstav a``, ``jf. § 5-1``,
``§§ 5-10 til 5-15``, ``denne lov § 2-1``, ``forskriften § 3``, ``skatteloven § 14-4`` and
``lov 26. mars 1999 nr. 14 om skatt av formue og inntekt (skatteloven) § 6-44``. It only finds
and describes the references. Pointing them at provisions is the job of the resolver.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

EN_DASH = chr(0x2013)
ORDINALS = {
    "første": 1,
    "fyrste": 1,
    "annet": 2,
    "andre": 2,
    "anna": 2,
    "tredje": 3,
    "fjerde": 4,
    "femte": 5,
    "sjette": 6,
    "sjuende": 7,
    "syvende": 7,
    "sjuande": 7,
    "åttende": 8,
    "åttande": 8,
    "niende": 9,
    "niande": 9,
    "tiende": 10,
    "tiande": 10,
    "ellevte": 11,
    "tolvte": 12,
}
MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "mai": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "okt": 10,
    "nov": 11,
    "des": 12,
}
ACT_SELF = "self"
ACT_NAMED = "named"
ACT_ABBREVIATION = "abbreviation"
ACT_CITATION = "citation"
ACT_THE_LAW = "the_law"
ACT_THE_REGULATION = "the_regulation"
ACT_NONE = "none"
ACT_SAME_SECTION = "same_section"

# A letter after the number belongs to it ("§ 3 a"), but "i" is the preposition ("§ 3 i loven").
_LETTER = r"(?:[ ]?[a-hj-zA-HJ-Z](?![a-zA-ZæøåÆØÅ]))?"
_PART = rf"\d+{_LETTER}"
# Acts mostly number sections "6-44". Regulations often add a third part, as in "6-44-2".
_NUMBER = rf"{_PART}(?:[ ]?-[ ]?{_PART})*"
_ORDINAL = "|".join(sorted(ORDINALS, key=len, reverse=True))
_ORDINALS = rf"(?:{_ORDINAL})(?:\s*(?:,|og|eller|til)\s*(?:{_ORDINAL}))*"
_JOIN = rf"\s*(?:,|\bog\b|\beller\b|\btil\b|{EN_DASH})\s*(?:§\s*)?"

SECTION_LIST = re.compile(rf"§§\s*(?P<numbers>{_NUMBER}(?:{_JOIN}{_NUMBER})*)")
SECTION_ONE = re.compile(rf"§\s*(?P<numbers>{_NUMBER})")
NUMBER = re.compile(_NUMBER)
QUALIFIER = re.compile(
    rf"(?:\s*,?\s*(?P<ordinals>{_ORDINALS})\s+ledd(?:et|a|ene)?)?"
    rf"(?:\s*,?\s*(?:{_ORDINALS})\s+punktum)?"
    rf"(?:\s*,?\s*bokstav(?:ene)?\s+(?P<letter>[a-zæøå])(?![a-zæøå]))?"
    rf"(?:\s*,?\s*nr\.?\s*(?P<number>\d+))?",
    re.IGNORECASE,
)
BARE_PARAGRAPH = re.compile(
    rf"\b(?P<ordinals>{_ORDINALS})\s+ledd(?:et|a|ene)?"
    rf"(?:\s*,?\s*(?:{_ORDINALS})\s+punktum)?"
    rf"(?:\s*,?\s*bokstav(?:ene)?\s+(?P<letter>[a-zæøå])(?![a-zæøå]))?",
    re.IGNORECASE,
)
_ACT_WORD = r"(?<![\w-])[\w-]*(?:loven|lova|lovens|forskriften|forskrifta|forskriftens)"
PREFIX_SELF = re.compile(
    r"(?:denne\s+(?:lov(?:en|a)?|forskrift(?:en|a)?)|(?:loven|lova|forskriften|forskrifta)\s+her)\s*$",
    re.IGNORECASE,
)
PREFIX_NAMED = re.compile(rf"(?P<name>{_ACT_WORD})\s*$")
PREFIX_ABBREVIATION = re.compile(r"(?<![a-zæøåA-ZÆØÅ])(?P<abbreviation>[a-zæøå]{2,12})\.\s*$")
PREFIX_CITATION = re.compile(
    r"(?P<kind>lov|forskrift)\s+(?:av\s+)?(?P<day>\d{1,2})\.?\s+(?P<month>[a-zæøå]{3})[a-zæøå]*\.?\s+"
    r"(?P<year>\d{4})\s+nr\.?\s*(?P<number>\d+)(?P<gap>[^§;]{0,110})$",
    re.IGNORECASE,
)
SUFFIX_ACT = re.compile(
    rf"\s+i\s+(?P<act>denne\s+(?:lov(?:en|a)?|forskrift(?:en|a)?)|{_ACT_WORD})", re.IGNORECASE
)
RANGE_TAIL = re.compile(rf"\s+til\s+§\s*(?P<last>{_NUMBER})")
# What may stand between two references that share the act named before the first one,
# as in "skatteloven § 2-3, § 14-81 og § 14-82" or "tolloven § 5-9, jf. § 5-10".
CONNECTOR = re.compile(r"\s*(?:,|og|eller|samt|til)?\s*(?:jfr?\.?)?\s*", re.IGNORECASE)
# Short words that end with a full stop before a section sign and are not names of acts.
NOT_ACTS = frozenset(
    "jf jfr cf se mv bl evt nr pkt kap art ev dvs osv mfl ca pr fl ff og eller etter".split()
)
PREFIX_WINDOW = 160
THE_LAW = {"loven", "lova", "lovens"}
THE_REGULATION = {"forskriften", "forskrifta", "forskriftens"}


@dataclass(frozen=True)
class Reference:
    """One reference to one section, or to a paragraph of the section it stands in."""

    start: int
    end: int
    raw: str
    act_kind: str
    act: str | None = None
    act_key: str | None = None
    section: str | None = None
    paragraph: int | None = None
    letter: str | None = None
    number: int | None = None
    range_end: bool = False


def normalise_number(number: str) -> str:
    """``6 - 44`` becomes ``6-44`` and ``3 a`` becomes ``3a``."""
    return re.sub(r"\s+", "", number)


def _ordinal_numbers(text: str | None) -> list[int | None]:
    """The paragraph numbers in ``første og annet`` or ``første til tredje``."""
    if not text:
        return [None]
    words = re.findall(_ORDINAL, text.lower())
    numbers = [ORDINALS[word] for word in words]
    if re.search(r"\btil\b", text.lower()) and len(numbers) == 2 and numbers[0] < numbers[1]:
        return list(range(numbers[0], numbers[1] + 1))
    return numbers or [None]


def _act_before(text: str, start: int) -> tuple[str, str | None, str | None]:
    """What the text right before a section sign says about the act: kind, name and key."""
    before = text[max(0, start - PREFIX_WINDOW) : start]
    if PREFIX_SELF.search(before):
        return ACT_SELF, None, None
    named = PREFIX_NAMED.search(before)
    if named:
        return _named(named.group("name"))
    abbreviation = PREFIX_ABBREVIATION.search(before)
    citation = PREFIX_CITATION.search(before)
    if citation and not abbreviation:
        month = MONTHS.get(citation.group("month").lower())
        if month:
            key = (
                f"{citation.group('kind').lower()}/{citation.group('year')}-{month:02d}-"
                f"{int(citation.group('day')):02d}-{int(citation.group('number'))}"
            )
            return ACT_CITATION, None, key
    if abbreviation:
        word = abbreviation.group("abbreviation").lower()
        if word not in NOT_ACTS:
            return ACT_ABBREVIATION, word, None
    return ACT_NONE, None, None


def _named(name: str) -> tuple[str, str | None, str | None]:
    name = name.lower()
    if name in THE_LAW:
        return ACT_THE_LAW, None, None
    if name in THE_REGULATION:
        return ACT_THE_REGULATION, None, None
    if name.endswith("ens"):
        name = name[:-1]
    return ACT_NAMED, name, None


def find_references(text: str) -> list[Reference]:
    """Every reference in the text, in reading order."""
    found: list[Reference] = []
    taken: list[tuple[int, int]] = []
    carried: tuple[str, str | None, str | None] | None = None
    carried_end = 0
    position = 0
    while True:
        sign = text.find("§", position)
        if sign < 0:
            break
        match = SECTION_LIST.match(text, sign) or SECTION_ONE.match(text, sign)
        if not match:
            position = sign + 1
            continue
        act_kind, act, act_key = _act_before(text, sign)
        own_act = act_kind != ACT_NONE
        if not own_act and carried and CONNECTOR.fullmatch(text[carried_end:sign]):
            act_kind, act, act_key = carried
        qualifier = QUALIFIER.match(text, match.end())
        end = qualifier.end()
        numbers = [number.group() for number in NUMBER.finditer(match.group("numbers"))]
        qualifiers = [None] * (len(numbers) - 1) + [qualifier]
        is_range = bool(
            re.search(rf"\btil\b|{EN_DASH}", match.group("numbers")) and len(numbers) == 2
        )
        tail = RANGE_TAIL.match(text, end) if len(numbers) == 1 else None
        if tail:
            # "§ 7-10-23 til § 7-10-25": the second section closes a range.
            numbers.append(tail.group("last"))
            qualifiers.append(QUALIFIER.match(text, tail.end()))
            end = qualifiers[-1].end()
            is_range = True
        suffix = SUFFIX_ACT.match(text, end)
        if suffix and act_kind == ACT_NONE:
            words = suffix.group("act").lower()
            if words.startswith("denne"):
                act_kind = ACT_SELF
            else:
                act_kind, act, act_key = _named(words)
            end = suffix.end()
        raw = text[sign:end]
        for index, number in enumerate(numbers):
            details = qualifiers[index]
            last = index == len(numbers) - 1
            for paragraph in _ordinal_numbers(details.group("ordinals") if details else None):
                found.append(
                    Reference(
                        start=sign,
                        end=end,
                        raw=raw,
                        act_kind=act_kind,
                        act=act,
                        act_key=act_key,
                        section=normalise_number(number),
                        paragraph=paragraph,
                        letter=((details.group("letter") or "").lower() or None)
                        if details
                        else None,
                        number=int(details.group("number"))
                        if details and details.group("number")
                        else None,
                        range_end=is_range and last,
                    )
                )
        taken.append((sign, end))
        carried = (act_kind, act, act_key) if act_kind != ACT_NONE else None
        carried_end = end
        position = max(end, sign + 1)
    for match in BARE_PARAGRAPH.finditer(text):
        if any(start <= match.start() < end for start, end in taken):
            continue
        for paragraph in _ordinal_numbers(match.group("ordinals")):
            found.append(
                Reference(
                    start=match.start(),
                    end=match.end(),
                    raw=match.group(),
                    act_kind=ACT_SAME_SECTION,
                    paragraph=paragraph,
                    letter=(match.group("letter") or "").lower() or None,
                )
            )
    return sorted(found, key=lambda reference: (reference.start, reference.section or ""))
