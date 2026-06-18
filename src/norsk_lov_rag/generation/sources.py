"""Sources shown to the generators, and the citation strings that point back to them.

Each source is shown with a fixed label such as ``skatteloven § 6-44``. The answer cites a source
by writing that label in square brackets. Normalising a citation means reading the act name and
the section number from the string and finding the section id, whatever the spacing, dash or
case. A citation that names no known section is kept as unresolved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from norsk_lov_rag.chunking.build import chunk_path

CITATION = re.compile(r"\[([^\[\]]{2,120})\]")
DASHES = re.compile("[\u2010\u2011\u2012\u2013\u2014\u2212]")
SECTION_NUMBER = re.compile(r"^\s*(\d+[a-z]?(?:\s*-\s*\d+[a-z]?)*)(?:\s+([a-z])(?![a-zæøå]))?")
LOOSE_PART = re.compile(r"\bdel\s+(\d+)\b")


def citation_name(short_name: str) -> str:
    """The act name used in labels: an abbreviation such as FSFIN as it is, a name in lower case."""
    first = short_name.split()[0] if short_name.strip() else ""
    if first.isupper() and len(first) > 1:
        return first
    return short_name.strip().lower()


NYNORSK_ENDINGS = (("lova", "loven"), ("forskrifta", "forskriften"))


def normalise_name(text: str) -> str:
    """Lower case, no punctuation, and the nynorsk form of an act name read as bokmål."""
    words = text.lower().replace(",", " ").replace(".", " ").split()
    for number, word in enumerate(words):
        for nynorsk, bokmal in NYNORSK_ENDINGS:
            if word.endswith(nynorsk):
                words[number] = word[: -len(nynorsk)] + bokmal
    return " ".join(words)


def normalise_number(text: str) -> str | None:
    """'§ 6-44 første ledd' and '§6 - 44' both give '6-44'; '§ 14 g' gives '14g'."""
    text = DASHES.sub("-", text.lower()).replace("§", " ")
    match = SECTION_NUMBER.match(text)
    if not match:
        return None
    return re.sub(r"\s+", "", match.group(1)) + (match.group(2) or "")


@dataclass
class Source:
    unit: str
    label: str
    title: str
    text: str


class SourceBook:
    """Labels and texts of every section in one corpus, and the way back from a label."""

    def __init__(self, data_dir: Path, processed: Path, scope: str) -> None:
        chunks = pd.read_parquet(chunk_path(data_dir, scope, "structure"))
        documents = pd.read_parquet(processed / "documents.parquet")
        names = {
            row.doc_key: (row.short_name or row.title or "").strip()
            for row in documents.itertuples(index=False)
        }
        sections = pd.read_parquet(
            processed / "sections.parquet", columns=["section_id", "section_label", "section_title"]
        ).set_index("section_id")
        self.texts: dict[str, list[str]] = {}
        self.labels: dict[str, str] = {}
        self.titles: dict[str, str] = {}
        self._lookup: dict[tuple[str, str], str] = {}
        for chunk in chunks.itertuples(index=False):
            unit = chunk.units[0]
            body = chunk.text.split("\n", 1)[1] if "\n" in chunk.text else chunk.text
            self.texts.setdefault(unit, []).append(body)
            if unit in self.labels:
                continue
            short = names.get(chunk.doc_key, chunk.doc_key)
            name = citation_name(short)
            if unit in sections.index:
                label_text = _text(sections.at[unit, "section_label"])
                self.labels[unit] = f"{name} {label_text}".strip()
                self.titles[unit] = _text(sections.at[unit, "section_title"])
                number = normalise_number(label_text)
            else:
                part = int(unit.rsplit("#", 1)[1]) + 1
                self.labels[unit] = f"{name} del {part}"
                self.titles[unit] = ""
                number = f"del{part}"
            if number is None:
                continue
            for alias in {name, short, short.split(" ", 1)[-1]}:
                self._lookup.setdefault((normalise_name(alias), number), unit)

    def text(self, unit: str) -> str:
        return "\n".join(self.texts.get(unit, []))

    def resolve(self, citation: str) -> str | None:
        """The section id a citation string points to, or None."""
        lowered = citation.lower()
        loose = LOOSE_PART.search(lowered)
        if "§" in lowered:
            name, rest = lowered.split("§", 1)
            number = normalise_number(rest)
        elif loose:
            name, number = lowered[: loose.start()], f"del{loose.group(1)}"
        else:
            return None
        if number is None:
            return None
        return self._lookup.get((normalise_name(name), number))

    def sources(
        self, units: list[str], top_k: int, max_words_per_source: int, max_context_words: int
    ) -> list[Source]:
        """The first ``top_k`` units as sources, cut to fit the word budgets."""
        out: list[Source] = []
        left = max_context_words
        for unit in units[:top_k]:
            words = self.text(unit).split()
            take = min(len(words), max_words_per_source)
            if take > left:
                # Cut the last source to what is left, unless too little is left to be useful.
                if left < min(50, take):
                    break
                take = left
            out.append(Source(unit, self.labels[unit], self.titles[unit], " ".join(words[:take])))
            left -= take
        return out


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


MENTION = re.compile(
    r"\b([A-ZÆØÅa-zæøå][\wæøåÆØÅ-]*(?:loven|lova|forskriften|forskrifta)|FSFIN|FSSD)"
    r"\s*§+\s*(\d+[a-z]?(?:\s*[-\u2010-\u2013]\s*\d+[a-z]?)*)"
)


def find_mentions(text: str) -> list[str]:
    """Every 'act § number' in the text, with or without brackets, for the lenient count."""
    return [f"{m.group(1)} § {m.group(2)}" for m in MENTION.finditer(text)]


def find_citations(text: str) -> list[str]:
    return [match.group(1).strip() for match in CITATION.finditer(text)]
