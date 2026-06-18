"""Read a generator's output and run the checks that need no model.

The generator writes plain text with a citation in brackets after each sentence. This module
builds the structured answer from it: the text, the list of citations read from the text, and a
flag that is true when the text is the fixed refusal sentence. The record is checked against a
schema. Output with chat tokens or other debris is marked as invalid, so that format failures
are counted rather than hidden.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ValidationError

from norsk_lov_rag.corpus.language import marker_counts
from norsk_lov_rag.generation.prompts import ABSTAIN
from norsk_lov_rag.generation.sources import CITATION, SourceBook, find_citations

FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-ZÆØÅ0-9«(\[])")


def _plain(text: str) -> str:
    return " ".join(text.lower().replace("«", "").replace("»", "").split()).strip(" .")


ABSTAIN_PLAIN = {_plain(phrase) for phrase in ABSTAIN.values()}


class AnswerRecord(BaseModel):
    """The structured answer: text, citations and whether the model refused to answer."""

    answer: str
    citations: list[str]
    abstain: bool


DEBRIS = re.compile(r"<\|?im_(?:start|end)\|?>|<\|[a-z_]+\|>|^\s*[{\[]")


def parse_output(text: str) -> dict:
    """The structured answer built from the text, and whether the output was clean."""
    answer = FENCE.sub("", text.strip()).strip()
    plain = _plain(answer)
    record = {
        "answer": answer,
        "citations": list(dict.fromkeys(find_citations(answer))),
        "abstain": any(phrase in plain for phrase in ABSTAIN_PLAIN),
    }
    try:
        AnswerRecord.model_validate(record)
        valid = bool(answer) and not DEBRIS.search(answer)
    except ValidationError:
        valid = False
    return {**record, "valid_output": valid}


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(text.strip()) if s.strip()]


def is_abstention(parsed: dict) -> bool:
    return parsed["abstain"]


def language_check(text: str, wanted: str) -> str:
    """'match', 'mismatch' or 'unclear', with the same marker words as the test-set filter."""
    bokmal, nynorsk = marker_counts(CITATION.sub(" ", text))
    if bokmal == nynorsk:
        return "unclear"
    found = "nn" if nynorsk > bokmal else "nb"
    return "match" if found == wanted else "mismatch"


def analyse(parsed: dict, question: dict, shown: list[str], book: SourceBook) -> dict:
    """Citation checks, the answer contract and the written standard for one answer."""
    abstained = is_abstention(parsed)
    text = parsed["answer"]
    in_text = find_citations(text)
    cited = list(dict.fromkeys(in_text + parsed["citations"]))
    resolved = {c: book.resolve(c) for c in cited}
    claims: dict[str, list[str]] = {c: [] for c in cited}
    parts = sentences(text) if not abstained else []
    with_citation = 0
    for sentence in parts:
        found = find_citations(sentence)
        if found:
            with_citation += 1
        for c in found:
            claims.setdefault(c, []).append(sentence)
    units = [u for u in dict.fromkeys(resolved.values()) if u is not None]
    gold = set(question.get("gold_section_ids") or [])
    return {
        "abstained": abstained,
        "citations": [
            {
                "citation": c,
                "unit": resolved[c],
                "in_sources": resolved[c] in shown,
                "in_text": c in in_text,
                "claims": claims.get(c, []),
            }
            for c in cited
        ],
        "cited_units": units,
        "cited_gold": sorted(gold.intersection(units)),
        "sentences": len(parts),
        "sentences_with_citation": with_citation,
        "language": _language(text, question["written_standard"]),
    }


def _language(text: str, wanted: str) -> str:
    """The fixed refusal sentence counts by which of the two versions was used."""
    for standard, phrase in ABSTAIN.items():
        if _plain(text) == _plain(phrase):
            return "match" if standard == wanted else "mismatch"
    return language_check(text, wanted)
