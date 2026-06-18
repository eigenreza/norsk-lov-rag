"""Pick the provisions the test questions are written from.

Provisions come from the tax corpus, stratified by document and chapter. Each document gets a
share in proportion to the square root of its number of usable paragraphs, so that small
documents are not left out and large ones do not take over. Inside a document, chapters take
turns, and no section is used twice. Cross-reference tasks are drawn from the reference graph.
Unanswerable questions are written from fixed topics, not from provisions.
"""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict

import pandas as pd
from pydantic import BaseModel

SEED = 20
MIN_CHARS = 150
MAX_CHARS = 2500
# Longest text shown around a target paragraph. Longer sections are cut to nearby paragraphs.
MAX_CONTEXT_CHARS = 4000
MAX_TARGET_SECTION_CHARS = 3000
SKIPPED_TITLES = re.compile(
    r"ikrafttred|overgangs|endring(?:er)? i andre lover|oppheving|opphevelse", re.IGNORECASE
)
# Cross-reference tasks use only links the resolver found directly. Fallbacks, such as reading
# a section number as one of the parent act, are right most of the time but not always.
TRUSTED_REASONS = ("same_document", "named", "citation")


class Counts(BaseModel):
    """How many tasks to write of each kind. Some are lost in the filters and the review."""

    direct_nb: int = 130
    everyday_nb: int = 130
    direct_nn: int = 30
    everyday_nn: int = 30
    cross_reference_nb: int = 150


# Topics for unanswerable questions: kind, written standard, topics. Each group is one call.
UNANSWERABLE_TOPICS: list[tuple[str, str, list[str]]] = [
    ("annual_rate", "nb", [
        "skattesatsen på alminnelig inntekt for personer",
        "innslagspunktet for første trinn i trinnskatten",
        "den høyeste satsen i trinnskatten",
        "størrelsen på personfradraget",
        "satsen og den øvre grensen for minstefradraget i lønn",
    ]),
    ("annual_rate", "nb", [
        "satsen for formuesskatt til staten",
        "bunnfradraget i formuesskatten",
        "satsen for trygdeavgift på lønnsinntekt",
        "satsen for arbeidsgiveravgift i sone 1",
        "den alminnelige satsen for merverdiavgift",
    ]),
    ("annual_rate", "nb", [
        "satsen for merverdiavgift på matvarer",
        "beløpsgrensen for frikort",
        "det høyeste foreldrefradraget for ett barn",
        "egenandelen i reisefradraget",
        "det høyeste fradraget for gaver til frivillige organisasjoner",
    ]),
    ("annual_rate", "nb", [
        "oppjusteringsfaktoren for utbytte til personer",
        "kilometersatsen for skattefri bilgodtgjørelse",
        "det høyeste årlige sparebeløpet i BSU",
        "satsen for finansskatt på lønn",
        "normrenten for lån fra arbeidsgiver",
    ]),
    ("administrative", "nb", [
        "åpningstidene på skattekontoret",
        "telefonnummeret til Skatteetaten",
        "hvor lang tid det tar å behandle en klage på skatteoppgjøret",
        "hvilken dag skatteoppgjøret kommer i år",
        "hvordan man logger inn på Skatteetatens nettsider",
    ]),
    ("administrative", "nb", [
        "hvor man finner skjemaet for å endre skattekortet",
        "om Skatteetaten har chat på nettsidene",
        "hvor lang ventetiden er for å få D-nummer",
        "hvor skattekontoret ligger i Bergen",
        "hvor mange som fikk restskatt i fjor",
    ]),
    ("foreign", "nb", [
        "skattesatsen på lønn i Sverige",
        "formuesskatt i Danmark",
        "merverdiavgift i Tyskland",
        "arveavgift i Frankrike",
        "fradrag for boliglån i Finland",
    ]),
    ("foreign", "nb", [
        "skatt på kryptovaluta i USA",
        "bilavgifter i Nederland",
        "selskapsskatten i Irland",
        "skatt på utbytte i Storbritannia",
        "skatt på pensjon for spanske borgere i Spania",
    ]),
    ("outside_tax_law", "nb", [
        "oppsigelsestiden for ansatte",
        "hvor stort depositum en utleier kan kreve",
        "fartsgrensen i tettbygd strøk",
        "hvor lang foreldrepermisjon man har",
        "aldersgrensen for å stemme ved stortingsvalg",
    ]),
    ("outside_tax_law", "nb", [
        "hvor lenge man må være separert før skilsmisse",
        "angrerett ved kjøp på nett",
        "reklamasjonsfristen for varer",
        "jakttiden for elg",
        "båndtvang for hunder",
    ]),
    ("outside_tax_law", "nb", [
        "hvor gammel man må være for å ta førerkort for bil",
        "hvem som arver når det ikke finnes testament",
        "når man kan få fri rettshjelp",
        "hvor lang prøvetid en arbeidsgiver kan avtale",
        "den lengste lovlige arbeidstiden per uke",
    ]),
    ("annual_rate", "nn", [
        "satsen for formuesskatt til kommunen",
        "satsen for trygdeavgift på næringsinntekt",
        "minstefrådraget for pensjonsinntekt",
        "satsen for meirverdiavgift på persontransport",
    ]),
    ("administrative", "nn", [
        "kor lang tid det tek å få skattekortet",
        "når Skatteetaten sender ut skattemeldinga",
    ]),
    ("foreign", "nn", ["skattesatsen for selskap i Sverige", "meirverdiavgifta i Danmark"]),
    ("outside_tax_law", "nn", [
        "kor mange feriedagar ein har krav på",
        "kor gammal ein må vere for å gifte seg",
    ]),
]  # fmt: skip

# Unanswerable questions outside tax law can be answered from the full corpus, so they count as
# unanswerable for the tax corpus only.
UNANSWERABLE_IN = {
    "annual_rate": ["tax", "full"],
    "administrative": ["tax", "full"],
    "foreign": ["tax", "full"],
    "outside_tax_law": ["tax"],
}


def allocate(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Share ``total`` in proportion to the square root of each size, with largest remainders."""
    weights = {key: math.sqrt(size) for key, size in sizes.items() if size > 0}
    scale = total / sum(weights.values())
    raw = {key: weight * scale for key, weight in weights.items()}
    shares = {key: min(sizes[key], int(value)) for key, value in raw.items()}
    order = sorted(raw, key=lambda key: raw[key] - int(raw[key]), reverse=True)
    while sum(shares.values()) < total:
        grown = False
        for key in order:
            if sum(shares.values()) >= total:
                break
            if shares[key] < sizes[key]:
                shares[key] += 1
                grown = True
        if not grown:
            break
    return shares


def usable_provisions(provisions: pd.DataFrame, sections: pd.DataFrame) -> pd.DataFrame:
    """Paragraphs in sections in force, of a useful length, outside transition rules."""
    titles = sections.set_index("section_id")["section_title"].fillna("")
    keep = (
        provisions["section_id"].notna()
        & provisions["text"].str.len().between(MIN_CHARS, MAX_CHARS)
        & provisions["section_id"].isin(titles.index)
    )
    usable = provisions[keep].copy()
    usable = usable[~usable["section_id"].map(titles).str.contains(SKIPPED_TITLES)]
    return usable


def section_parts(
    section_id: str, target_id: str, by_section: dict[str, pd.DataFrame]
) -> list[dict]:
    """The paragraphs of a section around the target, numbered from 1, within the length cap."""
    rows = by_section[section_id]
    texts = list(zip(rows["provision_id"], rows["text"], strict=True))
    index = next(i for i, (pid, _) in enumerate(texts) if pid == target_id)
    lo = hi = index
    length = len(texts[index][1])
    while True:
        grown = False
        for candidate in (lo - 1, hi + 1):
            if 0 <= candidate < len(texts):
                extra = len(texts[candidate][1])
                if length + extra <= MAX_CONTEXT_CHARS:
                    length += extra
                    lo, hi = min(lo, candidate), max(hi, candidate)
                    grown = True
        if not grown:
            break
    return [
        {"number": n, "provision_id": pid, "section_id": section_id, "text": text,
         "target": pid == target_id}
        for n, (pid, text) in enumerate(texts[lo : hi + 1], start=1)
    ]  # fmt: skip


def _heading(section: pd.Series, document: pd.Series) -> dict:
    return {
        "doc_key": section["doc_key"],
        "document": document["short_title"] or document["title"],
        "chapter": section["chapter_path"] or "",
        "section_label": section["section_label"] or "",
        "section_title": section["section_title"] or "",
    }


def _draw_by_chapter(
    candidates: pd.DataFrame, count: int, rng: random.Random, used_sections: set[str]
) -> list[pd.Series]:
    """Take ``count`` rows, letting chapters take turns and using each section once."""
    by_chapter: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for position, row in enumerate(candidates.itertuples(index=False)):
        if row.section_id not in used_sections:
            by_chapter[row.chapter_path or ""][row.section_id].append(position)
    chapters = sorted(by_chapter)
    rng.shuffle(chapters)
    picked: list[pd.Series] = []
    while len(picked) < count and chapters:
        for chapter in list(chapters):
            if len(picked) >= count:
                break
            open_sections = [s for s in sorted(by_chapter[chapter]) if s not in used_sections]
            if not open_sections:
                chapters.remove(chapter)
                continue
            section_id = rng.choice(open_sections)
            position = rng.choice(by_chapter[chapter][section_id])
            used_sections.add(section_id)
            picked.append(candidates.iloc[position])
    return picked


def single_tasks(
    tables: dict[str, pd.DataFrame], counts: Counts, used_sections: set[str]
) -> list[dict]:
    """Direct and everyday tasks, in bokmål and nynorsk, one paragraph each."""
    rng = random.Random(SEED)
    documents = tables["documents"].set_index("doc_key")
    sections = tables["sections"].set_index("section_id")
    provisions = tables["provisions"].sort_values(["doc_key", "order"])
    usable = usable_provisions(provisions, tables["sections"])
    by_section = {key: rows for key, rows in provisions.groupby("section_id", sort=False)}
    labels = (
        [("direct", "nb")] * counts.direct_nb
        + [("everyday", "nb")] * counts.everyday_nb
        + [("direct", "nn")] * counts.direct_nn
        + [("everyday", "nn")] * counts.everyday_nn
    )
    shares = allocate(usable.groupby("doc_key").size().to_dict(), len(labels))
    drawn: list[pd.Series] = []
    for doc_key in sorted(shares):
        drawn += _draw_by_chapter(
            usable[usable["doc_key"] == doc_key], shares[doc_key], rng, used_sections
        )
    rng.shuffle(drawn)
    rng.shuffle(labels)
    tasks = []
    for row, (question_type, standard) in zip(drawn, labels, strict=False):
        section = sections.loc[row["section_id"]]
        tasks.append(
            {
                "question_type": question_type,
                "written_standard": standard,
                **_heading(section, documents.loc[section["doc_key"]]),
                "section_id": row["section_id"],
                "target_provision_id": row["provision_id"],
                "parts": section_parts(row["section_id"], row["provision_id"], by_section),
            }
        )
    return tasks


def cross_reference_tasks(
    tables: dict[str, pd.DataFrame],
    references: pd.DataFrame,
    count: int,
    used_sections: set[str],
) -> list[dict]:
    """Tasks that need a paragraph and the provision it refers to."""
    rng = random.Random(SEED + 1)
    documents = tables["documents"].set_index("doc_key")
    sections = tables["sections"].set_index("section_id")
    provisions = tables["provisions"].sort_values(["doc_key", "order"])
    usable = usable_provisions(provisions, tables["sections"]).set_index("provision_id")
    text_of = provisions.set_index("provision_id")["text"]
    by_section = {key: rows for key, rows in provisions.groupby("section_id", sort=False)}
    edges = references[
        references["resolved"]
        & (references["scope"] != "same_section")
        & references["reason"].isin(TRUSTED_REASONS)
        & references["source_provision_id"].isin(usable.index)
        & references["target_section_id"].isin(sections.index)
        & (references["target_section_id"] != references["source_section_id"])
    ].copy()
    section_length = sections["text"].str.len()
    whole_section = edges["target_provision_id"].isna()
    too_long = whole_section & (
        edges["target_section_id"].map(section_length) > MAX_TARGET_SECTION_CHARS
    )
    missing = ~whole_section & ~edges["target_provision_id"].isin(text_of.index)
    edges = edges[~too_long & ~missing]
    edges = edges.drop_duplicates(["source_provision_id", "target_section_id"])
    edges = edges.rename(columns={"source_section_id": "section_id"})
    edges["chapter_path"] = edges["section_id"].map(sections["chapter_path"])
    shares = allocate(edges.groupby("source_doc_key").size().to_dict(), count)
    drawn: list[pd.Series] = []
    for doc_key in sorted(shares):
        drawn += _draw_by_chapter(
            edges[edges["source_doc_key"] == doc_key], shares[doc_key], rng, used_sections
        )
    rng.shuffle(drawn)
    tasks = []
    for edge in drawn:
        section = sections.loc[edge["section_id"]]
        target_section = sections.loc[edge["target_section_id"]]
        source_id = edge["source_provision_id"]
        parts = [{"number": 1, "provision_id": source_id, "section_id": edge["section_id"],
                  "text": text_of[source_id], "target": True, "side": "A"}]  # fmt: skip
        if isinstance(edge["target_provision_id"], str):
            target_ids = [edge["target_provision_id"]]
        else:
            target_ids = list(by_section[edge["target_section_id"]]["provision_id"])
        for number, pid in enumerate(target_ids, start=2):
            parts.append({"number": number, "provision_id": pid,
                          "section_id": edge["target_section_id"], "text": text_of[pid],
                          "target": False, "side": "B"})  # fmt: skip
        tasks.append(
            {
                "question_type": "cross_reference",
                "written_standard": "nb",
                **_heading(section, documents.loc[section["doc_key"]]),
                "section_id": edge["section_id"],
                "target_provision_id": source_id,
                "reference": edge["raw"],
                "b_heading": _heading(target_section, documents.loc[target_section["doc_key"]]),
                "parts": parts,
            }
        )
    return tasks


def unanswerable_tasks() -> list[dict]:
    return [
        {
            "question_type": "unanswerable",
            "written_standard": standard,
            "kind": kind,
            "unanswerable_in": UNANSWERABLE_IN[kind],
            "topics": topics,
        }
        for kind, standard, topics in UNANSWERABLE_TOPICS
    ]


def build_tasks(
    tables: dict[str, pd.DataFrame], references: pd.DataFrame, counts: Counts
) -> list[dict]:
    """All tasks, interleaved by type so that a part of the run covers every type."""
    used_sections: set[str] = set()
    single = single_tasks(tables, counts, used_sections)
    cross = cross_reference_tasks(tables, references, counts.cross_reference_nb, used_sections)
    groups: dict[str, list[dict]] = defaultdict(list)
    for task in single + cross + unanswerable_tasks():
        groups[f"{task['question_type']}-{task['written_standard']}"].append(task)
    for name, tasks in groups.items():
        for number, task in enumerate(tasks, start=1):
            task["task_id"] = f"{name}-{number:03d}"
    ordered = []
    longest = max(len(tasks) for tasks in groups.values())
    for position in range(longest):
        for name in sorted(groups):
            if position < len(groups[name]):
                ordered.append(groups[name][position])
    return ordered
