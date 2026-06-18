"""Two ways to cut the corpus into chunks, so that their effect on retrieval can be compared.

Structure-aware chunks follow the law: one section per chunk, with the act name, chapter and
section title in front. A long section is split at paragraph borders. Paragraphs that do not
belong to any section are packed together within their document.

Naive chunks ignore the structure: fixed windows of words that run through each document.

Every chunk lists the units it covers. A unit is a section id, or a paragraph id for text outside
sections. Evaluation compares units with the gold sections of a question. A window lists the unit
with most of its words first, so that a window that only starts with the tail of one section
counts mainly for the next one.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

MAX_WORDS = 300
WINDOW_WORDS = 200
WINDOW_STEP = 150
STRATEGIES = ("structure", "naive")


def _title(documents: pd.DataFrame) -> dict[str, str]:
    names = documents["short_name"].fillna("").str.strip()
    titles = documents["title"].fillna("").str.strip()
    return dict(zip(documents["doc_key"], names.where(names != "", titles), strict=True))


def prefix(
    title: str, chapter_path: str | None, label: str | None, section_title: str | None
) -> str:
    """Act name, chapter and section title, as one line in front of the text."""
    parts = [title, (chapter_path or "").replace(" > ", ". ")]
    parts.append(" ".join(part for part in (label, section_title) if part))
    return ". ".join(part.strip() for part in parts if part and part.strip())


def pack(paragraphs: list[str], max_words: int = MAX_WORDS) -> list[str]:
    """Join paragraphs into pieces of at most ``max_words`` words; a longer paragraph is cut."""
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in paragraphs:
        words = paragraph.split()
        if len(words) > max_words:
            if current:
                pieces.append("\n".join(current))
                current, size = [], 0
            pieces += [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]
            continue
        if current and size + len(words) > max_words:
            pieces.append("\n".join(current))
            current, size = [], 0
        current.append(paragraph)
        size += len(words)
    if current:
        pieces.append("\n".join(current))
    return pieces


def _ordered(provisions: pd.DataFrame) -> pd.DataFrame:
    return provisions.sort_values(["doc_key", "order"], kind="stable")


def structure_chunks(corpus: dict[str, pd.DataFrame]) -> pd.DataFrame:
    titles = _title(corpus["documents"])
    sections = corpus["sections"]
    live = sections[~sections["is_empty"] & ~sections["is_repealed"]]
    provisions = _ordered(corpus["provisions"])
    by_section = {
        key: group["text"].tolist()
        for key, group in provisions[provisions["section_id"].notna()].groupby(
            "section_id", sort=False
        )
    }
    rows = []
    for section in live.itertuples(index=False):
        head = prefix(
            titles.get(section.doc_key, ""),
            section.chapter_path,
            section.section_label,
            section.section_title,
        )
        paragraphs = by_section.get(section.section_id) or [section.text]
        pieces = pack(paragraphs)
        for number, piece in enumerate(pieces):
            chunk_id = section.section_id if len(pieces) == 1 else f"{section.section_id}#{number}"
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "doc_key": section.doc_key,
                    "units": [section.section_id],
                    "text": f"{head}\n{piece}",
                    "written_standard": section.written_standard,
                }
            )
    loose = provisions[provisions["section_id"].isna()]
    for doc_key, group in loose.groupby("doc_key", sort=False):
        head = prefix(titles.get(doc_key, ""), None, None, None)
        for number, piece in enumerate(pack(group["text"].tolist())):
            chunk_id = f"{doc_key}#{number}"
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "doc_key": doc_key,
                    "units": [chunk_id],
                    "text": f"{head}\n{piece}",
                    "written_standard": group["written_standard"].iloc[0],
                }
            )
    return pd.DataFrame(rows)


def window_units(owners: list[str]) -> list[str]:
    """The units in a window, the one with most words first; ties keep their reading order."""
    counts = Counter(owners)
    return sorted(dict.fromkeys(owners), key=lambda unit: -counts[unit])


def naive_chunks(corpus: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Windows of ``WINDOW_WORDS`` words every ``WINDOW_STEP`` words, without any heading."""
    sections = corpus["sections"]
    live = set(sections.loc[~sections["is_empty"] & ~sections["is_repealed"], "section_id"])
    provisions = _ordered(corpus["provisions"])
    keep = provisions["section_id"].isna() | provisions["section_id"].isin(live)
    rows = []
    for doc_key, group in provisions[keep].groupby("doc_key", sort=False):
        words: list[str] = []
        owners: list[str] = []
        for provision in group.itertuples(index=False):
            unit = provision.section_id if isinstance(provision.section_id, str) else None
            unit = unit or f"{doc_key}#{provision.provision_id.rsplit('/', 1)[-1]}"
            tokens = provision.text.split()
            words += tokens
            owners += [unit] * len(tokens)
        standard = group["written_standard"].iloc[0]
        starts = range(0, max(1, len(words) - WINDOW_WORDS + WINDOW_STEP), WINDOW_STEP)
        for number, start in enumerate(starts):
            end = min(len(words), start + WINDOW_WORDS)
            if start >= end:
                break
            rows.append(
                {
                    "chunk_id": f"{doc_key}#w{number}",
                    "doc_key": doc_key,
                    "units": window_units(owners[start:end]),
                    "text": " ".join(words[start:end]),
                    "written_standard": standard,
                }
            )
    return pd.DataFrame(rows)


def build_chunks(corpus: dict[str, pd.DataFrame], strategy: str) -> pd.DataFrame:
    if strategy == "structure":
        return structure_chunks(corpus)
    if strategy == "naive":
        return naive_chunks(corpus)
    raise ValueError(f"unknown chunking strategy {strategy!r}")


def chunk_path(data_dir: Path, scope: str, strategy: str) -> Path:
    return data_dir / "chunks" / f"{scope}-{strategy}.parquet"
