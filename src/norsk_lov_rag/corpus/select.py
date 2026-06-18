"""Pick the rows that belong to one of the two corpora.

The tax corpus is a fixed list of documents. The full corpus is every act and central
regulation. Repealed and empty sections are left out of both, since they hold no rule to find.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from norsk_lov_rag.config import CorpusConfig

SCOPES = ("tax", "full")


def corpus_documents(documents: pd.DataFrame, cfg: CorpusConfig, scope: str) -> pd.DataFrame:
    if scope == "tax":
        missing = sorted(set(cfg.tax_documents) - set(documents["doc_key"]))
        if missing:
            raise KeyError(f"tax documents not found in the corpus: {missing}")
        return documents[documents["doc_key"].isin(cfg.tax_documents)]
    if scope == "full":
        return documents[documents["folder"].isin(cfg.full_folders)]
    raise ValueError(f"unknown corpus scope {scope!r}, expected one of {SCOPES}")


def corpus_sections(
    sections: pd.DataFrame, documents: pd.DataFrame, cfg: CorpusConfig, scope: str
) -> pd.DataFrame:
    """Sections in force with text, for the documents of the corpus."""
    keys = set(corpus_documents(documents, cfg, scope)["doc_key"])
    keep = sections["doc_key"].isin(keys) & ~sections["is_repealed"] & ~sections["is_empty"]
    return sections[keep]


def corpus_provisions(
    provisions: pd.DataFrame,
    sections: pd.DataFrame,
    documents: pd.DataFrame,
    cfg: CorpusConfig,
    scope: str,
) -> pd.DataFrame:
    """Text blocks of the corpus: paragraphs of sections in force, and text outside sections."""
    keys = set(corpus_documents(documents, cfg, scope)["doc_key"])
    live = set(corpus_sections(sections, documents, cfg, scope)["section_id"])
    in_corpus = provisions["doc_key"].isin(keys)
    in_live_section = provisions["section_id"].isin(live)
    outside_sections = provisions["section_id"].isna()
    return provisions[in_corpus & (in_live_section | outside_sections)]


def load_corpus(processed: Path, cfg: CorpusConfig, scope: str) -> dict[str, pd.DataFrame]:
    """Read the tables from disk and cut them down to one corpus."""
    documents = pd.read_parquet(processed / "documents.parquet")
    sections = pd.read_parquet(processed / "sections.parquet")
    provisions = pd.read_parquet(processed / "provisions.parquet")
    return {
        "documents": corpus_documents(documents, cfg, scope),
        "sections": corpus_sections(sections, documents, cfg, scope),
        "provisions": corpus_provisions(provisions, sections, documents, cfg, scope),
    }
