"""Corpus selection: which documents, sections and text blocks belong to each corpus."""

from __future__ import annotations

import pandas as pd
import pytest

from norsk_lov_rag.config import CorpusConfig, load_settings
from norsk_lov_rag.corpus.select import corpus_documents, corpus_provisions, corpus_sections

CFG = CorpusConfig(full_folders=["nl", "sf"], tax_documents=["lov/1"])


def tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    documents = pd.DataFrame(
        [
            {"doc_key": "lov/1", "folder": "nl"},
            {"doc_key": "forskrift/2", "folder": "sf"},
            {"doc_key": "forskrift/3", "folder": "stv"},
        ]
    )
    sections = pd.DataFrame(
        [
            {"section_id": "lov/1/§1", "doc_key": "lov/1", "is_repealed": False, "is_empty": False},
            {"section_id": "lov/1/§2", "doc_key": "lov/1", "is_repealed": True, "is_empty": True},
            {"section_id": "forskrift/2/§1", "doc_key": "forskrift/2", "is_repealed": False,
             "is_empty": False},
            {"section_id": "forskrift/3/§1", "doc_key": "forskrift/3", "is_repealed": False,
             "is_empty": False},
        ]
    )  # fmt: skip
    provisions = pd.DataFrame(
        [
            {"provision_id": "lov/1/§1/ledd/1", "section_id": "lov/1/§1", "doc_key": "lov/1"},
            {"provision_id": "lov/1/§2/annet/1", "section_id": "lov/1/§2", "doc_key": "lov/1"},
            {"provision_id": "lov/1/#ledd-9", "section_id": None, "doc_key": "lov/1"},
            {"provision_id": "forskrift/2/§1/ledd/1", "section_id": "forskrift/2/§1",
             "doc_key": "forskrift/2"},
            {"provision_id": "forskrift/3/§1/ledd/1", "section_id": "forskrift/3/§1",
             "doc_key": "forskrift/3"},
        ]
    )  # fmt: skip
    return documents, sections, provisions


def test_tax_corpus_is_the_listed_documents_without_repealed_sections() -> None:
    documents, sections, provisions = tables()
    assert list(corpus_documents(documents, CFG, "tax")["doc_key"]) == ["lov/1"]
    assert list(corpus_sections(sections, documents, CFG, "tax")["section_id"]) == ["lov/1/§1"]
    kept = list(corpus_provisions(provisions, sections, documents, CFG, "tax")["provision_id"])
    assert kept == ["lov/1/§1/ledd/1", "lov/1/#ledd-9"]


def test_full_corpus_is_acts_and_central_regulations_only() -> None:
    documents, sections, provisions = tables()
    assert list(corpus_documents(documents, CFG, "full")["doc_key"]) == ["lov/1", "forskrift/2"]
    kept = list(corpus_provisions(provisions, sections, documents, CFG, "full")["provision_id"])
    assert "forskrift/3/§1/ledd/1" not in kept and "forskrift/2/§1/ledd/1" in kept


def test_unknown_scope_and_missing_tax_documents_are_errors() -> None:
    documents, _, _ = tables()
    with pytest.raises(ValueError):
        corpus_documents(documents, CFG, "everything")
    with pytest.raises(KeyError):
        corpus_documents(documents, CorpusConfig(tax_documents=["lov/404"]), "tax")


def test_default_settings_name_the_tax_documents_and_leave_out_other_folders() -> None:
    corpus = load_settings().corpus
    assert corpus.full_folders == ["nl", "sf"]
    assert "lov/1999-03-26-14" in corpus.tax_documents
    assert len(corpus.tax_documents) == len(set(corpus.tax_documents)) == 9
