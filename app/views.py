"""Pieces of the pages that show an answer, its citations and a recorded example."""

from __future__ import annotations

import streamlit as st
from common import t

TYPES = {
    "direct": {"nb": "direkte", "en": "direct"},
    "everyday": {"nb": "hverdagsspråk", "en": "everyday language"},
    "cross_reference": {"nb": "henvisning", "en": "cross-reference"},
    "unanswerable": {"nb": "uten svar i lovteksten", "en": "not answered by the law text"},
}
STANDARDS = {"nb": "bokmål", "nn": "nynorsk"}


def show_citations(citations: list[dict], demo) -> None:
    if not citations:
        st.caption(t("no_citations"))
        return
    for cite in citations:
        unit = cite.get("unit")
        if unit is None:
            st.error(f"[{cite['citation']}] {t('unresolved')}")
            continue
        status = t("in_sources") if cite.get("in_sources") else t("not_in_sources")
        label = demo.book.labels.get(unit, cite["citation"])
        title = demo.book.titles.get(unit, "")
        with st.expander(f"{label} {title} ({status})".strip()):
            st.text(demo.book.text(unit))


def show_example(example: dict, demo) -> None:
    from common import lang

    kind = TYPES.get(example["question_type"], {}).get(lang(), example["question_type"])
    standard = STANDARDS.get(example["written_standard"], example["written_standard"])
    st.markdown(f"**{example['question']}**")
    st.caption(f"{t('type')}: {kind}. {t('standard')}: {standard}.")
    if example["abstained"]:
        st.warning(t("abstained"))
    st.markdown(example["answer"])
    st.caption(t("recorded_with", model=example.get("model", "-"),
                 system=example.get("system", "-")))  # fmt: skip
    if example.get("correctness"):
        st.caption(f"{t('judge')}: {example['correctness']}")
    if example.get("reference_answer"):
        with st.expander(t("reference")):
            st.write(example["reference_answer"])
    if not example["abstained"]:
        show_citations(example["citations"], demo)
