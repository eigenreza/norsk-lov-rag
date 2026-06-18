"""Evaluation results, figures and failure examples, all read from demo_data."""

from __future__ import annotations

import streamlit as st
from common import data, lang, t
from texts import FAILURE_KINDS

from norsk_lov_rag.demo.core import generation_rows, retrieval_rows

st.header(t("page_results"))
st.caption(t("results_note"))

demo = data()
retrieval = demo.results.get("retrieval", {})
if retrieval:
    st.subheader(t("results_retrieval"))
    settings = list(retrieval.get("settings", {}))
    setting = st.selectbox(t("corpus"), settings, index=0)
    st.dataframe(retrieval_rows(retrieval, setting), hide_index=True, width="stretch")

st.subheader(t("results_generation"))
generation = demo.results.get("generation", {})
if generation:
    st.dataframe(generation_rows(generation), hide_index=True, width="stretch")
else:
    st.info(t("no_generation"))

if demo.figures:
    st.subheader(t("results_figures"))
    for path in demo.figures:
        st.image(str(path), width="stretch")

failures = generation.get("examples", {}) if generation else {}
if failures:
    st.subheader(t("results_failures"))
    for kind, entry in failures.items():
        name = FAILURE_KINDS.get(kind, {}).get(lang(), kind)
        with st.expander(f"{name}: {entry['count']}"):
            for item in entry["examples"]:
                st.markdown(f"**{item['question']}**")
                st.write(item["answer"])
                if item.get("reason"):
                    st.caption(item["reason"])
                st.caption(item["run"])
