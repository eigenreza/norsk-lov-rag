"""The same question through two lexical search systems, side by side. No generation."""

from __future__ import annotations

import streamlit as st
from common import data, t

from norsk_lov_rag.demo.core import MAX_QUESTION_CHARS, SYSTEMS, check_question, tradeoff

st.header(t("page_compare"))
st.write(t("compare_intro"))

demo = data()
numbers = tradeoff(demo.results.get("retrieval", {}))
if numbers:
    bm25, best = numbers["bm25"], numbers["hybrid+rerank"]
    st.caption(t("tradeoff", b_r=bm25["recall@10"], h_r=best["recall@10"],
                 b_n=bm25["ndcg@10"], h_n=best["ndcg@10"]))  # fmt: skip

with st.form("compare"):
    question = st.text_input(t("question"), max_chars=MAX_QUESTION_CHARS,
                             placeholder=t("placeholder"))  # fmt: skip
    submitted = st.form_submit_button(t("search_button"), type="primary")

if submitted:
    problem = check_question(question)
    if problem:
        st.error(t(problem, n=MAX_QUESTION_CHARS) if problem == "too_long" else t(problem))
        st.stop()
    columns = st.columns(len(SYSTEMS))
    for column, system in zip(columns, SYSTEMS, strict=True):
        with column:
            st.subheader(t(system))
            hits = demo.search(question, system, k=10)
            st.dataframe(
                [{t("rank"): n, t("provision"): h["label"], t("heading"): h["title"],
                  t("score"): round(h["score"], 2)} for n, h in enumerate(hits, start=1)],
                hide_index=True,
                width="stretch",
            )  # fmt: skip
