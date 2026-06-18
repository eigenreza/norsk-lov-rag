"""Ask a question: search, a live answer within the free limits, or replay when they are used."""

from __future__ import annotations

import time

import streamlit as st
from common import client, data, limiter, live_enabled, settings, t, visitor_ip
from views import show_citations, show_example

from norsk_lov_rag.demo.core import (
    MAX_QUESTION_CHARS,
    Answer,
    answer_question,
    build_sources,
    check_question,
    closest_example,
    log_event,
)

st.header(t("page_ask"))
st.info(t("privacy"), icon=":material/lock:")

demo = data()
with st.form("ask"):
    question = st.text_area(
        t("question"), max_chars=MAX_QUESTION_CHARS, placeholder=t("placeholder"), height=100
    )
    submitted = st.form_submit_button(t("ask_button"), type="primary")

if live_enabled():
    st.caption(t("left", n=limiter().left(visitor_ip())))

if submitted:
    started = time.perf_counter()
    problem = check_question(question)
    if problem:
        st.error(t(problem, n=MAX_QUESTION_CHARS) if problem == "too_long" else t(problem))
        log_event(question, problem, time.perf_counter() - started)
        st.stop()
    hits = demo.search(question, "bm25", k=10)
    if not live_enabled():
        answer = Answer("replay_no_key")
    elif not limiter().allow(visitor_ip()):
        answer = Answer("replay_ip_limit")
    else:
        with st.spinner(t("waiting")):
            answer = answer_question(demo, settings(), client(), question, hits)
    log_event(question, answer.outcome, time.perf_counter() - started,
              answer.prompt_tokens, answer.completion_tokens)  # fmt: skip

    if answer.outcome in ("answer", "abstain"):
        st.subheader(t("answer"))
        if answer.abstained:
            st.warning(t("abstained"))
        st.markdown(answer.answer)
        st.caption(t("usage", p=answer.prompt_tokens, c=answer.completion_tokens,
                     s=answer.seconds, m=answer.model_version or "-"))  # fmt: skip
        if not answer.abstained:
            st.subheader(t("citations"))
            show_citations(answer.citations, demo)
    else:
        st.warning(f"{t(answer.outcome)} {t('replay_hint')}")

    shown = {s.unit for s in build_sources(demo, settings(), hits)}
    st.subheader(t("passages"))
    st.dataframe(
        [
            {
                t("rank"): n,
                t("provision"): h["label"],
                t("heading"): h["title"],
                t("score"): round(h["score"], 2),
                t("shown"): h["unit"] in shown,
            }
            for n, h in enumerate(hits, start=1)
        ],  # fmt: skip
        hide_index=True,
        width="stretch",
    )
    for hit in hits[:5]:
        with st.expander(f"{hit['label']} {hit['title']}".strip()):
            st.text(demo.book.text(hit["unit"]))

    if answer.outcome not in ("answer", "abstain"):
        example = closest_example(question, demo.examples)
        if example:
            st.subheader(t("closest"))
            show_example(example, demo)
