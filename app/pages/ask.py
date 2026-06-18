"""Ask questions in a conversation: tax questions are answered from the law text, other
messages get a fitting reply, and replay takes over when the free limits are used."""

from __future__ import annotations

import dataclasses
import time

import streamlit as st
from common import client, data, limiter, live_enabled, settings, t, visitor_ip
from views import show_citations

from norsk_lov_rag.demo.core import (
    MAX_QUESTION_CHARS,
    Answer,
    answer_question,
    build_sources,
    check_question,
    log_event,
    route_message,
    smart_search,
    talk_answer,
)

EXAMPLES = (
    "Kan jeg trekke fra reiseutgifter til jobben?",
    "Må jeg betale skatt når jeg selger boligen min?",
    "Will I pay less tax if I have children?",
)

st.header(t("page_ask"))
st.info(t("privacy"), icon=":material/lock:")
demo = data()
history: list[dict] = st.session_state.setdefault("chat", [])


def show_turn(turn: dict) -> None:
    with st.chat_message(turn["role"]):
        if turn["role"] == "user":
            st.markdown(turn["content"])
            return
        answer = Answer(**turn["answer"])
        if answer.outcome in ("answer", "abstain", "talk"):
            if answer.abstained:
                st.warning(t("abstained"))
            st.markdown(answer.answer)
            st.caption(t("usage", p=answer.prompt_tokens, c=answer.completion_tokens,
                         s=answer.seconds, m=answer.model_version or "-"))  # fmt: skip
        else:
            st.warning(f"{t(answer.outcome)} {t('replay_hint')}")
        if answer.citations:
            with st.expander(t("citations")):
                show_citations(answer.citations, demo)
        hits = turn.get("hits") or []
        if hits:
            with st.expander(t("details")):
                if turn.get("terms"):
                    st.caption(t("terms", terms=turn["terms"]))
                shown = {s.unit for s in build_sources(demo, settings(), hits)}
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


def reply(message: str) -> dict:
    """Route the message, then search and answer, or answer without the law search."""
    terms, hits, kind = "", [], "tax"
    if not live_enabled():
        answer = Answer("replay_no_key")
    elif not limiter().allow(visitor_ip()):
        answer = Answer("replay_ip_limit")
    else:
        with st.spinner(t("waiting")):
            route = route_message(client(), message, history[:-1])
            kind = route.kind
            if kind == "tax":
                terms, hits = smart_search(client(), demo, route.question,
                                           terms=route.terms or None)  # fmt: skip
                answer = answer_question(
                    demo, settings(), client(), route.question, hits, terms, original=message
                )
            else:
                answer = talk_answer(client(), kind, route.question, history[:-1])
    if kind == "tax" and not hits:
        hits = demo.search(message, "bm25", k=10)
    return {"role": "assistant", "content": answer.answer or t(answer.outcome),
            "answer": dataclasses.asdict(answer), "hits": hits[:10], "terms": terms,
            "kind": kind}  # fmt: skip


if not history:
    with st.chat_message("assistant"):
        st.markdown(t("chat_hello"))
        columns = st.columns(len(EXAMPLES))
        for column, example in zip(columns, EXAMPLES, strict=True):
            if column.button(example, key=f"ex_{example}", width="stretch"):
                st.session_state["pending"] = example
                st.rerun()
else:
    for turn in history:
        show_turn(turn)
    if st.button(t("new_chat")):
        st.session_state["chat"] = []
        st.rerun()

if live_enabled():
    st.caption(t("left", n=limiter().left(visitor_ip())))

message = st.chat_input(t("chat_placeholder"), max_chars=MAX_QUESTION_CHARS)
message = message or st.session_state.pop("pending", None)
if message:
    started = time.perf_counter()
    problem = check_question(message)
    if problem:
        st.error(t(problem, n=MAX_QUESTION_CHARS) if problem == "too_long" else t(problem))
        log_event(message, problem, time.perf_counter() - started)
    else:
        history.append({"role": "user", "content": message})
        turn = reply(message)
        history.append(turn)
        answer = turn["answer"]
        log_event(message, answer["outcome"], time.perf_counter() - started,
                  answer["prompt_tokens"], answer["completion_tokens"])  # fmt: skip
        st.rerun()
