"""Recorded answers to test questions. No model calls."""

from __future__ import annotations

import streamlit as st
from common import data, t
from views import show_example

st.header(t("page_examples"))
st.write(t("examples_intro"))

demo = data()
if not demo.examples:
    st.info(t("no_examples"))
else:
    labels = [example["question"] for example in demo.examples]
    chosen = st.selectbox(t("pick_example"), range(len(labels)), format_func=labels.__getitem__)
    show_example(demo.examples[chosen], demo)
