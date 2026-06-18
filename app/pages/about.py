"""Method, data source, limits and privacy in plain words."""

from __future__ import annotations

import streamlit as st
from common import data, lang, t
from texts import ABOUT

from norsk_lov_rag.demo.core import DAILY_LIMIT_PER_IP, MAX_QUESTION_CHARS

st.header(t("page_about"))
manifest = data().manifest
st.markdown(ABOUT[lang()].format(
    sections=manifest.get("sections", "-"), corpus_hash=manifest.get("corpus_hash", "-"),
    per_ip=DAILY_LIMIT_PER_IP, max_chars=MAX_QUESTION_CHARS,
))  # fmt: skip
