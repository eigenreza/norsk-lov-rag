"""Entry point of the demo: language choice, navigation and the notices on every page.

Run locally from the repository root:

    streamlit run app/Home.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Norsk lov-RAG", layout="wide")

from common import disclaimer, footer, t  # noqa: E402

HERE = Path(__file__).parent

choice = st.sidebar.radio("Språk / Language", ["Norsk", "English"], horizontal=True)
st.session_state["lang"] = "en" if choice == "English" else "nb"

pages = [
    st.Page(HERE / "pages" / "ask.py", title=t("page_ask"), url_path="sporsmal", default=True),
    st.Page(HERE / "pages" / "examples.py", title=t("page_examples"), url_path="eksempler"),
    st.Page(HERE / "pages" / "compare.py", title=t("page_compare"), url_path="sammenlign"),
    st.Page(HERE / "pages" / "results.py", title=t("page_results"), url_path="resultater"),
    st.Page(HERE / "pages" / "about.py", title=t("page_about"), url_path="om"),
]
page = st.navigation(pages)
st.sidebar.markdown(t("author"))
st.title(t("title"))
st.markdown(t("author"))
st.write(t("intro"))
disclaimer()
page.run()
footer()
