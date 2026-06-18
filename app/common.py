"""Shared parts of the demo pages: data, settings, limits, language and the notices."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from texts import TEXT  # noqa: E402

from norsk_lov_rag.config import load_settings  # noqa: E402
from norsk_lov_rag.demo.core import DemoData, RateLimiter, has_key, make_client  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
# The HTTP client logs every request URL. Keep the log to the demo's own lines.
logging.getLogger("httpx").setLevel(logging.WARNING)


def lang() -> str:
    return st.session_state.get("lang", "nb")


def t(key: str, **values) -> str:
    text = TEXT[key][lang()]
    return text.format(**values) if values else text


@st.cache_resource
def data() -> DemoData:
    return DemoData.load()


@st.cache_resource
def settings():
    cfg = load_settings(ROOT / "configs" / "default.yaml")
    if not cfg.llm.env_file.is_absolute():
        cfg.llm.env_file = ROOT / cfg.llm.env_file
    if not cfg.paths.data_dir.is_absolute():
        cfg.paths.data_dir = ROOT / cfg.paths.data_dir
    return cfg


@st.cache_resource
def limiter() -> RateLimiter:
    return RateLimiter()


@st.cache_resource
def client():
    state = os.environ.get("DEMO_STATE_DIR")
    return make_client(settings(), Path(state) if state else None)


def live_enabled() -> bool:
    """Live answers need a key, and can be turned off with DEMO_REPLAY=1."""
    if os.environ.get("DEMO_REPLAY") == "1":
        return False
    return has_key(settings())


def visitor_ip() -> str | None:
    """The visitor's address, from the proxy header on the host, without the port."""
    forwarded = st.context.headers.get("X-Forwarded-For", "") if st.context.headers else ""
    ip = forwarded.split(",")[0].strip() or (st.context.ip_address or "")
    if ip.count(":") == 1:
        ip = ip.split(":")[0]
    return ip or None


def disclaimer() -> None:
    st.warning(t("disclaimer"), icon=":material/info:")


def footer() -> None:
    attribution = data().manifest.get("attribution", "")
    st.divider()
    st.caption(f"{t('attribution_intro')} {attribution}")
