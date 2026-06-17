"""Entry point for the Brand & Reputational Risk Intelligence app.

Navigation hub only — business logic lives in pages/.

Run: streamlit run app.py   (from the project root)
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

st.set_page_config(
    page_title="Brand & Reputational Risk Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading RAG models…")
def _init_rag() -> bool:
    """Pre-warm all Chroma singletons once per process via Streamlit's resource cache."""
    from brand_risk.rag import _get_store, _get_embeddings
    from brand_risk.rag_contracts import _get_contract_store
    from brand_risk.rag_playbook import _get_playbook_store
    _get_embeddings()
    _get_store()
    _get_contract_store()
    _get_playbook_store()
    return True


_init_rag()

# Restore persisted watchlist from disk (survives Streamlit restarts)
if "uploaded_watchlist" not in st.session_state:
    from brand_risk.watchlist_store import load_watchlist
    _saved = load_watchlist()
    if _saved:
        st.session_state.uploaded_watchlist = _saved

# Restore persisted RSS config from disk
if "rss_config" not in st.session_state:
    from brand_risk.rss_store import load_rss
    _rss = load_rss()
    if _rss:
        st.session_state.rss_config = _rss

with st.sidebar:
    st.title("Brand Risk · TCS-AMD")
    st.caption("Brand & Reputational Risk Intelligence · Hackathon")
    st.divider()

    serp_key = os.getenv("SERP_API_KEY", "")
    if serp_key:
        st.success("SerpAPI · live news enabled")
    else:
        st.warning("SERP_API_KEY not set — synthetic news")

    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        project = os.getenv("LANGCHAIN_PROJECT", "brand-risk")
        st.success(f"LangSmith · project **{project}**")
    else:
        st.info("LangSmith tracing off")

pg = st.navigation([
    st.Page("pages/01_dashboard.py",     title="Dashboard",     icon="📊"),
    st.Page("pages/02_entity_detail.py", title="Entity Detail", icon="🔍"),
    st.Page("pages/03_alerts.py",        title="Alert Rules",   icon="🔔"),
    st.Page("pages/04_documents.py",     title="Documents",     icon="📄"),
    st.Page("pages/05_chat.py",          title="Analyst Chat",  icon="💬"),
    st.Page("pages/06_history.py",       title="Run History",   icon="📈"),
    st.Page("pages/07_settings.py",      title="Settings",      icon="⚙️"),
    st.Page("pages/08_compare.py",       title="Compare",       icon="⚖️"),
])
pg.run()
