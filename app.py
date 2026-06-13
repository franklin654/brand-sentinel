"""Entry point for the Brand & Reputational Risk Intelligence app.

Navigation hub only — business logic lives in pages/.

Run: streamlit run app.py   (from the project root)
"""
from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

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
])
pg.run()
