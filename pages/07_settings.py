"""Page 7 — Settings.

Read and write .env configuration from the UI. Changes are persisted to disk;
a Streamlit restart is required for the new values to take effect.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import streamlit as st
from dotenv import dotenv_values, set_key

ENV_PATH = Path(__file__).parent.parent / ".env"

_FIELDS: list[tuple[str, str, bool]] = [
    ("LLM_BASE_URL",           "LLM base URL (OpenAI-compatible)",              False),
    ("LLM_MODEL",              "LLM model name",                                False),
    ("JUPYTER_TOKEN",          "Jupyter proxy token (cloud vLLM only)",         True),
    ("SERP_API_KEY",           "SerpAPI key (leave blank for synthetic data)",  True),
    ("LANGCHAIN_TRACING_V2",   "LangSmith tracing (true / false)",              False),
    ("LANGCHAIN_PROJECT",      "LangSmith project name",                        False),
    ("BRAND_RISK_DB",          "SQLite DB path",                                False),
    ("SMTP_HOST",              "SMTP host (for email alerts)",                  False),
    ("SMTP_PORT",              "SMTP port",                                     False),
    ("SMTP_USER",              "SMTP username",                                 False),
    ("SMTP_PASS",              "SMTP password",                                 True),
]

st.title("Settings")
st.caption("Edit pipeline configuration. Click **Save** — then restart the app to apply changes.")

# ── Load current .env values ──────────────────────────────────────────────────
current: dict[str, str] = {}
if ENV_PATH.exists():
    current = {k: v or "" for k, v in dotenv_values(ENV_PATH).items()}

# ── Editable fields ───────────────────────────────────────────────────────────
st.subheader("Configuration")
new_values: dict[str, str] = {}
for key, label, secret in _FIELDS:
    default = current.get(key, "")
    if secret:
        val = st.text_input(label, value=default, type="password", key=f"cfg_{key}")
    else:
        val = st.text_input(label, value=default, key=f"cfg_{key}")
    new_values[key] = val

_LLM_KEYS = {"LLM_BASE_URL", "LLM_MODEL", "JUPYTER_TOKEN"}

if st.button("Save settings", type="primary"):
    if not ENV_PATH.exists():
        ENV_PATH.touch()
    llm_changed = False
    for k, v in new_values.items():
        if v:
            set_key(str(ENV_PATH), k, v)
            if k in _LLM_KEYS:
                llm_changed = True
    if llm_changed:
        from brand_risk.llm import reset_llm
        reset_llm()
    st.toast("Settings saved — restart the app to apply.", icon="✅")

# ── Live status checks ────────────────────────────────────────────────────────
st.divider()
st.subheader("Live status")

llm_url = new_values.get("LLM_BASE_URL") or os.getenv("LLM_BASE_URL", "")
if llm_url:
    try:
        probe = llm_url.rstrip("/").replace("/v1", "") + "/health"
        urllib.request.urlopen(probe, timeout=2)
        st.success(f"LLM endpoint reachable: `{llm_url}`")
    except Exception:
        st.warning(f"LLM endpoint not reachable: `{llm_url}` (server may still be starting)")
else:
    st.info("LLM_BASE_URL not set.")

serp_key = new_values.get("SERP_API_KEY") or os.getenv("SERP_API_KEY", "")
if serp_key:
    st.success("SerpAPI key configured — live news enabled.")
else:
    st.info("No SERP_API_KEY — pipeline uses synthetic demo data.")

db_path = new_values.get("BRAND_RISK_DB") or os.getenv("BRAND_RISK_DB", "brand_risk.db")
db_file = Path(db_path)
if db_file.exists():
    size_kb = db_file.stat().st_size // 1024
    st.success(f"SQLite DB: `{db_path}` ({size_kb} KB)")
else:
    st.info(f"SQLite DB not yet created: `{db_path}` (created on first pipeline run).")

tracing = new_values.get("LANGCHAIN_TRACING_V2") or os.getenv("LANGCHAIN_TRACING_V2", "")
project = new_values.get("LANGCHAIN_PROJECT") or os.getenv("LANGCHAIN_PROJECT", "brand-risk")
if tracing == "true":
    st.success(f"LangSmith tracing enabled — project **{project}**")
else:
    st.info("LangSmith tracing off (set LANGCHAIN_TRACING_V2=true to enable).")
