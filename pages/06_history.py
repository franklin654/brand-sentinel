"""Page 6 — Run History.

Browse all past pipeline runs stored in SQLite. Rows are sortable by any column.
Select a row to expand the full dossier JSON for that run.
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from brand_risk import store as risk_store

st.title("Run History")
st.caption("All pipeline runs persisted to SQLite — sorted newest first.")

entities = risk_store.get_all_entities()
if not entities:
    st.info("No run history yet — execute a monitoring cycle on the Dashboard first.")
    st.stop()

rows: list[dict] = []
raw_map: dict[tuple, dict] = {}

for eid, ename in entities:
    for h in risk_store.get_history(eid):
        ts = h["run_ts"][:16]
        rows.append({
            "Entity":    ename,
            "Run (UTC)": ts,
            "Score":     h["risk_score"],
            "Level":     h["risk_category"].upper(),
        })
        raw_map[(ename, ts)] = h

if not rows:
    st.info("No run data found.")
    st.stop()

df = pd.DataFrame(rows).sort_values("Run (UTC)", ascending=False).reset_index(drop=True)

# ── Summary KPIs ──────────────────────────────────────────────────────────────
k1, k2, k3 = st.columns(3)
k1.metric("Total runs recorded", len(df))
k2.metric("Entities tracked", df["Entity"].nunique())
k3.metric("Peak score (all time)", int(df["Score"].max()))

st.divider()

# ── Filterable table ──────────────────────────────────────────────────────────
entity_filter = st.multiselect(
    "Filter by entity",
    options=sorted(df["Entity"].unique()),
    default=[],
    placeholder="All entities",
)
level_filter = st.multiselect(
    "Filter by risk level",
    options=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
    default=[],
    placeholder="All levels",
)

view = df.copy()
if entity_filter:
    view = view[view["Entity"].isin(entity_filter)]
if level_filter:
    view = view[view["Level"].isin(level_filter)]

st.dataframe(
    view,
    use_container_width=True,
    column_config={
        "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100),
        "Level": st.column_config.TextColumn("Level"),
    },
    hide_index=True,
)

# ── Row drill-down ────────────────────────────────────────────────────────────
st.divider()
st.subheader("Dossier drill-down")
col_ent, col_ts = st.columns(2)
sel_entity = col_ent.selectbox("Entity", options=sorted(df["Entity"].unique()))
sel_ts_opts = sorted(
    {r["Run (UTC)"] for _, r in df[df["Entity"] == sel_entity].iterrows()},
    reverse=True,
)
sel_ts = col_ts.selectbox("Run timestamp", options=sel_ts_opts)

record = raw_map.get((sel_entity, sel_ts))
if record:
    with st.expander("Raw dossier record", expanded=True):
        st.json(record)
