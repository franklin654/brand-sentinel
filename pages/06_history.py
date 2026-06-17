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
    hist = risk_store.get_history(eid)
    for i, h in enumerate(hist):
        ts = h["run_ts"][:16]
        delta = h["risk_score"] - hist[i - 1]["risk_score"] if i > 0 else None
        rows.append({
            "Entity":    ename,
            "Run (UTC)": ts,
            "Score":     h["risk_score"],
            "Δ":         delta,
            "Level":     h["risk_category"].upper(),
        })
        raw_map[(ename, ts)] = h

if not rows:
    st.info("No run data found.")
    st.stop()

df = pd.DataFrame(rows).sort_values("Run (UTC)", ascending=False).reset_index(drop=True)

# ── Summary KPIs ──────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total runs recorded", len(df))
k2.metric("Entities tracked", df["Entity"].nunique())
k3.metric("Peak score (all time)", int(df["Score"].max()))
k4.download_button(
    "Export CSV",
    data=df.to_csv(index=False).encode(),
    file_name="run_history.csv",
    mime="text/csv",
)

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
        "Δ":     st.column_config.NumberColumn("Δ", format="%+d", help="Change from previous run"),
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
    dossier_json = record.get("dossier_json")
    if dossier_json:
        try:
            from brand_risk.schemas import ReputationDossier
            d = ReputationDossier.model_validate_json(dossier_json)
            with st.container(border=True):
                dc1, dc2 = st.columns([3, 1])
                dc1.markdown(f"### {d.entity_name}")
                dc1.write(f"_{d.headline}_")
                dc2.metric("Risk score", f"{d.adverse.risk_score}/100", d.overall_risk.upper())
                attr = d.risk_attribution
                if attr:
                    da1, da2, da3 = st.columns(3)
                    da1.progress(int(attr.get("social_pct", 0)),
                                 text=f"Social {attr.get('social_pct', 0):.0f}%")
                    da2.progress(int(attr.get("media_pct", 0)),
                                 text=f"Media {attr.get('media_pct', 0):.0f}%")
                    da3.progress(int(attr.get("vendor_pct", 0)),
                                 text=f"Vendor {attr.get('vendor_pct', 0):.0f}%")
                if d.vendor_impacts:
                    st.markdown("**Vendor impacts:**")
                    for v in d.vendor_impacts:
                        st.write(f"- **{v.vendor_name}** — {v.exposure} → `{v.recommended_action}`")
                if d.suggested_response:
                    with st.expander("Suggested response"):
                        st.write(d.suggested_response)
                st.caption(f"Run: {record['run_ts'][:19].replace('T', ' ')} UTC")
        except Exception:
            with st.expander("Raw dossier record", expanded=True):
                st.json(record)
    else:
        with st.expander("Raw dossier record", expanded=True):
            st.json(record)
