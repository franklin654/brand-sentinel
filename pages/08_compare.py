"""Page 8 — Entity Comparison.

Side-by-side risk profile for two entities: latest score, attribution, vendor
impacts, and a dual trend chart overlaying both risk histories on one axes.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from brand_risk import synthetic_data as data
from brand_risk import store as risk_store

st.title("Compare Entities")
st.caption("Select two entities to view their risk profiles side by side.")

# ── Entity selectors ──────────────────────────────────────────────────────────
entities  = list(st.session_state.get("uploaded_watchlist") or data.WATCHLIST)
eid_list  = [e.entity_id for e in entities]
name_map  = {e.entity_id: e.name for e in entities}

if len(entities) < 2:
    st.info("Need at least 2 entities in the watchlist to compare.")
    st.stop()

col_a, col_b = st.columns(2)
eid_a = col_a.selectbox(
    "Entity A",
    options=eid_list,
    format_func=lambda eid: name_map.get(eid, eid),
    index=0,
    key="cmp_a",
)
eid_b = col_b.selectbox(
    "Entity B",
    options=eid_list,
    format_func=lambda eid: name_map.get(eid, eid),
    index=min(1, len(eid_list) - 1),
    key="cmp_b",
)

if eid_a == eid_b:
    st.warning("Select two different entities.")
    st.stop()


def _get_dossier(eid: str):
    """Try focused result → full run → None."""
    focus = (st.session_state.get("entity_focus_results") or {}).get(eid)
    if focus and focus.get("dossiers"):
        return focus["dossiers"][0]
    result = st.session_state.get("result")
    if result:
        return next((d for d in result.get("dossiers", []) if d.entity_id == eid), None)
    return None


def _render_side(eid: str, container) -> None:
    """Render score, attribution, vendor impacts, and DB history for one entity."""
    name    = name_map.get(eid, eid)
    dossier = _get_dossier(eid)
    latest  = risk_store.get_latest(eid)

    with container:
        st.subheader(name)
        if dossier:
            score = dossier.adverse.risk_score
            cat   = dossier.overall_risk.upper()
            st.metric("Risk score", f"{score}/100", cat)
            if dossier.headline:
                st.caption(f"_{dossier.headline}_")
            attr = dossier.risk_attribution
            if attr:
                st.markdown("**Attribution**")
                st.progress(int(attr.get("social_pct", 0)), text=f"Social {attr.get('social_pct',0):.0f}%")
                st.progress(int(attr.get("media_pct",  0)), text=f"Media {attr.get('media_pct',0):.0f}%")
                st.progress(int(attr.get("vendor_pct", 0)), text=f"Vendor {attr.get('vendor_pct',0):.0f}%")
            if dossier.vendor_impacts:
                st.markdown("**Vendor impacts**")
                for v in dossier.vendor_impacts:
                    st.write(f"- **{v.vendor_name}** · {v.exposure} → `{v.recommended_action}`")
        elif latest:
            st.metric("Risk score (last run)", f"{latest['risk_score']}/100", latest["overall_risk"].upper())
        else:
            st.info("No data — run a scan for this entity first.")


# ── Side-by-side panels ───────────────────────────────────────────────────────
st.divider()
panel_a, panel_b = st.columns(2)
_render_side(eid_a, panel_a)
_render_side(eid_b, panel_b)

# ── Overlaid trend chart ──────────────────────────────────────────────────────
st.divider()
st.subheader("Risk score trend comparison")

rows_a = risk_store.get_history(eid_a)
rows_b = risk_store.get_history(eid_b)

if len(rows_a) < 2 and len(rows_b) < 2:
    st.caption("Need ≥ 2 pipeline runs per entity to show a trend overlay.")
else:
    fig, ax = plt.subplots(figsize=(10, 4))
    if len(rows_a) >= 2:
        ax.plot(
            [r["run_ts"][:16] for r in rows_a],
            [r["risk_score"]   for r in rows_a],
            marker="o", linewidth=2, color="#e74c3c",
            label=name_map.get(eid_a, eid_a),
        )
    if len(rows_b) >= 2:
        ax.plot(
            [r["run_ts"][:16] for r in rows_b],
            [r["risk_score"]   for r in rows_b],
            marker="s", linewidth=2, color="#3498db",
            label=name_map.get(eid_b, eid_b),
        )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Risk score")
    ax.tick_params(axis="x", rotation=35, labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
