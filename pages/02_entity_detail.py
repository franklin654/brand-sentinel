"""Entity detail drill-down — posts, article hits, risk attribution, history."""
from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from brand_risk import synthetic_data as data
from brand_risk.analytics import source_credibility
from brand_risk import store as risk_store

st.title("Entity Detail")
st.caption("Drill-down view for a single monitored entity.")

# ── Entity selector ───────────────────────────────────────────────────────────
entities = data.WATCHLIST
entity_ids = [e.entity_id for e in entities]
entity_names = {e.entity_id: e.name for e in entities}

selected_id = st.selectbox(
    "Select entity",
    options=entity_ids,
    format_func=lambda eid: entity_names.get(eid, eid),
)
selected_name = entity_names.get(selected_id, selected_id)

# ── Latest score from SQLite or session state ─────────────────────────────────
result = st.session_state.get("result")
latest_db = risk_store.get_latest(selected_id)

dossier = None
if result:
    dossier = next(
        (d for d in result.get("dossiers", []) if d.entity_id == selected_id), None
    )

st.divider()
col_score, col_cat = st.columns(2)
if dossier:
    col_score.metric("Risk score", f"{dossier.adverse.risk_score}/100")
    col_cat.metric("Category", dossier.overall_risk.upper())
elif latest_db:
    col_score.metric("Risk score (last run)", f"{latest_db['risk_score']}/100")
    col_cat.metric("Category", latest_db["overall_risk"].upper())
else:
    st.info("No data yet — run the pipeline on the Dashboard first.")

# ── Attribution ───────────────────────────────────────────────────────────────
attr = dossier.risk_attribution if dossier else (
    {k: latest_db.get(k, 0) for k in ("social_pct", "media_pct", "vendor_pct")}
    if latest_db else {}
)
if attr:
    st.subheader("Risk attribution")
    a1, a2, a3 = st.columns(3)
    a1.progress(int(attr.get("social_pct", 0)), text=f"Social {attr.get('social_pct', 0):.0f}%")
    a2.progress(int(attr.get("media_pct",  0)), text=f"Media {attr.get('media_pct', 0):.0f}%")
    a3.progress(int(attr.get("vendor_pct", 0)), text=f"Vendor {attr.get('vendor_pct', 0):.0f}%")

# ── Suggested response ────────────────────────────────────────────────────────
if dossier and dossier.suggested_response:
    with st.expander("Suggested response (from playbook)"):
        st.write(dossier.suggested_response)

# ── Social posts ──────────────────────────────────────────────────────────────
st.subheader("Social posts")
if result and result.get("posts"):
    posts = [p for p in result["posts"] if p.entity_id == selected_id]
    posts_sorted = sorted(posts, key=lambda p: p.sentiment)
    if not posts_sorted:
        st.info(f"No posts for {selected_name} in this run.")
    for p in posts_sorted:
        sentiment = p.sentiment
        colour = "#e74c3c" if sentiment < -0.20 else ("#f39c12" if sentiment < 0 else "#27ae60")
        st.markdown(
            f"<span style='color:{colour}'>●</span> `{sentiment:+.3f}` — {p.text}",
            unsafe_allow_html=True,
        )
else:
    st.info("Run the pipeline on the Dashboard for full post detail.")

# ── Adverse media hits ────────────────────────────────────────────────────────
st.subheader("Adverse media hits")
if dossier:
    finding = dossier.adverse
    hits = finding.hits
    if hits:
        for h in hits:
            cred = source_credibility(h.url)
            badge = "🟢" if cred >= 0.90 else ("🟡" if cred >= 0.70 else "🔴")
            flag = "✓" if h.relevant else "✗"
            st.write(
                f"{flag} {badge} `{cred:.2f}` [{h.title}]({h.url}) — {h.relevance_reason}"
            )
    else:
        st.info("No article hits for this entity.")
else:
    st.info("Run the pipeline on the Dashboard for article hit detail.")

# ── Risk history chart ────────────────────────────────────────────────────────
st.subheader("Risk score history")
rows = risk_store.get_history(selected_id)
if len(rows) < 2:
    st.caption("Need ≥ 2 pipeline runs to show a trend.")
else:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(
        [r["run_ts"][:16] for r in rows],
        [r["risk_score"] for r in rows],
        marker="o", linewidth=2, color="#e74c3c",
    )
    ax.set_ylim(0, 100)
    ax.set_title(f"{selected_name} — risk score over time", fontsize=10)
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=35, labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
