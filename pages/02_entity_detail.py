"""Entity detail drill-down — focused scan, posts, article hits, risk attribution, heatmap."""
from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import streamlit as st

from brand_risk import synthetic_data as data
from brand_risk.analytics import source_credibility
from brand_risk import store as risk_store
from brand_risk.orchestrator import build_graph

st.title("Entity Detail")
st.caption("Drill-down view for a single monitored entity.")

# ── Entity selector ───────────────────────────────────────────────────────────
entities = st.session_state.get("uploaded_watchlist") or []
if not entities:
    st.info("No entities loaded — upload a watchlist on the **Documents** page first.")
    st.stop()
entity_map = {e.entity_id: e for e in entities}
entity_ids = list(entity_map.keys())
entity_names = {eid: e.name for eid, e in entity_map.items()}

selected_id = st.selectbox(
    "Select entity",
    options=entity_ids,
    format_func=lambda eid: entity_names.get(eid, eid),
)
selected_entity = entity_map[selected_id]
selected_name = selected_entity.name

# ── Single-entity focused scan ────────────────────────────────────────────────
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

_AGENT_LABELS = {
    "social": "Social signal",
    "adverse": "Adverse media",
    "vendor": "Vendor impact",
    "synthesise": "Synthesising dossier",
}


def _stream_focus_run(graph, entity, g_full, posts_full) -> dict:
    """Run the 3-agent pipeline scoped to a single entity."""
    g_ego = nx.ego_graph(g_full, entity.entity_id, radius=1) if entity.entity_id in g_full else g_full
    focused_posts = [p for p in posts_full if p.entity_id == entity.entity_id]
    state: dict = {}
    for chunk in graph.stream(
        {"watchlist": [entity], "graph": g_ego, "posts": focused_posts},
        stream_mode="updates",
    ):
        for node_name, delta in chunk.items():
            state.update(delta)
            st.session_state._focus_status_holder.write(
                f"✓ {_AGENT_LABELS.get(node_name, node_name)}"
            )
    return state


scan_clicked = st.button(
    "Scan this entity", type="primary",
    help="Run the full 3-agent pipeline for this entity only.",
)

if scan_clicked:
    g_full    = st.session_state.get("uploaded_graph") or data.vendor_graph()
    posts_all = st.session_state.get("uploaded_posts") or data.social_stream()
    with st.status(f"Scanning {selected_name}…", expanded=True) as _st:
        st.session_state._focus_status_holder = _st
        focus_result = _stream_focus_run(st.session_state.graph, selected_entity, g_full, posts_all)
        _st.update(label=f"{selected_name} — scan complete", state="complete", expanded=False)
    focus_store = st.session_state.get("entity_focus_results", {})
    focus_store[selected_id] = focus_result
    st.session_state.entity_focus_results = focus_store

# ── Latest score: focused scan > full run > SQLite ────────────────────────────
focus_result = (st.session_state.get("entity_focus_results") or {}).get(selected_id)
full_result  = st.session_state.get("result")
latest_db    = risk_store.get_latest(selected_id)

dossier = None
if focus_result and focus_result.get("dossiers"):
    dossier = focus_result["dossiers"][0]
elif full_result:
    dossier = next(
        (d for d in full_result.get("dossiers", []) if d.entity_id == selected_id), None
    )

st.divider()
col_score, col_cat = st.columns(2)
if dossier:
    _delta = risk_store.get_delta(selected_id)
    col_score.metric("Risk score", f"{dossier.adverse.risk_score}/100",
                     delta=_delta, delta_color="inverse")
    col_cat.metric("Category", dossier.overall_risk.upper())
elif latest_db:
    _delta = risk_store.get_delta(selected_id)
    col_score.metric("Risk score (last run)", f"{latest_db['risk_score']}/100",
                     delta=_delta, delta_color="inverse")
    col_cat.metric("Category", latest_db["overall_risk"].upper())
else:
    st.info("No data yet — click 'Scan this entity' or run the pipeline on the Dashboard.")

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

# ── PDF export (focused scan only) ───────────────────────────────────────────
if dossier:
    try:
        from brand_risk.pdf_export import generate_pdf
        st.download_button(
            "Download report (PDF)",
            data=generate_pdf([dossier]),
            file_name=f"{selected_id}_report.pdf",
            mime="application/pdf",
        )
    except Exception as _pdf_exc:
        st.caption(f"PDF unavailable: {_pdf_exc}")

# ── Suggested response ────────────────────────────────────────────────────────
if dossier and dossier.suggested_response:
    with st.expander("Suggested response (from playbook)"):
        st.write(dossier.suggested_response)

# ── Social posts ──────────────────────────────────────────────────────────────
st.subheader("Social posts")
posts_src = focus_result or full_result
if posts_src and posts_src.get("posts"):
    posts = [p for p in posts_src["posts"] if p.entity_id == selected_id]
    posts_sorted = sorted(posts, key=lambda p: p.sentiment)
    if not posts_sorted:
        st.info(f"No posts for {selected_name} in this run.")
    for p in posts_sorted:
        colour = "#e74c3c" if p.sentiment < -0.20 else ("#f39c12" if p.sentiment < 0 else "#27ae60")
        st.markdown(
            f"<span style='color:{colour}'>●</span> `{p.sentiment:+.3f}` — {p.text}",
            unsafe_allow_html=True,
        )
else:
    st.info("Click 'Scan this entity' or run the full pipeline on the Dashboard for post detail.")

# ── Adverse media hits ────────────────────────────────────────────────────────
st.subheader("Adverse media hits")
if dossier:
    hits = dossier.adverse.hits
    if hits:
        for h in hits:
            cred = source_credibility(h.url)
            badge = "🟢" if cred >= 0.90 else ("🟡" if cred >= 0.70 else "🔴")
            flag = "✓" if h.relevant else "✗"
            st.write(f"{flag} {badge} `{cred:.2f}` [{h.title}]({h.url}) — {h.relevance_reason}")
            if h.snippet:
                with st.expander("Snippet", expanded=False):
                    st.caption(h.snippet)
    else:
        st.info("No article hits for this entity.")
else:
    st.info("Run a scan for article hit detail.")

# ── Risk history chart ────────────────────────────────────────────────────────
st.subheader("Risk score history")
hist_rows = risk_store.get_history(selected_id)
if len(hist_rows) < 2:
    st.caption("Need ≥ 2 pipeline runs to show a trend.")
else:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(
        [r["run_ts"][:16] for r in hist_rows],
        [r["risk_score"] for r in hist_rows],
        marker="o", linewidth=2, color="#e74c3c",
    )
    ax.set_ylim(0, 100)
    ax.set_title(f"{selected_name} — risk score over time", fontsize=10)
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=35, labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

# ── Risk heatmap (entity × run) ───────────────────────────────────────────────
all_entities = risk_store.get_all_entities()
if len(all_entities) > 1:
    st.subheader("Risk score heatmap")
    st.caption("All entities × pipeline runs — darker red = higher risk")
    histories = {eid: risk_store.get_history(eid) for eid, _ in all_entities}
    all_ts = sorted({r["run_ts"][:16] for rows in histories.values() for r in rows})
    if len(all_ts) >= 2:
        row_labels = [ename for _, ename in all_entities]
        matrix = np.full((len(all_entities), len(all_ts)), np.nan)
        ts_idx = {ts: i for i, ts in enumerate(all_ts)}
        for r_idx, (eid, _) in enumerate(all_entities):
            for row in histories[eid]:
                c_idx = ts_idx.get(row["run_ts"][:16])
                if c_idx is not None:
                    matrix[r_idx, c_idx] = row["risk_score"]
        fig2, ax2 = plt.subplots(
            figsize=(max(5, len(all_ts) * 1.2), len(all_entities) * 0.9 + 1)
        )
        im = ax2.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=100, aspect="auto")
        ax2.set_xticks(range(len(all_ts)))
        ax2.set_xticklabels(all_ts, rotation=35, ha="right", fontsize=7)
        ax2.set_yticks(range(len(row_labels)))
        ax2.set_yticklabels(row_labels, fontsize=8)
        plt.colorbar(im, ax=ax2, label="Risk score")
        st.pyplot(fig2)
        plt.close(fig2)
