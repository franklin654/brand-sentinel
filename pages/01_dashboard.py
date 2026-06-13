"""Dashboard — streaming pipeline view, dossier cards, PDF export, risk history."""
from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

from brand_risk import synthetic_data as data
from brand_risk.orchestrator import build_graph
from pages._helpers import render_adverse, render_social, render_vendors

st.title("Brand & Reputational Risk Intelligence")
serp_live = bool(__import__("os").getenv("SERP_API_KEY"))
data_badge = "Live · SerpAPI" if serp_live else "Synthetic demo data"
st.caption(
    f"AGENTS_039 → AGENTS_001 → AGENTS_016"
    f"   ·   LangGraph 1.x  ·  vLLM / AMD Instinct MI300X  ·   **{data_badge}**"
)

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

ctrl1, ctrl2 = st.columns([4, 1])
run_clicked   = ctrl1.button("Run monitoring cycle", type="primary")
include_peers = ctrl2.checkbox("Peer benchmarking", help="Add competitor entities for rank comparison")

left, mid, right = st.columns(3)
left.subheader("1 · Social signal")
mid.subheader("2 · Adverse media")
right.subheader("3 · Vendor impact")
left_ph, mid_ph, right_ph = left.empty(), mid.empty(), right.empty()


def _stream_run(graph, init: dict) -> dict:
    state: dict = {}
    for chunk in graph.stream(init, stream_mode="updates"):
        for node_name, delta in chunk.items():
            state.update(delta)
            if node_name == "social" and state.get("trend_signals"):
                render_social(state["trend_signals"], left_ph)
            elif node_name == "adverse" and state.get("adverse_findings"):
                render_adverse(state["adverse_findings"], mid_ph)
            elif node_name == "vendor" and state.get("vendor_risks"):
                render_vendors(state["vendor_risks"], right_ph)
    return state


def _load_inputs(include_peers: bool = False) -> tuple:
    watchlist = list(st.session_state.get("uploaded_watchlist") or data.WATCHLIST)
    if include_peers:
        from brand_risk.synthetic_data import PEER_WATCHLIST
        watchlist = watchlist + [e for e in PEER_WATCHLIST if e.entity_id not in {w.entity_id for w in watchlist}]
    g     = st.session_state.get("uploaded_graph") or data.vendor_graph()
    posts = st.session_state.get("uploaded_posts") or data.social_stream()
    return watchlist, g, posts


if run_clicked:
    watchlist, g, posts = _load_inputs(include_peers)
    with st.spinner("Agents running…"):
        result = _stream_run(st.session_state.graph,
                             {"watchlist": watchlist, "graph": g, "posts": posts})
    st.session_state.result = result
    # In-app alert banners
    dossiers_now = result.get("dossiers", [])
    if dossiers_now:
        from brand_risk.notifier import collect_alerts
        from brand_risk.alert_config_loader import load_alert_config
        alerts = collect_alerts(dossiers_now, load_alert_config())
        st.session_state.alerts = alerts
        for msg in alerts:
            st.toast(msg, icon="🚨")

result = st.session_state.get("result")
if result and not run_clicked:
    render_social(result.get("trend_signals", []), left_ph)
    render_adverse(result.get("adverse_findings", []), mid_ph)
    render_vendors(result.get("vendor_risks", []), right_ph)

# ── Dossier cards ─────────────────────────────────────────────────────────────
if result:
    st.divider()
    st.subheader("Reputation dossiers")
    dossiers = result.get("dossiers", [])
    if not dossiers:
        st.info("No dossiers generated — pipeline found no material risk.")
    for msg in st.session_state.get("alerts", []):
        st.warning(f"🚨 Alert: {msg}")
    for d in dossiers:
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"### {d.entity_name}")
            c1.write(f"_{d.headline}_")
            c2.metric("Risk score", f"{d.adverse.risk_score}/100", d.overall_risk.upper())
            if d.peer_rank > 0:
                c2.caption(f"Rank #{d.peer_rank}/{len(dossiers)} · Median {d.industry_median_score:.0f}")
            attr = d.risk_attribution
            if attr:
                a1, a2, a3 = st.columns(3)
                a1.progress(int(attr.get("social_pct", 0)),
                            text=f"Social {attr.get('social_pct', 0):.0f}%")
                a2.progress(int(attr.get("media_pct",  0)),
                            text=f"Media {attr.get('media_pct', 0):.0f}%")
                a3.progress(int(attr.get("vendor_pct", 0)),
                            text=f"Vendor {attr.get('vendor_pct', 0):.0f}%")
            if d.vendor_impacts:
                st.markdown("**Vendor impacts:**")
                for v in d.vendor_impacts:
                    st.write(f"- **{v.vendor_name}** — {v.exposure} → `{v.recommended_action}`")
            if d.suggested_response:
                with st.expander("Suggested response"):
                    st.write(d.suggested_response)
            st.caption(
                f"Spike {d.trend.detected_at[:19].replace('T',' ')}"
                f"  ·  Generated {d.generated_at[:19].replace('T',' ')}"
            )

    # ── Vendor graph ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Vendor relationship graph")
    flagged = {v.vendor_id for v in result.get("vendor_risks", [])}
    g_viz = data.vendor_graph()
    fig, ax = plt.subplots(figsize=(6, 4))
    pos = nx.spring_layout(g_viz, seed=42)
    colours = ["#e74c3c" if n in flagged else "#95a5a6" for n in g_viz.nodes()]
    nx.draw_networkx_nodes(g_viz, pos, node_color=colours, node_size=900, ax=ax)
    nx.draw_networkx_labels(
        g_viz, pos,
        labels={n: g_viz.nodes[n]["name"].replace(" ", "\n") for n in g_viz.nodes()},
        font_size=7, ax=ax,
    )
    nx.draw_networkx_edges(g_viz, pos, ax=ax, width=1.5, alpha=0.7)
    nx.draw_networkx_edge_labels(
        g_viz, pos, edge_labels=nx.get_edge_attributes(g_viz, "relation"), font_size=6, ax=ax,
    )
    ax.axis("off")
    ax.set_title("Red = flagged vendor risk", fontsize=9)
    st.pyplot(fig)
    plt.close(fig)

    # ── Downloads ─────────────────────────────────────────────────────────────
    if dossiers:
        dl1, dl2 = st.columns(2)
        json_bytes = ("\n\n".join(d.model_dump_json(indent=2) for d in dossiers)).encode()
        dl1.download_button("Download dossiers (JSON)", json_bytes,
                            file_name="dossiers.json", mime="application/json")
        try:
            from brand_risk.pdf_export import generate_pdf
            dl2.download_button("Download board report (PDF)", generate_pdf(dossiers),
                                file_name="board_report.pdf", mime="application/pdf")
        except Exception as exc:
            dl2.caption(f"PDF unavailable: {exc}")

# ── Risk history ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("Risk history")
try:
    from brand_risk import store as risk_store
    entities = risk_store.get_all_entities()
    if not entities:
        st.caption("No history yet — run the pipeline at least once.")
    for eid, ename in entities:
        rows = risk_store.get_history(eid)
        if len(rows) < 2:
            st.caption(f"{ename}: only 1 run recorded — need ≥ 2 to show a trend.")
            continue
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.plot([r["run_ts"][:16] for r in rows], [r["risk_score"] for r in rows],
                marker="o", linewidth=2)
        ax.set_ylim(0, 100)
        ax.set_title(f"{ename} — risk score trend", fontsize=10)
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
except Exception as exc:
    st.caption(f"History unavailable: {exc}")

# ── Auto-refresh (reads scheduler output from SQLite) ─────────────────────────
st.divider()
ref1, ref2 = st.columns([3, 1])
auto_refresh  = ref1.checkbox("Auto-refresh from scheduler output",
                              help="Polls SQLite and reruns when the scheduler writes new data.")
interval_min  = ref2.selectbox("Interval (min)", [1, 5, 15, 30], index=1,
                               disabled=not auto_refresh, label_visibility="collapsed")
if auto_refresh:
    import time
    time.sleep(interval_min * 60)
    st.rerun()
