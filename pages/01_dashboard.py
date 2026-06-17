"""Dashboard — streaming pipeline view, dossier cards, PDF export, risk history."""
from __future__ import annotations

import streamlit as st

from brand_risk import synthetic_data as data
from brand_risk import store as risk_store
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

if _kpi_result := st.session_state.get("result"):
    _dos = _kpi_result.get("dossiers", [])
    k1, k2, k3 = st.columns(3)
    k1.metric("Entities monitored", len(_dos))
    _top = max(_dos, key=lambda d: d.adverse.risk_score, default=None)
    _top_score = _top.adverse.risk_score if _top else 0
    _top_delta = risk_store.get_delta(_top.entity_id) if _top else None
    k2.metric("Highest risk score", _top_score, delta=_top_delta, delta_color="inverse")
    k3.metric("Alerts fired", len(st.session_state.get("alerts", [])))

ctrl1, ctrl2 = st.columns([4, 1])
run_clicked   = ctrl1.button("Run monitoring cycle", type="primary")
include_peers = ctrl2.checkbox("Peer benchmarking", help="Add competitor entities for rank comparison")

_all_entities = list(st.session_state.get("uploaded_watchlist") or [])
scope_entities = st.multiselect(
    "Scope — select entities to scan (leave blank = all)",
    options=[e.entity_id for e in _all_entities],
    format_func=lambda eid: next((e.name for e in _all_entities if e.entity_id == eid), eid),
    default=[],
    placeholder="All entities",
)

left, mid, right = st.columns(3)
left.subheader("1 · Social signal")
mid.subheader("2 · Adverse media")
right.subheader("3 · Vendor impact")
left_ph, mid_ph, right_ph = left.empty(), mid.empty(), right.empty()


_AGENT_LABELS = {"social": "Social signal", "adverse": "Adverse media", "vendor": "Vendor impact", "synthesise": "Synthesising dossiers"}


def _stream_run(graph, init: dict, status) -> dict:
    state: dict = {}
    for chunk in graph.stream(init, stream_mode="updates"):
        for node_name, delta in chunk.items():
            state.update(delta)
            label = _AGENT_LABELS.get(node_name, node_name)
            status.write(f"✓ {label}")
            if node_name == "social" and state.get("trend_signals"):
                render_social(state["trend_signals"], left_ph)
            elif node_name == "adverse" and state.get("adverse_findings"):
                render_adverse(state["adverse_findings"], mid_ph)
            elif node_name == "vendor" and state.get("vendor_risks"):
                render_vendors(state["vendor_risks"], right_ph)
    return state


def _load_inputs(include_peers: bool = False, scope: list[str] | None = None) -> tuple:
    import networkx as nx
    watchlist = list(st.session_state.get("uploaded_watchlist") or [])
    if scope:
        watchlist = [e for e in watchlist if e.entity_id in scope]
    g     = st.session_state.get("uploaded_graph") or data.vendor_graph()
    if scope:
        nodes = set(scope)
        for eid in list(scope):
            if eid in g:
                nodes.update(g.neighbors(eid))
        g = g.subgraph(nodes).copy()
    posts = st.session_state.get("uploaded_posts") or data.social_stream()
    return watchlist, g, posts


if run_clicked:
    watchlist, g, posts = _load_inputs(include_peers, scope=scope_entities or None)
    if not watchlist:
        st.warning("No entities in watchlist — upload a watchlist on the **Documents** page first.")
        st.stop()
    with st.status("Agents running…", expanded=True) as _status:
        result = _stream_run(st.session_state.graph,
                             {"watchlist": watchlist, "graph": g, "posts": posts},
                             _status)
        _status.update(label="Pipeline complete", state="complete", expanded=False)
    st.session_state.result = result
    # In-app alert banners
    dossiers_now = result.get("dossiers", [])
    if dossiers_now:
        from brand_risk.notifier import collect_alerts, send_slack, send_email
        from brand_risk.alert_config_loader import load_alert_config
        alert_cfg = load_alert_config()
        alerts = collect_alerts(dossiers_now, alert_cfg)
        st.session_state.alerts = alerts
        for d in dossiers_now:
            ecfg = alert_cfg.get(d.entity_id, {})
            msg  = next((a for a in alerts if d.entity_name in a), "")
            if not msg:
                continue
            try:
                if "slack" in ecfg.get("notify", []) and ecfg.get("webhook_url"):
                    send_slack(msg, ecfg["webhook_url"])
                if "email" in ecfg.get("notify", []) and ecfg.get("email_to"):
                    send_email(msg, ecfg["email_to"])
            except Exception as _exc:
                pass
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
    _risk_filter = st.multiselect(
        "Show risk levels",
        ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        default=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        key="dossier_risk_filter",
        label_visibility="collapsed",
    )
    visible_dossiers = [d for d in dossiers if d.overall_risk.upper() in _risk_filter]
    for d in visible_dossiers:
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
    g_viz = st.session_state.get("uploaded_graph") or data.vendor_graph()
    if g_viz.number_of_nodes() > 0:
        st.divider()
        st.subheader("Vendor relationship graph")
        st.caption("Drag nodes to rearrange · hover edges for relation label · red = flagged risk")
        flagged = {v.vendor_id for v in result.get("vendor_risks", [])}
        from pyvis.network import Network
        net = Network(height="420px", width="100%", bgcolor="#1a1f2e", font_color="#fafafa")
        net.set_options('{"physics":{"stabilization":{"iterations":60}},"edges":{"color":{"color":"#95a5a6"}}}')
        for node in g_viz.nodes():
            color = "#e74c3c" if node in flagged else "#5d6d7e"
            net.add_node(node, label=g_viz.nodes[node].get("name", node), color=color, size=22)
        for src, tgt, edata in g_viz.edges(data=True):
            net.add_edge(src, tgt, title=edata.get("relation", ""), color="#7f8c8d")
        st.components.v1.html(net.generate_html(), height=430)

    # ── Downloads + executive summary ─────────────────────────────────────────
    if dossiers:
        dl1, dl2, dl3 = st.columns(3)
        json_bytes = ("\n\n".join(d.model_dump_json(indent=2) for d in dossiers)).encode()
        dl1.download_button("Download dossiers (JSON)", json_bytes,
                            file_name="dossiers.json", mime="application/json")
        try:
            from brand_risk.pdf_export import generate_pdf
            dl2.download_button("Download board report (PDF)", generate_pdf(dossiers),
                                file_name="board_report.pdf", mime="application/pdf")
        except Exception as exc:
            dl2.caption(f"PDF unavailable: {exc}")
        if dl3.button("Generate executive summary"):
            with st.spinner("Drafting summary…"):
                from brand_risk.llm import chat
                payload = "\n\n".join(d.model_dump_json(indent=2) for d in dossiers)
                st.session_state.exec_summary = chat(
                    "You are a risk analyst. Write a concise 3-paragraph executive summary "
                    "of these brand risk dossiers for a board audience. Plain text only.",
                    payload, temperature=0.3,
                )
        if exec_sum := st.session_state.get("exec_summary"):
            with st.expander("Executive summary", expanded=True):
                st.write(exec_sum)

# ── Auto-refresh (reads scheduler output from SQLite) ─────────────────────────
st.divider()
ref1, ref2 = st.columns([3, 1])
auto_refresh  = ref1.checkbox("Auto-refresh from scheduler output",
                              help="Polls SQLite and reruns when the scheduler writes new data.")
interval_min  = ref2.selectbox("Interval (min)", [1, 5, 15, 30], index=1,
                               disabled=not auto_refresh, label_visibility="collapsed")
if auto_refresh:
    @st.fragment(run_every=interval_min * 60)
    def _auto_refresh_fragment() -> None:
        if risk_store.get_all_entities():
            st.caption(f"Auto-refresh active — polling every {interval_min} min.")
        else:
            st.caption("Waiting for scheduler to write data…")

    _auto_refresh_fragment()
