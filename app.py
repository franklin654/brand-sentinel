"""Streamlit control tower — the thing the jury actually watches.

Three panels mirror the three agents so the demo narrates itself:
  1. Live social feed with a sentiment spike lighting up red.
  2. Adverse-media dossier with source-grounded, explainable risk.
  3. Entity graph showing which vendors inherited the risk.

Uses LangGraph 1.x graph.stream(mode="updates") so each column lights up
the moment its agent node completes — no async/event-loop conflicts with
Streamlit's runtime.

Run: streamlit run app.py   (from the project root)
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

from brand_risk import synthetic_data as data
from brand_risk.orchestrator import build_graph

st.set_page_config(page_title="Brand & Reputational Risk Intelligence", layout="wide")
st.title("Brand & Reputational Risk Intelligence")
st.caption(
    "AGENTS_039  →  AGENTS_001  →  AGENTS_016"
    "   ·   LangGraph 1.x  ·  vLLM on AMD Instinct MI300X / ROCm"
)

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

# ── Run button ────────────────────────────────────────────────────────────────
col_run, col_info = st.columns([1, 4])
run_clicked = col_run.button("Run monitoring cycle", type="primary")

if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    project = os.getenv("LANGCHAIN_PROJECT", "brand-risk")
    col_info.info(f"LangSmith tracing active — project **{project}**")

# ── Three result columns (placeholders filled during streaming) ───────────────
left, mid, right = st.columns(3)
left.subheader("1 · Social signal")
mid.subheader("2 · Adverse media")
right.subheader("3 · Vendor impact")

left_ph  = left.empty()
mid_ph   = mid.empty()
right_ph = right.empty()


def _render_social(signals: list, container) -> None:
    with container:
        for sig in signals:
            st.error(f"**{sig.entity_name}** — spike detected")
            st.write(f"Mean sentiment {sig.sentiment_delta:.3f}  ·  {sig.volume} negative posts")
            st.write(f"_{sig.narrative_cluster}_")
            with st.expander("Sample posts"):
                for p in sig.sample_posts:
                    st.write("•", p)


def _render_adverse(findings: list, container) -> None:
    with container:
        for f in findings:
            st.metric(f.entity_name, f"{f.risk_score}/100", f.risk_category.upper())
            st.write(f.explanation)
            with st.expander("Sources"):
                for h in f.hits:
                    flag = "✓" if h.relevant else "✗"
                    st.write(f"{flag} [{h.title}]({h.url}) — {h.relevance_reason}")


def _render_vendors(risks: list, container) -> None:
    with container:
        for v in risks:
            st.warning(f"**{v.vendor_name}** — {v.exposure} exposure")
            st.write(f"Action: **{v.recommended_action}**")
            st.write(v.rationale)


def _stream_run(graph, init: dict) -> dict:
    """Stream the graph via synchronous graph.stream(mode='updates').

    Each column lights up the moment its agent node completes.
    LangGraph 1.x stream() yields {node_name: state_delta} dicts.
    """
    state: dict = {}
    for chunk in graph.stream(init, stream_mode="updates"):
        for node_name, delta in chunk.items():
            state.update(delta)
            if node_name == "social" and state.get("trend_signals"):
                _render_social(state["trend_signals"], left_ph)
            elif node_name == "adverse" and state.get("adverse_findings"):
                _render_adverse(state["adverse_findings"], mid_ph)
            elif node_name == "vendor" and state.get("vendor_risks"):
                _render_vendors(state["vendor_risks"], right_ph)
    return state


if run_clicked:
    init = {
        "watchlist": data.WATCHLIST,
        "graph": data.vendor_graph(),
        "posts": data.social_stream(),
    }
    with st.spinner("Agents running…"):
        result = _stream_run(st.session_state.graph, init)
    st.session_state.result = result

# ── Render from cached result (after page reloads) ───────────────────────────
result = st.session_state.get("result")
if result and not run_clicked:
    _render_social(result.get("trend_signals", []), left_ph)
    _render_adverse(result.get("adverse_findings", []), mid_ph)
    _render_vendors(result.get("vendor_risks", []), right_ph)

# ── Dossier summary cards ─────────────────────────────────────────────────────
if result:
    st.divider()
    st.subheader("Reputation dossiers")
    dossiers = result.get("dossiers", [])
    for d in dossiers:
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"### {d.entity_name}")
            c1.write(f"_{d.headline}_")
            c2.metric("Risk score", f"{d.adverse.risk_score}/100", d.overall_risk.upper())
            if d.vendor_impacts:
                st.markdown("**Vendor impacts:**")
                for v in d.vendor_impacts:
                    st.write(
                        f"- **{v.vendor_name}** — {v.exposure} exposure → "
                        f"`{v.recommended_action}`"
                    )
            ts_spike = d.trend.detected_at[:19].replace("T", " ")
            ts_gen   = d.generated_at[:19].replace("T", " ")
            st.caption(f"Spike detected {ts_spike}  ·  Dossier generated {ts_gen}")

    # ── Vendor graph visualisation ────────────────────────────────────────────
    st.divider()
    st.subheader("Vendor relationship graph")
    flagged_vendors = {v.vendor_id for v in result.get("vendor_risks", [])}
    g = data.vendor_graph()
    fig, ax = plt.subplots(figsize=(6, 4))
    pos = nx.spring_layout(g, seed=42)
    node_colours = [
        "#e74c3c" if n in flagged_vendors else "#95a5a6"
        for n in g.nodes()
    ]
    nx.draw_networkx_nodes(g, pos, node_color=node_colours, node_size=900, ax=ax)
    nx.draw_networkx_labels(
        g, pos,
        labels={n: g.nodes[n]["name"].replace(" ", "\n") for n in g.nodes()},
        font_size=7, ax=ax,
    )
    edge_labels = nx.get_edge_attributes(g, "relation")
    nx.draw_networkx_edges(g, pos, ax=ax, width=1.5, alpha=0.7)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=6, ax=ax)
    ax.axis("off")
    ax.set_title("Red = flagged vendor risk", fontsize=9)
    st.pyplot(fig)
    plt.close(fig)

    # ── Dossier download ──────────────────────────────────────────────────────
    if dossiers:
        json_bytes = ("\n\n".join(d.model_dump_json(indent=2) for d in dossiers)).encode()
        st.download_button(
            "Download dossiers (JSON)", json_bytes,
            file_name="dossiers.json", mime="application/json",
        )
