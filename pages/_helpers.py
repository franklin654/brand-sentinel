"""Shared render helpers used by dashboard and entity detail pages."""
from __future__ import annotations

import streamlit as st

from brand_risk.schemas import Entity
from brand_risk.upload_parser import build_vendor_graph


def render_social(signals: list, container) -> None:
    with container:
        if not signals:
            st.info("No spikes detected.")
            return
        for sig in signals:
            st.error(f"**{sig.entity_name}** — spike detected")
            st.write(f"Mean sentiment {sig.sentiment_delta:.3f}  ·  {sig.volume} negative posts")
            st.write(f"_{sig.narrative_cluster}_")
            with st.expander("Sample posts"):
                for p in sig.sample_posts:
                    st.write("•", p)


def render_adverse(findings: list, container) -> None:
    with container:
        if not findings:
            st.info("No adverse findings.")
            return
        st.caption(f"{sum(len(f.hits) for f in findings)} articles screened")
        for f in findings:
            st.metric(f.entity_name, f"{f.risk_score}/100", f.risk_category.upper())
            st.write(f.explanation)
            with st.expander("Sources"):
                for h in f.hits:
                    flag = "✓" if h.relevant else "✗"
                    published = getattr(h, "published", "")
                    date_str = f" · {published}" if published else ""
                    st.write(f"{flag} [{h.title}]({h.url}){date_str} — {h.relevance_reason}")
            if f.calibration_anchors:
                with st.expander("Calibration anchors (similar past cases)"):
                    for anchor in f.calibration_anchors:
                        st.write("•", anchor)


def render_manual_entry(default_watchlist: list) -> None:
    """Expander with two tabs for adding a watchlist entity or vendor edge without a file upload."""
    with st.expander("Or add entries manually"):
        tab_wl, tab_eg = st.tabs(["Add entity", "Add vendor edge"])
        with tab_wl:
            with st.form("manual_entity", clear_on_submit=True):
                c1, c2 = st.columns(2)
                eid  = c1.text_input("Entity ID", placeholder="e_acme")
                name = c2.text_input("Name", placeholder="Acme Foods")
                kind = st.selectbox("Kind", ["brand", "executive", "vendor"])
                aliases = st.text_input("Aliases (semicolon-separated)", placeholder="Acme;AcmeFoods")
                if st.form_submit_button("Add entity") and eid and name:
                    wl = list(st.session_state.get("uploaded_watchlist") or default_watchlist)
                    if any(e.entity_id == eid for e in wl):
                        st.warning(f"Entity ID '{eid}' already exists.")
                    else:
                        wl.append(Entity(
                            entity_id=eid, name=name, kind=kind,
                            aliases=[a.strip() for a in aliases.split(";") if a.strip()],
                        ))
                        st.session_state.uploaded_watchlist = wl
                        st.success(f"Added '{name}' — {len(wl)} entities total.")
        with tab_eg:
            with st.form("manual_edge", clear_on_submit=True):
                c1, c2, c3 = st.columns(3)
                src = c1.text_input("Source ID", placeholder="e_acme")
                tgt = c2.text_input("Target ID", placeholder="e_nimbus")
                rel = c3.text_input("Relation", placeholder="logistics_supplier")
                if st.form_submit_button("Add edge") and src and tgt and rel:
                    stored = list(st.session_state.get("_manual_edges", []))
                    stored.append((src, tgt, rel))
                    st.session_state._manual_edges = stored
                    wl = st.session_state.get("uploaded_watchlist") or default_watchlist
                    st.session_state.uploaded_graph = build_vendor_graph(wl, stored)
                    st.success(f"Edge {src} → {tgt} added ({len(stored)} total).")


def render_vendors(risks: list, container) -> None:
    with container:
        if not risks:
            st.info("No vendor impacts.")
            return
        for v in risks:
            st.warning(f"**{v.vendor_name}** — {v.exposure} exposure")
            st.write(f"Action: **{v.recommended_action}**")
            st.write(v.rationale)
            if v.cited_clauses:
                with st.expander("Contract clauses referenced"):
                    for clause in v.cited_clauses:
                        st.caption(clause)
