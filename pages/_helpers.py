"""Shared render helpers used by dashboard and entity detail pages."""
from __future__ import annotations

import streamlit as st


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
