"""Shared render helpers used by dashboard and entity detail pages."""
from __future__ import annotations

import streamlit as st

from brand_risk.schemas import Entity
from brand_risk.upload_parser import build_vendor_graph
from brand_risk import store as risk_store


def build_system_prompt(dossiers: list, entity_filter: str | None = None) -> str:
    """Serialize dossiers into a plain-text system prompt for the analyst chat."""
    targets = [d for d in dossiers if entity_filter is None or d.entity_id == entity_filter]
    lines = [
        "You are a brand and reputational risk analyst. The following intelligence was "
        f"produced by the latest pipeline run ({targets[0].generated_at[:19].replace('T', ' ')} UTC "
        f"— {len(targets)} entit{'y' if len(targets) == 1 else 'ies'}).\n",
    ]
    for d in targets:
        attr = d.risk_attribution
        lines.append(
            f"--- ENTITY: {d.entity_name} [{d.overall_risk.upper()} risk, "
            f"score {d.adverse.risk_score}/100] ---"
        )
        lines.append(f"Narrative: {d.trend.narrative_cluster}")
        lines.append(
            f"Sentiment delta: {d.trend.sentiment_delta:.3f}  |  "
            f"Volume: {d.trend.volume} negative posts"
        )
        if attr:
            lines.append(
                f"Attribution: Social {attr.get('social_pct', 0):.0f}%  |  "
                f"Media {attr.get('media_pct', 0):.0f}%  |  "
                f"Vendor {attr.get('vendor_pct', 0):.0f}%"
            )
        lines.append(f"Explanation: {d.adverse.explanation}")
        relevant_hits = [h for h in d.adverse.hits if h.relevant]
        if relevant_hits:
            lines.append("Media sources (relevant):")
            for i, h in enumerate(relevant_hits, 1):
                lines.append(f"  {i}. \"{h.title}\" — {h.url}")
        if d.vendor_impacts:
            lines.append("Vendor impacts:")
            for v in d.vendor_impacts:
                lines.append(
                    f"  - {v.vendor_name} ({v.exposure}): "
                    f"{v.recommended_action.upper()} — {v.rationale}"
                )
        if d.peer_rank > 0:
            lines.append(
                f"Peer rank: #{d.peer_rank} of {len(dossiers)}  |  "
                f"Industry median score: {d.industry_median_score:.0f}"
            )
        lines.append("")
    lines.append(
        "Answer analyst questions concisely. Cite article titles and URLs when discussing "
        "adverse findings. Flag uncertainties. Do not invent data not present above."
    )
    return "\n".join(lines)


def format_history_context(entity_id: str) -> str:
    """Return a compact chronological risk history string for injection into user messages."""
    rows = risk_store.get_history(entity_id)
    if not rows:
        return "No historical run data found for this entity."
    lines = ["Historical risk scores (chronological):"]
    for r in rows:
        lines.append(f"  {r['run_ts'][:16]}  —  {r['risk_score']}/100  ({r['risk_category']})")
    return "\n".join(lines)


def render_starter_buttons(questions: list[str]) -> str | None:
    """Render question strings as buttons; return the label of the one clicked, or None."""
    cols = st.columns(2)
    for i, q in enumerate(questions):
        if cols[i % 2].button(q, use_container_width=True, key=f"starter_{i}"):
            return q
    return None


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


def render_entity_manager(default_watchlist: list) -> None:
    """Current watchlist table with per-entity delete buttons."""
    wl = list(st.session_state.get("uploaded_watchlist") or default_watchlist)
    if not wl:
        st.caption("Watchlist is empty.")
        return
    st.caption(f"{len(wl)} entities")
    for i, e in enumerate(wl):
        c1, c2, c3 = st.columns([2, 3, 1])
        c1.write(f"**{e.name}**  `{e.entity_id}`")
        c2.caption(f"{e.kind}  ·  {', '.join(e.aliases) if e.aliases else 'no aliases'}")
        if c3.button("Delete", key=f"del_ent_{i}"):
            wl.pop(i)
            st.session_state.uploaded_watchlist = wl
            st.rerun()


def render_edge_manager(default_watchlist: list) -> None:
    """Current vendor graph edges with per-edge delete buttons."""
    from brand_risk import synthetic_data as _data
    g = st.session_state.get("uploaded_graph") or _data.vendor_graph()
    wl = list(st.session_state.get("uploaded_watchlist") or default_watchlist)
    name_map = {e.entity_id: e.name for e in wl}
    edges = [(u, v, d.get("relation", "")) for u, v, d in g.edges(data=True)]
    if not edges:
        st.caption("No vendor edges.")
        return
    st.caption(f"{len(edges)} edges")
    for i, (src, tgt, rel) in enumerate(edges):
        c1, c2 = st.columns([5, 1])
        c1.write(f"{name_map.get(src, src)}  →  **{rel}**  →  {name_map.get(tgt, tgt)}")
        if c2.button("Delete", key=f"del_edge_{i}"):
            remaining = [(s, t, r) for j, (s, t, r) in enumerate(edges) if j != i]
            st.session_state._manual_edges = remaining
            st.session_state.uploaded_graph = build_vendor_graph(wl, remaining)
            st.rerun()


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
