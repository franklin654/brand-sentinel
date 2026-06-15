"""Chat-page helpers — system prompt builder, history formatter, starter buttons."""
from __future__ import annotations

import streamlit as st

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
