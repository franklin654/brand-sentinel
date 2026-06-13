"""LangGraph wiring: watchlist -> social -> (spike?) -> adverse -> (risky?) ->
vendor -> dossier.

The two conditional edges are the whole efficiency argument of the flow:
an entity only costs an expensive adverse-media screen if social detected a
spike, and a vendor screen only runs if the adverse finding was material.
That gating is what makes a three-agent pipeline cheap enough to run
continuously rather than as three separate batch jobs.
"""
from __future__ import annotations

from datetime import datetime
from typing import TypedDict

import networkx as nx
from langgraph.graph import END, StateGraph

from . import agents
from .schemas import (
    AdverseFinding, Entity, ReputationDossier, SocialPost, TrendSignal, VendorRisk,
)


class FlowState(TypedDict, total=False):
    watchlist: list[Entity]
    graph: nx.Graph
    posts: list[SocialPost]
    trend_signals: list[TrendSignal]
    adverse_findings: list[AdverseFinding]
    vendor_risks: list[VendorRisk]
    dossiers: list[ReputationDossier]


def _has_spike(state: FlowState) -> str:
    return "adverse" if state.get("trend_signals") else END


def _is_risky(state: FlowState) -> str:
    risky = [f for f in state.get("adverse_findings", [])
             if f.risk_category != "low"]
    return "vendor" if risky else "synthesise"


def _synthesise(state: FlowState) -> FlowState:
    dossiers = []
    for sig in state["trend_signals"]:
        finding = next((f for f in state["adverse_findings"]
                        if f.entity_id == sig.entity_id), None)
        if not finding:
            continue
        g = state["graph"]
        neighbor_ids = set(g.neighbors(sig.entity_id))
        impacts = [v for v in state.get("vendor_risks", []) if v.vendor_id in neighbor_ids]
        dossiers.append(ReputationDossier(
            entity_id=sig.entity_id, entity_name=sig.entity_name,
            headline=sig.narrative_cluster,
            overall_risk=finding.risk_category,
            trend=sig, adverse=finding, vendor_impacts=impacts,
            generated_at=datetime.utcnow().isoformat(),
        ))
    state["dossiers"] = dossiers
    return state


def build_graph():
    g = StateGraph(FlowState)
    g.add_node("social", agents.social_agent)
    g.add_node("adverse", agents.adverse_agent)
    g.add_node("vendor", agents.vendor_agent)
    g.add_node("synthesise", _synthesise)

    g.set_entry_point("social")
    g.add_conditional_edges("social", _has_spike, {"adverse": "adverse", END: END})
    g.add_conditional_edges("adverse", _is_risky,
                            {"vendor": "vendor", "synthesise": "synthesise"})
    g.add_edge("vendor", "synthesise")
    g.add_edge("synthesise", END)
    return g.compile()
