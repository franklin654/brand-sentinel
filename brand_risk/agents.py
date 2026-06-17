"""The three agents.

Design choices that matter for a 3-day build:

* Sentiment scoring (AGENTS_039) uses VADER, not an LLM, per post. Scoring
  hundreds of posts with an LLM is slow and pointless — VADER is instant on
  CPU and good enough to *detect a spike*. The LLM is reserved for the
  reasoning-heavy steps where it actually earns its latency: narrative
  clustering, disambiguation, risk explanation, vendor reasoning.
  (Upgrade path: swap VADER for cardiffnlp/twitter-roberta-base-sentiment on
  ROCm if you want GPU throughput — same call site.)

* Each agent is a plain function `state -> state`. That keeps them unit-
  testable in isolation and trivial to wire into LangGraph (orchestrator.py).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from . import synthetic_data as data
from .llm import chat_json
from .schemas import (
    AdverseFinding, MediaHit, TrendSignal, VendorRisk,
)

_vader = SentimentIntensityAnalyzer()
NEG_POST   = -0.20   # a single post counts as "negative" below this
NEG_RATIO  =  0.30   # fraction of negative posts that constitutes a spike (default)
MIN_VOLUME =  3      # absolute floor so tiny samples don't fire

_ALERT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "alert_config.json")


def _load_alert_config() -> dict:
    """Load per-entity alert thresholds from alert_config.json. Returns {} if absent."""
    if not os.path.exists(_ALERT_CONFIG_PATH):
        return {}
    try:
        with open(_ALERT_CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


# ── AGENTS_039 : Social Media Insights ───────────────────────────────────────
def social_agent(state: dict) -> dict:
    posts = state["posts"]
    for p in posts:
        p.sentiment = _vader.polarity_scores(p.text)["compound"]

    by_entity: dict[str, list] = defaultdict(list)
    for p in posts:
        by_entity[p.entity_id].append(p)

    alert_cfg = _load_alert_config()
    signals = []
    for eid, group in by_entity.items():
        neg = [p for p in group if p.sentiment < NEG_POST]
        ratio = len(neg) / len(group)
        ratio_threshold = alert_cfg.get(eid, {}).get("neg_ratio", NEG_RATIO)
        if len(neg) >= MIN_VOLUME and ratio >= ratio_threshold:
            name = next(e.name for e in state["watchlist"] if e.entity_id == eid)
            samples = [p.text for p in sorted(neg, key=lambda x: x.sentiment)[:4]]
            cluster = _summarise_cluster(name, samples)
            signals.append(TrendSignal(
                entity_id=eid, entity_name=name,
                sentiment_delta=round(sum(p.sentiment for p in group) / len(group), 3),
                volume=len(neg),
                narrative_cluster=cluster, sample_posts=samples,
                detected_at=datetime.utcnow().isoformat(),
            ))
    state["trend_signals"] = signals
    return state


def _summarise_cluster(name: str, samples: list[str]) -> str:
    from .llm import chat
    sys = ("You label clusters of negative social posts. Given posts about an "
           "entity, reply with ONE short sentence naming the core complaint.")
    user = f"Entity: {name}\nPosts:\n- " + "\n- ".join(samples)
    return chat(sys, user, temperature=0.1).strip()


# ── AGENTS_001 : Adverse Media Screening ─────────────────────────────────────
_ADVERSE_SYS = (
    "You are an adverse-media analyst. You are given a monitored entity, the "
    "social narrative that flagged it, and candidate news articles. For each "
    "article decide whether it genuinely concerns THIS entity (disambiguation) "
    "and is adverse. Then assign an overall risk_score 0-100 and a category. "
    "Ground your explanation in the article snippets; never invent facts."
)


def _assess_signal(sig: TrendSignal, watchlist: list) -> AdverseFinding:
    """Score one trend signal against news articles. Designed for parallel execution."""
    from .rag import retrieve_similar_incidents
    from .analytics import annotate_articles

    ent = next(e for e in watchlist if e.entity_id == sig.entity_id)
    articles = annotate_articles(data.search_news(ent.name, ent.aliases))
    anchors = retrieve_similar_incidents(sig.narrative_cluster, k=3)
    anchor_block = (
        "\n\nSimilar past cases — use as scoring anchors:\n"
        + "\n".join(f"- {a}" for a in anchors)
    ) if anchors else ""
    user = (
        f"Entity: {ent.name} (aliases: {', '.join(ent.aliases) or 'none'})\n"
        f"Social narrative: {sig.narrative_cluster}"
        + anchor_block
        + "\n\nCandidate articles:\n" + "\n".join(
            f"[{i}] {a['title']} — {a['snippet']} ({a['url']})"
            f" [source credibility: {a.get('credibility', 0.5):.2f}]"
            for i, a in enumerate(articles)
        )
    )
    finding = chat_json(_ADVERSE_SYS, user, AdverseFinding)
    finding.entity_id = ent.entity_id
    finding.entity_name = ent.name
    finding.calibration_anchors = anchors
    return finding


def adverse_agent(state: dict) -> dict:
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_assess_signal, sig, state["watchlist"]): sig
            for sig in state["trend_signals"]
        }
        findings = [f.result() for f in as_completed(futures)]
    state["adverse_findings"] = findings
    return state


# ── AGENTS_016 : Third-Party / Vendor Risk ───────────────────────────────────
_VENDOR_SYS = (
    "You assess third-party risk. Given an adverse finding about an entity and "
    "its neighbours in a supplier graph, decide each neighbour's exposure "
    "(none/indirect/direct), the risk drivers, and a recommended action "
    "(monitor/engage/diversify/exit). Be conservative; justify with the graph "
    "relation and the adverse finding."
)


def _assess_vendor(
    finding: AdverseFinding, nbr: str, relation: str, nbr_name: str,
) -> VendorRisk:
    """Assess one vendor neighbour. Designed for parallel execution."""
    from .rag import retrieve_contract_clauses

    clauses = retrieve_contract_clauses(nbr, finding.explanation, k=3)
    clause_block = (
        "\n\nRelevant contract clauses — cite these in your rationale:\n"
        + "\n".join(f"- {c[:300]}" for c in clauses)
    ) if clauses else ""
    user = (
        f"Flagged entity: {finding.entity_name} "
        f"(risk {finding.risk_score}/100, {finding.risk_category})\n"
        f"Finding: {finding.explanation}"
        + clause_block
        + f"\n\nNeighbour under review: {nbr_name}\n"
        f"Graph relation: {relation}"
    )
    vr = chat_json(_VENDOR_SYS, user, VendorRisk)
    vr.vendor_id = nbr
    vr.vendor_name = nbr_name
    vr.cited_clauses = clauses
    return vr


def vendor_agent(state: dict) -> dict:
    g = state["graph"]
    seen: set = set()
    tasks: list[tuple] = []

    for finding in state["adverse_findings"]:
        if finding.risk_category == "low":
            continue
        for nbr in g.neighbors(finding.entity_id):
            if (finding.entity_id, nbr) in seen:
                continue
            seen.add((finding.entity_id, nbr))
            relation = g.edges[finding.entity_id, nbr]["relation"]
            nbr_name = g.nodes[nbr]["name"]
            tasks.append((finding, nbr, relation, nbr_name))

    risks: list[VendorRisk] = []
    if tasks:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(_assess_vendor, finding, nbr, relation, nbr_name)
                for finding, nbr, relation, nbr_name in tasks
            ]
            risks = [f.result() for f in as_completed(futures)]

    state["vendor_risks"] = risks
    return state
