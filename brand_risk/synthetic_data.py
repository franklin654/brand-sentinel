"""Generates a demo world grounded in real aerospace industry entities.

Boeing's 2024 safety crisis (737 Max door plug blowout, FAA production audit,
Spirit AeroSystems fuselage defects, CEO Senate testimony) gives the jury
recognisable context while the pipeline reasoning remains fully live.

search_news() has three tiers:
  1. SerpAPI Google News (when SERP_API_KEY env var is set)
  2. Semantic similarity over NEWS_CORPUS (when sentence-transformers available)
  3. Substring match over NEWS_CORPUS (always available — keeps smoke.py working)
"""
from __future__ import annotations

import logging
import os
import random
from datetime import datetime, timedelta

import networkx as nx

from .schemas import Entity, SocialPost

logger = logging.getLogger(__name__)

random.seed(7)
_START = datetime(2026, 6, 1, 9, 0, 0)

WATCHLIST = [
    Entity(entity_id="e_boeing",  name="Boeing",             kind="brand",
           aliases=["BA", "Boeing Co", "The Boeing Company"]),
    Entity(entity_id="e_calhoun", name="Dave Calhoun",       kind="executive",
           aliases=["David Calhoun", "Boeing CEO", "David L. Calhoun"]),
    Entity(entity_id="e_spirit",  name="Spirit AeroSystems", kind="vendor",
           aliases=["Spirit Aero", "SPR", "Spirit AeroSystems Holdings"]),
    Entity(entity_id="e_ge_aero", name="GE Aerospace",       kind="vendor",
           aliases=["GE Aviation", "GEA", "General Electric Aerospace"]),
]

# Competitor entities used for peer benchmarking (opt-in via dashboard toggle).
PEER_WATCHLIST: list[Entity] = [
    Entity(entity_id="e_airbus",   name="Airbus",
           kind="brand", aliases=["AIR", "Airbus SE"]),
    Entity(entity_id="e_rtx",      name="RTX Corporation",
           kind="brand", aliases=["RTX", "Raytheon Technologies"]),
]

# Planted negative posts — one scenario per crisis entity.
_SCENARIOS = {
    "e_boeing": [
        "The 737 Max door plug blowout is terrifying — I am never flying Boeing again",
        "Boeing's quality control is in freefall, FAA found hundreds of defects in new planes",
        "How many more incidents before Boeing is held accountable for cutting corners on safety?",
        "Shocking that Boeing jets are still certified after the door plug blew mid-flight, unacceptable",
    ],
    "e_calhoun": [
        "Dave Calhoun's Senate testimony was evasive and insulting to the victims' families",
        "Boeing CEO Calhoun pocketed $45M while the company ignored safety warnings — disgusting",
        "Calhoun admitted knowing about quality issues and did nothing, that is criminal negligence",
        "The Boeing CEO's response to the door plug crisis was tone-deaf and completely unacceptable",
    ],
}

_NEUTRAL = [
    "{n} posted solid earnings this quarter, analysts seem cautiously optimistic",
    "Anyone following {n} lately? Interesting moves in the sector",
    "{n} had a presence at the airshow, impressive display",
    "Reading up on {n} for a case study — decent overview in the annual report",
]

NEWS_CORPUS = [
    {
        "title": "Boeing 737 Max door plug blowout forces emergency landing, FAA launches audit",
        "url": "https://news.example.com/boeing-737-door-plug-2024",
        "snippet": "The Federal Aviation Administration launched a comprehensive audit of Boeing's "
                   "737 Max production line after a door plug separated mid-flight on an Alaska Airlines "
                   "jet. Boeing has been ordered to cap monthly output while inspections are completed.",
        "entities": ["e_boeing"],
    },
    {
        "title": "Spirit AeroSystems fuselage defects at centre of Boeing quality probe",
        "url": "https://news.example.com/spirit-aero-defects-2024",
        "snippet": "Investigators identified improperly drilled fastener holes in fuselage panels "
                   "produced by Spirit AeroSystems, Boeing's primary fuselage supplier. The FAA has "
                   "restricted Boeing's 737 production rate pending corrective action at Spirit's Wichita plant.",
        "entities": ["e_boeing", "e_spirit"],
    },
    {
        "title": "Boeing CEO Dave Calhoun grilled by Senate over safety culture and whistleblower claims",
        "url": "https://news.example.com/calhoun-senate-testimony-2024",
        "snippet": "Senators pressed Boeing chief executive Dave Calhoun on whether financial pressure "
                   "led managers to override safety engineers. Whistleblowers testified that quality "
                   "concerns were systematically suppressed to meet production targets.",
        "entities": ["e_calhoun", "e_boeing"],
    },
    {
        "title": "GE Aerospace engine deliveries on schedule despite Boeing production slowdown",
        "url": "https://news.example.com/ge-aerospace-deliveries-2024",
        "snippet": "GE Aerospace confirmed LEAP engine deliveries remain on track, though analysts warn "
                   "a prolonged Boeing output cap could weigh on GE's order backlog recognition in H2.",
        "entities": ["e_ge_aero", "e_boeing"],
    },
    {
        "title": "Aviation summit focuses on sustainable fuel mandates for 2030",
        "url": "https://news.example.com/aviation-saf-summit-2026",
        "snippet": "Industry leaders convened to discuss SAF blending requirements with no specific "
                   "safety or reputational issues raised for monitored entities.",
        "entities": [],
    },
]


def social_stream(n_per_entity: int = 12) -> list[SocialPost]:
    posts, t = [], _START
    pid = 0
    for ent in WATCHLIST:
        neg = _SCENARIOS.get(ent.entity_id, [])
        for i in range(n_per_entity):
            if neg and i % 3 == 0:
                text = random.choice(neg)
            else:
                text = random.choice(_NEUTRAL).format(n=ent.name)
            t += timedelta(minutes=random.randint(1, 6))
            posts.append(SocialPost(
                post_id=f"p{pid:04d}", entity_id=ent.entity_id,
                text=text, timestamp=t.isoformat(),
            ))
            pid += 1
    random.shuffle(posts)
    return posts


def vendor_graph() -> nx.Graph:
    """Supplier relationships for the Boeing ecosystem."""
    g = nx.Graph()
    for e in WATCHLIST:
        g.add_node(e.entity_id, name=e.name, kind=e.kind)
    g.add_edge("e_boeing",  "e_spirit",  relation="fuselage_supplier")
    g.add_edge("e_boeing",  "e_ge_aero", relation="engine_supplier")
    g.add_edge("e_calhoun", "e_boeing",  relation="ceo_of")
    return g


_SIM_THRESHOLD = 0.25


def search_news(entity_name: str, aliases: list[str]) -> list[dict]:
    """Return relevant news articles for an entity.

    Three-tier resolution:
      1. SerpAPI Google News — when SERP_API_KEY env var is set.
      2. Semantic similarity over NEWS_CORPUS — when sentence-transformers available.
      3. Substring match over NEWS_CORPUS — always available (smoke.py fallback).
    """
    if os.getenv("SERP_API_KEY"):
        try:
            from .live_search import fetch_news
            return fetch_news(entity_name, list(aliases))
        except Exception as exc:
            logger.warning("SerpAPI fetch failed (%s); falling back to corpus", exc)

    try:
        from .embeddings import cosine_scores
        query = " ".join([entity_name] + list(aliases))
        candidates = [f"{a['title']} {a['snippet']}" for a in NEWS_CORPUS]
        scores = cosine_scores(query, candidates)
        return [art for art, score in zip(NEWS_CORPUS, scores) if score >= _SIM_THRESHOLD]
    except ImportError:
        pass

    names = [entity_name.lower(), *[a.lower() for a in aliases]]
    return [
        art for art in NEWS_CORPUS
        if any(nm in (art["title"] + " " + art["snippet"]).lower() for nm in names)
    ]
