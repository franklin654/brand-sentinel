"""Deterministic spine test — runs the non-LLM half end to end so you can
verify spike detection, the vendor graph, and news retrieval without a GPU or
a running Ollama. Uses its own hardcoded test data so it is independent of
synthetic_data.py corpus content.

    python -m brand_risk.smoke
"""
from __future__ import annotations

from collections import defaultdict

import networkx as nx
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .agents import MIN_VOLUME, NEG_POST, NEG_RATIO
from .schemas import Entity, SocialPost

_TEST_ENTITIES: list[Entity] = [
    Entity(entity_id="e_acme", name="Acme Foods", kind="brand", aliases=["Acme"]),
    Entity(entity_id="e_nimbus", name="Nimbus Logistics", kind="vendor", aliases=["Nimbus"]),
]

_TEST_POSTS_RAW: list[tuple[str, str]] = [
    ("e_acme", "Terrible product recall, completely unacceptable safety failure."),
    ("e_acme", "Awful contamination scandal — never buying Acme again."),
    ("e_acme", "Disgusting response to the crisis, shareholders furious."),
    ("e_acme", "Another bad quarter for Acme, revenue down sharply."),
    ("e_acme", "Great new product launch from Acme today, very impressed."),
    ("e_acme", "Acme partnership with local farmers is a positive step."),
    ("e_acme", "Solid earnings, Acme beats estimates on cost reduction."),
    ("e_nimbus", "Nimbus delivers on time as usual, good service."),
    ("e_nimbus", "Nimbus Logistics announces fleet expansion across Europe."),
]


def _build_test_graph() -> nx.Graph:
    g: nx.Graph = nx.Graph()
    for e in _TEST_ENTITIES:
        g.add_node(e.entity_id, name=e.name)
    g.add_edge("e_acme", "e_nimbus", relation="logistics_supplier")
    return g


def main() -> None:
    vader = SentimentIntensityAnalyzer()
    posts: list[SocialPost] = []
    for i, (eid, text) in enumerate(_TEST_POSTS_RAW):
        p = SocialPost(post_id=f"smoke_{i}", entity_id=eid, text=text,
                       timestamp="2026-01-01T00:00:00",
                       sentiment=vader.polarity_scores(text)["compound"])
        posts.append(p)

    by: defaultdict[str, list[SocialPost]] = defaultdict(list)
    for p in posts:
        by[p.entity_id].append(p)

    entity_names = {e.entity_id: e.name for e in _TEST_ENTITIES}

    print("Spike detection:")
    spikes: list[str] = []
    for eid, grp in by.items():
        name = entity_names.get(eid, eid)
        neg = [p for p in grp if p.sentiment < NEG_POST]
        ratio = len(neg) / len(grp)
        fired = len(neg) >= MIN_VOLUME and ratio >= NEG_RATIO
        if fired:
            spikes.append(name)
        print(f"  {name:18} neg={len(neg):2}/{len(grp):2} ratio={ratio:.0%} "
              f"-> {'SPIKE' if fired else 'ok'}")

    g = _build_test_graph()
    print("\nVendor graph:")
    for u, v, r in g.edges(data=True):
        print(f"  {g.nodes[u]['name']} -[{r['relation']}]- {g.nodes[v]['name']}")

    from . import synthetic_data as data
    print("\nNews retrieval (Acme Foods):")
    for a in data.search_news("Acme Foods", ["Acme"]):
        print("  -", a["title"])

    assert spikes, "expected at least one spike — check thresholds"
    print(f"\nOK — {len(spikes)} spike(s) detected: {', '.join(spikes)}")


if __name__ == "__main__":
    main()
