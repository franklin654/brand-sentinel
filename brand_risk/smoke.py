"""Deterministic spine test — runs the non-LLM half end to end so you can
verify data generation, spike detection, the vendor graph, and news retrieval
without a GPU or a running Ollama. This is your Day-1 green light.

    python -m brand_risk.smoke
"""
from __future__ import annotations

from collections import defaultdict

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from . import synthetic_data as data
from .agents import MIN_VOLUME, NEG_POST, NEG_RATIO


def main() -> None:
    vader = SentimentIntensityAnalyzer()
    posts = data.social_stream()
    for p in posts:
        p.sentiment = vader.polarity_scores(p.text)["compound"]

    by = defaultdict(list)
    for p in posts:
        by[p.entity_id].append(p)

    print("Spike detection:")
    spikes = []
    for eid, grp in by.items():
        name = next(e.name for e in data.WATCHLIST if e.entity_id == eid)
        neg = [p for p in grp if p.sentiment < NEG_POST]
        ratio = len(neg) / len(grp)
        fired = len(neg) >= MIN_VOLUME and ratio >= NEG_RATIO
        if fired:
            spikes.append(name)
        print(f"  {name:18} neg={len(neg):2}/{len(grp):2} ratio={ratio:.0%} "
              f"-> {'SPIKE' if fired else 'ok'}")

    g = data.vendor_graph()
    print("\nVendor graph:")
    for u, v, r in g.edges(data=True):
        print(f"  {g.nodes[u]['name']} -[{r['relation']}]- {g.nodes[v]['name']}")

    print("\nNews retrieval (Acme Foods):")
    for a in data.search_news("Acme Foods", ["Acme"]):
        print("  -", a["title"])

    assert spikes, "expected at least one spike — check thresholds"
    print(f"\nOK — {len(spikes)} spike(s) detected: {', '.join(spikes)}")


if __name__ == "__main__":
    main()
