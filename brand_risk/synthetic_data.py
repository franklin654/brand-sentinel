"""Generates a demo-safe world: a watchlist, a social stream with two planted
crisis scenarios, a small news corpus, and an entity->vendor graph.

Demo reliability beats realism for a hackathon. A planted, reproducible
scenario means your live demo cannot fail because a web search returned
nothing — you get a guaranteed spike, guaranteed adverse hits, and a
guaranteed vendor impact, while the *reasoning* over them is still real.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta

import networkx as nx

from .schemas import Entity, SocialPost

random.seed(7)
_START = datetime(2026, 6, 1, 9, 0, 0)

WATCHLIST = [
    Entity(entity_id="e_acme",   name="Acme Foods",        kind="brand",     aliases=["Acme", "AcmeFoods"]),
    Entity(entity_id="e_rivera", name="Dana Rivera",       kind="executive", aliases=["D. Rivera", "CEO Rivera"]),
    Entity(entity_id="e_nimbus", name="Nimbus Logistics",  kind="vendor",    aliases=["Nimbus"]),
    Entity(entity_id="e_orchard",name="Orchard Packaging", kind="vendor",    aliases=["Orchard Pack"]),
]

# Each scenario plants negative posts for one entity plus matching news.
_SCENARIOS = {
    "e_acme": [
        "Just found mold in my Acme Foods cereal box, absolutely disgusting",
        "Acme Foods recall? my whole family got sick after eating their product",
        "Why is no one talking about the Acme Foods contamination, this is serious",
        "Acme Foods customer service ignored my complaint about spoiled product",
    ],
    "e_rivera": [
        "Dana Rivera's awful, tone-deaf remarks at the conference were completely irresponsible",
        "CEO Rivera's shameful handling of the layoffs is dishonest and wrong",
        "Disgusted by Dana Rivera after that terrible interview, she handled it horribly",
        "Dana Rivera is a disgrace after those dismissive comments about workers",
    ],
}

_NEUTRAL = [
    "{n} launched a new product line today, looks interesting",
    "Anyone tried {n} recently? thinking about it",
    "{n} had a booth at the expo, decent",
    "Solid quarter for {n} from what I read",
]

NEWS_CORPUS = [
    {"title": "Acme Foods issues voluntary recall over contamination concerns",
     "url": "https://news.example.com/acme-recall",
     "snippet": "Acme Foods has recalled several batches after reports of "
                "contamination; regulators have opened an inquiry.",
     "entities": ["e_acme"]},
    {"title": "Health authority investigates Acme Foods supplier hygiene",
     "url": "https://news.example.com/acme-supplier-probe",
     "snippet": "The investigation extends to a packaging supplier linked to "
                "the affected production line.",
     "entities": ["e_acme", "e_orchard"]},
    {"title": "Acme CEO Dana Rivera criticised for handling of layoffs",
     "url": "https://news.example.com/rivera-layoffs",
     "snippet": "Commentators called the executive's remarks dismissive amid "
                "staff cuts.",
     "entities": ["e_rivera", "e_acme"]},
    {"title": "Local sports club announces new sponsor",
     "url": "https://news.example.com/sports-sponsor",
     "snippet": "Unrelated community news with no bearing on monitored brands.",
     "entities": []},
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
    """Who supplies / is associated with whom. Edges carry a relation label."""
    g = nx.Graph()
    for e in WATCHLIST:
        g.add_node(e.entity_id, name=e.name, kind=e.kind)
    g.add_edge("e_acme", "e_nimbus",  relation="logistics_supplier")
    g.add_edge("e_acme", "e_orchard", relation="packaging_supplier")
    g.add_edge("e_rivera", "e_acme",  relation="executive_of")
    return g


_SIM_THRESHOLD = 0.25  # cosine similarity floor for semantic search


def search_news(entity_name: str, aliases: list[str]) -> list[dict]:
    """Stand-in for a live news API. Swap for ddgs/SerpAPI by keeping this
    signature and returning the same dict shape.

    Uses semantic similarity (sentence-transformers) when available; falls back
    to substring matching so smoke.py works without GPU deps.
    """
    try:
        from .embeddings import cosine_scores
        query = " ".join([entity_name] + list(aliases))
        candidates = [f"{a['title']} {a['snippet']}" for a in NEWS_CORPUS]
        scores = cosine_scores(query, candidates)
        return [art for art, score in zip(NEWS_CORPUS, scores) if score >= _SIM_THRESHOLD]
    except ImportError:
        names = [entity_name.lower(), *[a.lower() for a in aliases]]
        return [
            art for art in NEWS_CORPUS
            if any(nm in (art["title"] + " " + art["snippet"]).lower() for nm in names)
        ]
