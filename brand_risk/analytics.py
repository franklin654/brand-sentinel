"""Risk attribution model and source credibility lookup.

Two pure-Python utilities — no LangChain or ML deps.

compute_attribution():  breaks a risk score into social / media / vendor %
source_credibility():   returns a 0.0–1.0 weight for a news article's domain
annotate_articles():    stamps credibility onto article dicts for prompt injection
"""
from __future__ import annotations

CREDIBILITY_TIER: dict[str, float] = {
    "reuters.com": 1.0,
    "apnews.com": 1.0,
    "bbc.com": 1.0,
    "bbc.co.uk": 1.0,
    "ft.com": 0.95,
    "bloomberg.com": 0.95,
    "wsj.com": 0.90,
    "theguardian.com": 0.85,
    "nytimes.com": 0.85,
    "economist.com": 0.85,
    "forbes.com": 0.75,
    "businessinsider.com": 0.70,
    "cnbc.com": 0.70,
}
DEFAULT_CREDIBILITY = 0.50

EXPOSURE_WEIGHT: dict[str, int] = {"none": 0, "indirect": 1, "direct": 2}


def source_credibility(url: str) -> float:
    """Return a credibility weight 0.0–1.0 for a news article URL.

    Looks up the domain in CREDIBILITY_TIER; defaults to 0.5 for unknown sources.
    """
    try:
        domain = url.split("/")[2].replace("www.", "").lower()
        return CREDIBILITY_TIER.get(domain, DEFAULT_CREDIBILITY)
    except (IndexError, AttributeError):
        return DEFAULT_CREDIBILITY


def annotate_articles(articles: list[dict]) -> list[dict]:
    """Stamp a 'credibility' key onto each article dict (mutates in place).

    Called in adverse_agent() before building the user prompt so the credibility
    score appears inline as "[source credibility: 0.95]" for each article.
    """
    for art in articles:
        art["credibility"] = source_credibility(art.get("url", ""))
    return articles


def compute_attribution(signal, finding, vendor_impacts: list) -> dict[str, float]:
    """Break a risk score into social / media / vendor percentage contributions.

    Weights:
      social = |sentiment_delta| × negative_post_volume
      media  = risk_score × number_of_relevant_hits
      vendor = sum of exposure weights (none=0, indirect=1, direct=2)

    Returns dict with keys: social_pct, media_pct, vendor_pct (floats summing to ~100).
    """
    social_w = abs(signal.sentiment_delta) * signal.volume
    media_w = finding.risk_score * len([h for h in finding.hits if h.relevant])
    vendor_w = float(sum(EXPOSURE_WEIGHT.get(v.exposure, 0) for v in vendor_impacts))
    total = social_w + media_w + vendor_w or 1.0
    return {
        "social_pct": round(social_w / total * 100, 1),
        "media_pct":  round(media_w  / total * 100, 1),
        "vendor_pct": round(vendor_w / total * 100, 1),
    }
