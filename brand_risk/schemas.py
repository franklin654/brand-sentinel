"""Typed contracts that flow between the three agents.

Keeping every agent's output as a Pydantic model is the single most useful
discipline in a 3-day build: it forces each LLM call to return parseable
structure, makes the handoffs explicit, and lets the dashboard render off
known fields instead of free text.
"""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


# ── Watchlist ────────────────────────────────────────────────────────────────
class Entity(BaseModel):
    """A brand, executive, or vendor we are monitoring."""
    entity_id: str
    name: str
    kind: Literal["brand", "executive", "vendor"]
    aliases: list[str] = Field(default_factory=list)


# ── AGENTS_039 : Social Media Insights ───────────────────────────────────────
class SocialPost(BaseModel):
    post_id: str
    entity_id: str
    text: str
    timestamp: str          # ISO 8601
    sentiment: float = 0.0  # -1.0 .. +1.0, filled by the agent


class TrendSignal(BaseModel):
    """A detected spike in negative chatter about one entity."""
    entity_id: str
    entity_name: str
    sentiment_delta: float      # how far below baseline this window dropped
    volume: int                 # number of posts in the spike window
    narrative_cluster: str      # one-line summary of what people are saying
    sample_posts: list[str]
    detected_at: str


# ── AGENTS_001 : Adverse Media Screening ─────────────────────────────────────
class MediaHit(BaseModel):
    title: str
    url: str
    snippet: str
    relevant: bool              # did disambiguation keep it?
    relevance_reason: str


class AdverseFinding(BaseModel):
    entity_id: str
    entity_name: str
    risk_score: int             # 0..100
    risk_category: Literal["low", "medium", "high", "critical"]
    hits: list[MediaHit]
    explanation: str            # human-readable, source-grounded rationale


# ── AGENTS_016 : Third-Party / Vendor Risk ───────────────────────────────────
class VendorRisk(BaseModel):
    vendor_id: str
    vendor_name: str
    exposure: Literal["none", "indirect", "direct"]
    risk_drivers: list[str]
    recommended_action: Literal["monitor", "engage", "diversify", "exit"]
    rationale: str


# ── Final artifact ───────────────────────────────────────────────────────────
class ReputationDossier(BaseModel):
    entity_id: str
    entity_name: str
    headline: str
    overall_risk: Literal["low", "medium", "high", "critical"]
    trend: TrendSignal
    adverse: AdverseFinding
    vendor_impacts: list[VendorRisk]
    generated_at: str
