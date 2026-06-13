"""Parses user-uploaded files (CSV / JSON) into Pydantic models accepted by the pipeline.

Each parser is independent and returns either a list of models or raises a
ValueError with a human-readable message that Streamlit can display directly.
"""
from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime, timezone

import networkx as nx

from .schemas import Entity, SocialPost

logger = logging.getLogger(__name__)

_ENTITY_KINDS = {"brand", "executive", "vendor"}


def parse_watchlist(file) -> list[Entity]:
    """Parse an uploaded file into Entity objects.

    CSV expected columns: entity_id, name, kind, aliases
      - aliases: semicolon-separated alternative names (optional column)
    JSON: list of dicts matching Entity fields.

    Args:
        file: A Streamlit UploadedFile object (has .name and .read()).

    Returns:
        List of Entity objects.

    Raises:
        ValueError: On unrecognised format, missing required columns, or
            invalid kind values.
    """
    raw = file.read()
    name = file.name.lower()

    if name.endswith(".json"):
        rows = json.loads(raw)
        entities = [Entity.model_validate(r) for r in rows]
    elif name.endswith(".csv"):
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
        _require_columns(reader.fieldnames or [], ["entity_id", "name", "kind"])
        entities = []
        for row in reader:
            kind = row["kind"].strip().lower()
            if kind not in _ENTITY_KINDS:
                raise ValueError(f"Invalid kind '{kind}'. Must be one of {_ENTITY_KINDS}.")
            raw_aliases = row.get("aliases", "")
            aliases = [a.strip() for a in raw_aliases.split(";") if a.strip()]
            entities.append(Entity(
                entity_id=row["entity_id"].strip(),
                name=row["name"].strip(),
                kind=kind,
                aliases=aliases,
            ))
    else:
        raise ValueError(f"Unsupported format '{file.name}'. Upload a .csv or .json file.")

    if not entities:
        raise ValueError("Watchlist file is empty — add at least one entity row.")
    logger.info("Parsed %d entities from upload", len(entities))
    return entities


def parse_vendor_edges(file) -> list[tuple[str, str, str]]:
    """Parse an uploaded CSV into vendor edge tuples.

    CSV expected columns: source_id, target_id, relation

    Args:
        file: A Streamlit UploadedFile object.

    Returns:
        List of (source_id, target_id, relation) tuples.

    Raises:
        ValueError: On missing required columns or empty file.
    """
    raw = file.read().decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(raw))
    _require_columns(reader.fieldnames or [], ["source_id", "target_id", "relation"])
    edges = [
        (row["source_id"].strip(), row["target_id"].strip(), row["relation"].strip())
        for row in reader
        if row["source_id"].strip()
    ]
    if not edges:
        raise ValueError("Vendor edges file is empty — add at least one row.")
    logger.info("Parsed %d vendor edges from upload", len(edges))
    return edges


def parse_social_posts(file) -> list[SocialPost]:
    """Parse an uploaded file into SocialPost objects.

    CSV expected columns: entity_id, text, post_id (optional), timestamp (optional)
    JSON: list of dicts matching SocialPost fields.

    Missing post_id values are auto-generated.
    Missing timestamp defaults to the current UTC time.

    Args:
        file: A Streamlit UploadedFile object.

    Returns:
        List of SocialPost objects.

    Raises:
        ValueError: On unrecognised format, missing required columns, or empty file.
    """
    raw = file.read()
    name = file.name.lower()
    now_iso = datetime.now(timezone.utc).isoformat()

    if name.endswith(".json"):
        rows = json.loads(raw)
        posts = []
        for i, r in enumerate(rows):
            r.setdefault("post_id", f"up{i:05d}")
            r.setdefault("timestamp", now_iso)
            posts.append(SocialPost.model_validate(r))
    elif name.endswith(".csv"):
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
        _require_columns(reader.fieldnames or [], ["entity_id", "text"])
        posts = []
        for i, row in enumerate(reader):
            posts.append(SocialPost(
                post_id=row.get("post_id", "").strip() or f"up{i:05d}",
                entity_id=row["entity_id"].strip(),
                text=row["text"].strip(),
                timestamp=row.get("timestamp", "").strip() or now_iso,
            ))
    else:
        raise ValueError(f"Unsupported format '{file.name}'. Upload a .csv or .json file.")

    if not posts:
        raise ValueError("Social posts file is empty — add at least one row.")
    logger.info("Parsed %d social posts from upload", len(posts))
    return posts


def build_vendor_graph(
    entities: list[Entity],
    edges: list[tuple[str, str, str]],
) -> nx.Graph:
    """Build a NetworkX graph from a list of entities and edge tuples.

    Args:
        entities: List of Entity objects to register as nodes.
        edges: List of (source_id, target_id, relation) tuples.

    Returns:
        NetworkX Graph with node attrs ``name`` and ``kind``, edge attr ``relation``.
    """
    g = nx.Graph()
    for e in entities:
        g.add_node(e.entity_id, name=e.name, kind=e.kind)
    for src, tgt, rel in edges:
        g.add_edge(src, tgt, relation=rel)
    return g


def _require_columns(fieldnames: list[str], required: list[str]) -> None:
    """Raise ValueError listing any missing required column names."""
    missing = [c for c in required if c not in fieldnames]
    if missing:
        raise ValueError(f"CSV is missing required column(s): {missing}")
