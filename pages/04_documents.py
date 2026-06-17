"""Documents — upload hub for watchlist, vendor data, contracts, and playbook.

Uploaded watchlist/graph/posts are stored in st.session_state so the Dashboard
picks them up automatically on the next run cycle.
"""
from __future__ import annotations

import streamlit as st

from brand_risk import synthetic_data as data
from brand_risk.watchlist_store import clear_watchlist
from pages._helpers import render_manual_entry, render_entity_manager, render_edge_manager

st.title("Documents")
st.caption("Upload data files and supplier contracts. Changes apply to the next pipeline run.")

# ── Pipeline data uploads ─────────────────────────────────────────────────────
st.subheader("Pipeline data")
st.caption("Leave any slot empty to keep using the built-in synthetic data for that input.")

col_wl, col_eg, col_sp = st.columns(3)

wl_file = col_wl.file_uploader(
    "Watchlist (CSV or JSON)",
    type=["csv", "json"],
    key="wl_upload",
    help="CSV columns: entity_id, name, kind, aliases (semicolon-separated)",
)
eg_file = col_eg.file_uploader(
    "Vendor edges (CSV)",
    type=["csv"],
    key="eg_upload",
    help="CSV columns: source_id, target_id, relation",
)
sp_file = col_sp.file_uploader(
    "Social posts (CSV or JSON)",
    type=["csv", "json"],
    key="sp_upload",
    help="CSV columns: entity_id, text, post_id (optional), timestamp (optional)",
)

if wl_file:
    try:
        from brand_risk.upload_parser import parse_watchlist
        watchlist = parse_watchlist(wl_file)
        st.session_state.uploaded_watchlist = watchlist
        col_wl.success(f"{len(watchlist)} entities loaded")
    except Exception as exc:
        col_wl.error(f"Parse error: {exc}")

if eg_file:
    try:
        from brand_risk.upload_parser import parse_vendor_edges, build_vendor_graph
        wl = list(st.session_state.get("uploaded_watchlist") or [])
        edges = parse_vendor_edges(eg_file)
        g = build_vendor_graph(wl, edges)
        st.session_state.uploaded_graph = g
        col_eg.success(f"{len(edges)} edges loaded")
    except Exception as exc:
        col_eg.error(f"Parse error: {exc}")

if sp_file:
    try:
        from brand_risk.upload_parser import parse_social_posts
        posts = parse_social_posts(sp_file)
        st.session_state.uploaded_posts = posts
        col_sp.success(f"{len(posts)} posts loaded")
    except Exception as exc:
        col_sp.error(f"Parse error: {exc}")

with st.expander("Format guide"):
    st.markdown("""
**Watchlist CSV**
```
entity_id,name,kind,aliases
e_acme,Acme Foods,brand,Acme;AcmeFoods
```

**Vendor edges CSV**
```
source_id,target_id,relation
e_acme,e_nimbus,logistics_supplier
```

**Social posts CSV**
```
entity_id,text,post_id,timestamp
e_acme,"Awful recall, unacceptable",p001,2026-06-14T09:00:00
```
""")

render_manual_entry([])

# ── Manage current data ───────────────────────────────────────────────────────
st.divider()
st.subheader("Manage current data")
st.caption("Delete entities or edges from the active session. Changes apply immediately.")
tab_wl_mgr, tab_eg_mgr = st.tabs(["Watchlist entities", "Vendor edges"])
with tab_wl_mgr:
    render_entity_manager([])
    wl_now = list(st.session_state.get("uploaded_watchlist") or [])
    import csv, io as _io
    _buf = _io.StringIO()
    _writer = csv.DictWriter(_buf, fieldnames=["entity_id", "name", "kind", "aliases"])
    _writer.writeheader()
    for _e in wl_now:
        _writer.writerow({
            "entity_id": _e.entity_id, "name": _e.name,
            "kind": _e.kind, "aliases": ";".join(_e.aliases),
        })
    st.download_button(
        "Export watchlist (CSV)", data=_buf.getvalue().encode(),
        file_name="watchlist.csv", mime="text/csv",
    )
    if st.button("Clear watchlist"):
        clear_watchlist()
        st.session_state.pop("uploaded_watchlist", None)
        st.success("Watchlist cleared. Upload a new CSV to continue.")
with tab_eg_mgr:
    render_edge_manager([])

# ── Supplier contracts ────────────────────────────────────────────────────────
st.divider()
st.subheader("Supplier contracts")
st.caption("Contracts ground vendor recommendations in actual clause language.")

_wl_vendors = list(st.session_state.get("uploaded_watchlist") or [])
vendor_entities = [e for e in _wl_vendors if e.kind == "vendor"]
if not vendor_entities:
    st.info("No vendor entities — upload a watchlist with vendor entries on this page first.")
else:
    selected_vendor = st.selectbox(
        "Vendor",
        options=[e.entity_id for e in vendor_entities],
        format_func=lambda eid: next(
            (e.name for e in vendor_entities if e.entity_id == eid), eid
        ),
        key="contract_vendor_sel",
    )
contract_file = st.file_uploader(
    "Contract (PDF or DOCX)",
    type=["pdf", "docx", "doc"],
    key="contract_upload",
)
if contract_file is not None and vendor_entities:
    try:
        from brand_risk.doc_ingestor import ingest
        n = ingest(contract_file.read(), contract_file.name, selected_vendor)
        vendor_label = next(
            (e.name for e in vendor_entities if e.entity_id == selected_vendor),
            selected_vendor,
        )
        st.success(f"{n} clauses indexed for **{vendor_label}**")
    except Exception as exc:
        st.error(f"Contract parse error: {exc}")

# ── Crisis response playbook ──────────────────────────────────────────────────
st.divider()
st.subheader("Crisis response playbook")
st.caption(
    "Upload a PDF/DOCX playbook. The pipeline will retrieve the most relevant "
    "response template automatically when a crisis fires."
)

playbook_file = st.file_uploader(
    "Playbook (PDF or DOCX)",
    type=["pdf", "docx", "doc"],
    key="playbook_upload",
)
if playbook_file is not None:
    try:
        from brand_risk.doc_ingestor import ingest_playbook
        n = ingest_playbook(playbook_file.read(), playbook_file.name)
        st.success(f"{n} playbook chunks indexed — response templates now active.")
    except Exception as exc:
        st.error(f"Playbook parse error: {exc}")

# ── Live social feeds (RSS) ───────────────────────────────────────────────────
st.divider()
st.subheader("Live social feeds (RSS)")
st.caption(
    "Paste one RSS feed URL per line for each entity. The pipeline fetches recent "
    "articles, filters for items mentioning the entity, and scores them as social posts. "
    "No API key required — any public RSS URL works (BBC, Reuters, The Guardian, etc.)."
)

_wl_rss = list(st.session_state.get("uploaded_watchlist") or [])
rss_config: dict = st.session_state.get("rss_config", {})
if not _wl_rss:
    st.info("No entities — upload a watchlist on this page first to configure RSS feeds.")
else:
    for entity in _wl_rss:
        urls_str = st.text_area(
            entity.name,
            value="\n".join(rss_config.get(entity.entity_id, [])),
            placeholder="https://feeds.bbci.co.uk/news/business/rss.xml",
            height=68,
            key=f"rss_{entity.entity_id}",
        )
        rss_config[entity.entity_id] = [u.strip() for u in urls_str.splitlines() if u.strip()]

if st.button("Fetch RSS posts", help="Pull articles from configured feeds into the session"):
    if not _wl_rss:
        st.warning("Upload a watchlist first before fetching RSS posts.")
    else:
        try:
            from brand_risk.social_connectors import fetch_all
            from brand_risk.rss_store import save_rss
            fetched = fetch_all(_wl_rss, rss_config)
            st.session_state.rss_config = rss_config
            save_rss(rss_config)
            if fetched:
                existing = list(st.session_state.get("uploaded_posts") or [])
                st.session_state.uploaded_posts = existing + fetched
                st.success(f"Fetched {len(fetched)} post(s) from RSS feeds — ready for next pipeline run.")
            else:
                st.info("No articles matching your entities found in the configured feeds.")
        except Exception as exc:
            st.error(f"RSS fetch error: {exc}")

# ── Document search (RAG) ─────────────────────────────────────────────────────
st.divider()
st.subheader("Search documents")
st.caption("Query indexed contracts and playbook chunks using semantic similarity.")

search_query = st.text_input("Search query", placeholder="termination clause 30-day notice")
col_src, col_k = st.columns([3, 1])
search_source = col_src.radio("Search in", ["Contracts", "Playbook", "Both"], horizontal=True)
top_k = col_k.number_input("Results", min_value=1, max_value=20, value=5)

if st.button("Search", disabled=not search_query):
    from brand_risk.rag_contracts import search_contracts
    from brand_risk.rag_playbook import search_playbook

    contract_hits: list[dict] = []
    playbook_hits: list[dict] = []

    if search_source in ("Contracts", "Both"):
        contract_hits = search_contracts(search_query, k=int(top_k))
    if search_source in ("Playbook", "Both"):
        playbook_hits = search_playbook(search_query, k=int(top_k))

    if not contract_hits and not playbook_hits:
        st.info("No results — upload contracts or a playbook first.")
    for hit in contract_hits:
        with st.container(border=True):
            st.caption(f"CONTRACT · vendor: `{hit['vendor_id']}` · {hit['source']}")
            st.write(hit["text"])
    for hit in playbook_hits:
        with st.container(border=True):
            st.caption(f"PLAYBOOK · {hit['source']}")
            st.write(hit["text"])
