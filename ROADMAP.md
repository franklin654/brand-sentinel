# Brand & Reputational Risk Intelligence — Feature Roadmap

> **Stack philosophy:** Every new capability uses an existing LangChain / LangGraph
> integration where one exists. Raw third-party clients are only used when no
> LangChain wrapper is available. This keeps the dependency graph shallow and
> all LLM + retrieval calls visible in LangSmith traces.

---

## Current State (Baseline)

| Component | Status |
|-----------|--------|
| 3-agent LangGraph pipeline (social → adverse → vendor → synthesise) | ✅ |
| VADER spike detection (ratio-based, CPU) | ✅ |
| SerpAPI live news via `google-search-results` | ✅ |
| File-upload capability (watchlist / vendor edges / social posts CSV/JSON) | ✅ |
| Streamlit 3-column streaming dashboard + dossier cards + graph viz | ✅ |
| LangSmith full-graph observability | ✅ |
| `langchain_openai.ChatOpenAI` → llama-server / vLLM (AMD Cloud) | ✅ |
| `sentence-transformers` semantic search over NEWS_CORPUS | ✅ |

---

## Phase 1 — RAG Foundation: Historical Incident Intelligence

**Problem:** `adverse_agent` scores risk 0–100 with no calibration anchor.
A contamination report gets 73 one run and 51 the next. There is no "why 73
vs 51" for a CRO to defend.

**Solution:** A persistent vector store of past brand crises (public record — FDA
enforcement actions, recall databases, SEC filings). Before the LLM scores a new
finding, retrieve the 3 most similar historical incidents and inject them as
calibration anchors into the system prompt.

**Personas:** CRO (defensible, reproducible scores), Brand Analyst (context-aware severity).

---

### LangChain integrations used

| Concern | LangChain class | Package |
|---------|----------------|---------|
| Embeddings | `langchain_huggingface.HuggingFaceEmbeddings` | `langchain-huggingface>=1.2.2` |
| Vector store | `langchain_community.vectorstores.Chroma` | `langchain-community>=0.4.2` + `chromadb>=1.5.9` |
| Document loading (corpus JSON) | `langchain_community.document_loaders.JSONLoader` | `langchain-community` |
| Text splitting | `langchain_text_splitters.RecursiveCharacterTextSplitter` | `langchain-text-splitters>=1.1.2` |
| Retriever | `.as_retriever(search_type="similarity", search_kwargs={"k": 3})` | built into Chroma integration |

`HuggingFaceEmbeddings` wraps the existing `BAAI/bge-small-en-v1.5` model already
downloaded by `embeddings.py` — no new model download.

---

### New files

```
brand_risk/rag.py                  # ChromaDB wrapper via LangChain — index, retrieve, purge
brand_risk/incident_corpus/        # Seed JSON: 50–100 past brand crises (public record)
scripts/build_index.py             # One-shot: embed + index the seed corpus into .chroma/
```

### Changes to existing files

| File | Change |
|------|--------|
| `brand_risk/agents.py` | `adverse_agent()`: call `rag.retrieve_similar_incidents(narrative, k=3)` before `chat_json`; inject returned docs into system prompt |
| `brand_risk/schemas.py` | `AdverseFinding`: add `calibration_anchors: list[str] = []` (shown in UI expander) |
| `requirements.txt` | Add `langchain-community>=0.4.2`, `langchain-huggingface>=1.2.2`, `langchain-text-splitters>=1.1.2`, `chromadb>=1.5.9` |

### `brand_risk/rag.py` — key signatures

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_incident_store() -> Chroma: ...          # lazy-init, persisted to .chroma/
def index_incidents(corpus_dir: str) -> None: ... # called by build_index.py
def retrieve_similar_incidents(narrative: str, k: int = 3) -> list[str]: ...
    # returns human-readable strings: "2019 Tyson recall → score 78, stock -12%"
```

### Incident corpus seed format

```json
{
  "incident_id": "tyson_recall_2019",
  "entity": "Tyson Foods",
  "kind": "brand",
  "crisis_type": "product_recall",
  "narrative": "Tyson recalled 12M lbs of chicken over listeria contamination",
  "risk_score": 78,
  "stock_impact_pct": -12,
  "resolution": "Voluntary recall; no fatalities; recovered in 6 weeks",
  "source": "FDA enforcement action database"
}
```

### What changes in the adverse_agent prompt

```
# Before Phase 1:
"You are an adverse-media analyst. Assign risk_score 0-100..."

# After Phase 1 (retrieved context injected at runtime):
"You are an adverse-media analyst. Assign risk_score 0-100...
Similar past cases for calibration:
- 2019 Tyson chicken recall → risk_score 78, stock impact -12%, recovered in 6 weeks
- 2021 Blue Bell listeria recall → risk_score 85, stock impact -25%, CEO resigned
Use these as scoring anchors. Return ONLY valid JSON."
```

### New requirements

```
langchain-community>=0.4.2
langchain-huggingface>=1.2.2
langchain-text-splitters>=1.1.2
chromadb>=1.5.9
```

### Verification

```bash
python scripts/build_index.py          # indexes seed corpus → .chroma/ local dir
python -m brand_risk.smoke             # must still pass — RAG is bypassed when Chroma absent
streamlit run app.py
# adverse column → "Calibration anchors" expander shows retrieved past cases
# risk scores are now consistent across runs for same narrative
```

---

## Phase 2 — Contract RAG: Supplier-Grounded Vendor Recommendations

**Problem:** `vendor_agent` recommends "diversify" or "exit" based on graph topology
and adverse narrative alone. A procurement manager needs "why can I / can't I exit
Nimbus Logistics right now?" answered from the actual contract.

**Solution:** Users upload supplier contracts (PDF/DOCX) via the Streamlit sidebar.
LangChain ingests, chunks, and indexes them per vendor into a separate Chroma
collection. Before the LLM reasons about a vendor, relevant contract clauses are
retrieved and appended to the user prompt. The LLM's `rationale` field then cites
actual clause numbers.

**Personas:** Procurement Manager (contract-aware exit decisions), CRO (liability-aware recommendations).

> **Prerequisite:** Phase 1 (reuses `rag.py` and the same Chroma + HuggingFaceEmbeddings setup).

---

### LangChain integrations used

| Concern | LangChain class | Package |
|---------|----------------|---------|
| PDF loading | `langchain_community.document_loaders.PyPDFLoader` | `langchain-community` + `pypdf>=6.13.2` |
| DOCX loading | `langchain_community.document_loaders.Docx2txtLoader` | `langchain-community` + `docx2txt` |
| Text splitting | `RecursiveCharacterTextSplitter` (reused from Phase 1) | `langchain-text-splitters` |
| Vector store | Second Chroma collection `"contracts"` (reused from Phase 1) | `langchain-community` |
| Retriever | `store.as_retriever(search_kwargs={"k": 3, "filter": {"vendor_id": v}})` | built-in |

Chroma's metadata filter `{"vendor_id": vendor_id}` ensures retrieval only returns
clauses from the specific vendor's contract, not another supplier's document.

---

### New files

```
brand_risk/doc_ingestor.py   # wraps PyPDFLoader + Docx2txtLoader + chunking + indexing
```

### Changes to existing files

| File | Change |
|------|--------|
| `brand_risk/rag.py` | Add `index_document(file_bytes, filename, vendor_id)` and `retrieve_contract_clauses(vendor_id, context, k=3)` |
| `brand_risk/agents.py` | `vendor_agent()`: call `rag.retrieve_contract_clauses(vendor_id, adverse_narrative, k=3)` before `chat_json`; append clauses to user prompt |
| `brand_risk/schemas.py` | `VendorRisk`: add `cited_clauses: list[str] = []` |
| `app.py` | Sidebar Phase 5 uploads section: "Supplier contracts (PDF/DOCX)" uploader per entity; on upload call `doc_ingestor.ingest(file, vendor_id)` |

### What changes in the vendor_agent prompt

```
# After Phase 2 (contract clauses injected at runtime):
"Vendor: Nimbus Logistics | Relation: logistics_supplier
Adverse finding: Acme Foods recall, risk_score=82
Relevant contract clauses:
- §8.2 Termination: 60-day written notice; £50k breakage fee unless regulatory shutdown.
- §12.1 Force Majeure: regulatory action by a government body waives the breakage fee.
Assess exposure and recommended action. Cite the clause number in your rationale."
```

### New requirements

```
pypdf>=6.13.2
docx2txt>=0.8
```

### Verification

```bash
# Upload supplier_nimbus_contract.pdf via sidebar
# Run pipeline → vendor panel shows "cited_clauses: ['§8.2 Termination...']"
# Downloaded dossier JSON includes cited_clauses per vendor
```

---

## Phase 3 — Risk Analytics: Trajectory, Attribution, and Source Credibility

**Problem:** Every run produces a point-in-time snapshot. The CRO wants "is Acme
Foods getting better or worse week-over-week?" and "what's actually driving this score —
social posts, news articles, or vendor exposure?"

**Solution:** Persist dossier scores to SQLite after every run. Add a risk attribution
model and source credibility weighting. Expose history charts in the dashboard.

**Personas:** CRO (trend briefings, board slides), Brand Analyst (root-cause attribution).

---

### LangChain integrations used

None for this phase — analytics and persistence are pure Python / SQLite.
LangSmith already captures per-run token costs and latency; no duplication needed.

---

### New files

```
brand_risk/store.py       # SQLite wrapper — save_run(), get_history(), get_latest()
brand_risk/analytics.py   # risk attribution model + source credibility lookup
```

### Changes to existing files

| File | Change |
|------|--------|
| `brand_risk/orchestrator.py` | `_synthesise()`: call `store.save_run(dossiers)` after building dossiers |
| `brand_risk/schemas.py` | `ReputationDossier`: add `risk_attribution: dict[str, float]` |
| `app.py` | Dossier cards: add attribution bar chart (social % / media % / vendor %); new "History" section below dossier cards with per-entity line chart |

### Risk attribution formula

```python
# brand_risk/analytics.py
def compute_attribution(signal, finding, vendor_impacts) -> dict[str, float]:
    social_w = abs(signal.sentiment_delta) * signal.volume
    media_w  = finding.risk_score * len([h for h in finding.hits if h.relevant])
    vendor_w = sum({"none": 0, "indirect": 1, "direct": 2}[v.exposure]
                   for v in vendor_impacts)
    total = social_w + media_w + vendor_w or 1.0
    return {
        "social_pct": round(social_w / total * 100, 1),
        "media_pct":  round(media_w  / total * 100, 1),
        "vendor_pct": round(vendor_w / total * 100, 1),
    }
```

### Source credibility weighting

```python
# brand_risk/analytics.py — extensible lookup, not hardcoded inline
CREDIBILITY_TIER: dict[str, float] = {
    "reuters.com": 1.0, "bbc.com": 1.0, "ft.com": 0.95,
    "bloomberg.com": 0.95, "wsj.com": 0.90, "theguardian.com": 0.85,
}
DEFAULT_CREDIBILITY = 0.5

def source_credibility(url: str) -> float:
    domain = url.split("/")[2].replace("www.", "")
    return CREDIBILITY_TIER.get(domain, DEFAULT_CREDIBILITY)
```

`adverse_agent` multiplies each MediaHit's contribution to `risk_score` by its
credibility weight before passing to the LLM.

### Verification

```bash
# Run the pipeline 3+ times
# Dossier cards show attribution bar (social / media / vendor)
# Below dossier section: "Risk history" line chart per entity
# SQLite file brand_risk.db exists in project root
```

---

## Phase 4 — Crisis Response Playbook RAG + Full UI Overhaul

**Goal:** Complete the RAG stack with response template retrieval; restructure the
app into a proper multi-page Streamlit application with drill-down, alert rules,
and board-ready PDF export.

**Personas:** CCO (pre-approved response templates), CRO (PDF for board minutes),
all personas (entity detail drill-down).

> **Prerequisite:** Phases 1, 3.

---

### 4a — Playbook RAG

**LangChain integrations used:** same as Phase 2 (`PyPDFLoader`, `Chroma` collection `"playbook"`).

| File | Change |
|------|--------|
| `brand_risk/rag.py` | Add `index_playbook(file_bytes, filename)` and `retrieve_response_template(crisis_type, k=1)` |
| `brand_risk/orchestrator.py` | `_synthesise()`: after dossier built, call `rag.retrieve_response_template(crisis_type)` → populate `suggested_response` |
| `brand_risk/schemas.py` | `ReputationDossier`: add `suggested_response: str = ""` |
| `app.py` | Sidebar: "Upload crisis playbook (PDF)"; dossier card: expandable "Suggested response" |

---

### 4b — Multi-page Streamlit app

Restructure using Streamlit's native `st.navigation` (1.x API):

```
app.py                    ← entry point: st.navigation([...]) + shared session state
pages/
  01_dashboard.py         ← current 3-column streaming view (moved verbatim)
  02_entity_detail.py     ← drill-down: posts, article hits, vendor subgraph, history
  03_alerts.py            ← alert rules CRUD: threshold sliders per entity, notification config
  04_documents.py         ← upload contracts + playbook; shows index status per vendor
```

---

### 4c — Entity detail page (`pages/02_entity_detail.py`)

- Entity selector dropdown → loads latest dossier from SQLite (`store.get_latest(entity_id)`)
- Full post list sorted by VADER score (worst first), each coloured by sentiment
- All MediaHit rows: credibility badge (Reuters / unknown), relevance flag ✓/✗, direct link
- Vendor subgraph centred on selected entity (filtered from full graph)
- Risk attribution pie chart (from Phase 3)
- Risk history line chart (from Phase 3)
- Suggested response (from Playbook RAG, if indexed)

---

### 4d — Alert rules panel (`pages/03_alerts.py`)

Editable config persisted to `brand_risk/alert_config.json`:

```json
{
  "e_acme":   {"neg_ratio": 0.25, "risk_score_ceiling": 60, "notify": ["slack"]},
  "e_rivera": {"neg_ratio": 0.30, "risk_score_ceiling": 70, "notify": ["email"]}
}
```

UI: per-entity rows with threshold sliders and notification channel checkboxes.
At runtime, `agents.py` reads alert config to override module-level `NEG_RATIO`.

---

### 4e — PDF export

```
brand_risk/pdf_export.py   # reportlab renders dossier cards to branded PDF
```

Sidebar: "Download board report (PDF)" button alongside existing JSON download.

### New requirements

```
reportlab>=4.2
```

### Verification

```bash
streamlit run app.py
# Entity Detail page → select Acme Foods → see post list, credibility badges
# Alerts page → lower Acme threshold → re-run → spike fires at lower ratio
# Upload playbook PDF → run pipeline → dossier shows "Suggested response" expander
# Download PDF → opens correctly, shows entity name, score, vendor table
```

---

## Phase 5 — Automation: Scheduled Monitoring + Multi-source Social Ingestion

**Goal:** Convert from on-demand to always-on. Alerts fire to Slack/email without
anyone pressing a button. Real social signals replace the synthetic stream.

**Personas:** Brand Analyst (real-time early warning), CRO (passive risk tracking).

---

### LangChain integrations used

| Concern | LangChain class | Package |
|---------|----------------|---------|
| Reddit posts | `langchain_community.document_loaders.RedditPostsLoader` | `langchain-community` + `praw>=7.7` |
| RSS feeds | `langchain_community.document_loaders.RSSFeedLoader` | `langchain-community` + `feedparser>=6.0` |

Both loaders return `list[Document]` with `.page_content` (post text) and `.metadata`.
A thin adapter converts each `Document` → `SocialPost` so `social_agent` is untouched.

---

### 5a — Scheduler

```
brand_risk/scheduler.py     # APScheduler wrapper — cron job calls the LangGraph pipeline
scripts/run_scheduler.py    # entry point: python scripts/run_scheduler.py
```

```python
# brand_risk/scheduler.py
from apscheduler.schedulers.background import BackgroundScheduler

def start(interval_minutes: int = 15) -> BackgroundScheduler:
    sched = BackgroundScheduler()
    sched.add_job(_run_pipeline_job, "interval", minutes=interval_minutes)
    sched.start()
    return sched
```

Results written to SQLite (Phase 3). Streamlit dashboard reads from store and
auto-refreshes via `st.rerun()` on a configurable poll interval.

---

### 5b — Alert delivery

```
brand_risk/notifier.py   # Slack webhook + SMTP email; reads alert_config.json (Phase 4)
```

```python
def notify(dossier: ReputationDossier, channels: list[str]) -> None:
    if "slack" in channels:  _post_slack_webhook(dossier)
    if "email" in channels:  _send_smtp(dossier)
```

No LangChain wrapper exists for Slack webhooks or SMTP — uses `requests` and `smtplib`
(stdlib) directly.

---

### 5c — Social connectors (LangChain document loaders)

```
brand_risk/social_connectors.py   # adapts LangChain Document loaders → list[SocialPost]
```

```python
# brand_risk/social_connectors.py
from langchain_community.document_loaders import RedditPostsLoader, RSSFeedLoader

def fetch_reddit_posts(entity: Entity, subreddits: list[str]) -> list[SocialPost]:
    loader = RedditPostsLoader(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent="brand-risk/1.0",
        search_queries=[entity.name] + entity.aliases,
        mode="subreddit",
        subreddits=subreddits,
        number_posts=50,
    )
    return [_doc_to_post(doc, entity.entity_id) for doc in loader.load()]

def fetch_rss_posts(entity: Entity, feed_urls: list[str]) -> list[SocialPost]:
    loader = RSSFeedLoader(urls=feed_urls)
    docs = loader.load()
    relevant = [d for d in docs if entity.name.lower() in d.page_content.lower()]
    return [_doc_to_post(doc, entity.entity_id) for doc in relevant]
```

`social_stream()` in `synthetic_data.py` gains a `connectors` parameter; when
connectors are configured, their output replaces the synthetic generator.

---

### 5d — Competitor benchmarking

`FlowState` gains a `peer_watchlist: list[Entity]` field. The pipeline runs against
peer entities in parallel (separate LangGraph invocations). `_synthesise()` computes
`peer_rank` and `industry_median_score` by comparing dossier scores across the full set.
Shown as a badge on each dossier card: "72/100 · Industry median: 41".

### New requirements

```
apscheduler>=3.11.2
praw>=7.7.1
feedparser>=6.0.11
```

### Verification

```bash
export REDDIT_CLIENT_ID=... REDDIT_CLIENT_SECRET=...
python scripts/run_scheduler.py &
# Wait for one tick (set interval_minutes=1 for testing)
# SQLite gains a new row; if score > ceiling → Slack webhook fires
# Entity Detail page auto-refreshes with latest run data
```

---

## Phase Summary

| Phase | Theme | Key deliverable | New files | Prereqs |
|-------|-------|-----------------|-----------|---------|
| **1** | RAG Foundation | Calibrated, reproducible risk scores | `rag.py`, `incident_corpus/`, `scripts/build_index.py` | — |
| **2** | Contract RAG | Vendor recommendations cite contract clauses | `doc_ingestor.py` | Phase 1 |
| **3** | Risk Analytics | Trend charts + attribution breakdown | `store.py`, `analytics.py` | — |
| **4** | UI Overhaul + Playbook RAG | Multi-page app, PDF export, alert rules | `pages/`, `pdf_export.py` | Phases 1, 3 |
| **5** | Automation | Always-on monitoring, Slack/email alerts, live social | `scheduler.py`, `notifier.py`, `social_connectors.py` | Phase 3 |

---

## Cumulative requirements additions

```
# Phase 1
langchain-community>=0.4.2
langchain-huggingface>=1.2.2
langchain-text-splitters>=1.1.2
chromadb>=1.5.9

# Phase 2
pypdf>=6.13.2
docx2txt>=0.8

# Phase 4
reportlab>=4.2

# Phase 5
apscheduler>=3.11.2
praw>=7.7.1
feedparser>=6.0.11
```

## Files that never change

`schemas.py` (only gains optional fields), `llm.py`, `embeddings.py`, `smoke.py`.
The LangGraph graph topology in `orchestrator.py` does not change until Phase 5's
parallel peer-benchmarking requires a second graph invocation.
