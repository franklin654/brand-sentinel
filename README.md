# Brand & Reputational Risk Intelligence

**TCS-AMD AI Hackathon · Track 1 (Agents) · Flow: AGENTS_039 → AGENTS_001 → AGENTS_016**

Autonomous multi-agent pipeline that monitors brands, executives, and vendor ecosystems
for reputational risk. Detects social sentiment spikes, screens adverse media, traces
supplier contagion, and produces structured intelligence dossiers — all without human
intervention.

```
watchlist ─▶ AGENTS_039 ─(spike?)─▶ AGENTS_001 ─(material?)─▶ AGENTS_016 ─▶ dossier
            social signal           adverse media             vendor risk
```

---

## Stack

| Layer | Technology |
|-------|-----------|
| Agent orchestration | LangGraph 1.2.5 — StateGraph with conditional edges |
| LLM inference | Qwen2.5-72B-Instruct via vLLM on AMD Instinct MI300X |
| Sentiment scoring | VADER (CPU — no GPU cost per post) |
| Data validation | Pydantic v2 — every agent output is a typed model |
| Parallel LLM calls | `concurrent.futures.ThreadPoolExecutor` |
| RAG | ChromaDB + BAAI/bge-small-en-v1.5 (contracts + playbook) |
| Vendor graph | NetworkX ego-graph traversal |
| Storage | SQLite — full run history + dossier JSON |
| Front-end | Streamlit 8-page app |
| PDF export | ReportLab |
| Alerts | Slack webhook + SMTP email, per-entity thresholds |
| Scheduler | APScheduler background thread |

---

## Setup

### 1. Start vLLM on AMD MI300X

```bash
vllm serve Qwen/Qwen2.5-72B-Instruct \
  --port 8000 \
  --guided-decoding-backend outlines \
  --max-model-len 32768 \
  --tensor-parallel-size 2
```

### 2. Configure environment

```bash
cp .env.example .env
```

Minimum required:

```env
# Local vLLM:
LLM_BASE_URL=http://localhost:8000/v1
LLM_MODEL=Qwen/Qwen2.5-72B-Instruct

# Cloud vLLM via Jupyter server proxy (AMD MI300X):
# LLM_BASE_URL=https://<jupyter-host>/proxy/8000/v1
# LLM_MODEL=Qwen/Qwen2.5-72B-Instruct
# JUPYTER_TOKEN=<your-jupyter-server-token>
```

### 3. Install dependencies

```bash
conda activate cuda_env
pip install -r requirements.txt
```

### 4. Smoke test (no LLM required)

```bash
python -m brand_risk.smoke
```

Must print `OK — 1 spike(s) detected: Acme Foods`. Verifies VADER spike detection,
vendor graph construction, and news retrieval with zero GPU or network calls.

### 5. Run the app

```bash
streamlit run app.py
```

---

## Demo walkthrough

The app starts with no data loaded. Upload the example files from `examples/`:

| File | Contents |
|------|----------|
| `examples/watchlist.csv` | Apple, Tim Cook, TSMC, Samsung, Foxconn |
| `examples/vendor_edges.csv` | Tech supply chain edges |
| `examples/social_posts.csv` | 60 posts from real 2024–2026 brand risk events |

1. **Documents page** — upload all three CSVs
2. **Dashboard** — click "Run monitoring cycle"
3. **Entity Detail** — drill into any entity for score history and PDF export
4. **Chat** — ask natural language questions over the dossiers

---

## Architecture

### The three agents

**AGENTS_039 — Social signal (`social_agent`)**
Scores every post with VADER compound sentiment on CPU. Fires a `TrendSignal` when
≥30% of posts for an entity are strongly negative (compound < −0.20) and volume ≥ 3.
Calls the LLM once per spike to cluster the narrative into a one-line summary.

**AGENTS_001 — Adverse media (`adverse_agent`)**
Only runs when AGENTS_039 detected a spike (LangGraph conditional edge). Searches
news via SerpAPI (live) or RSS (no API key). Runs LLM assessments in parallel via
`ThreadPoolExecutor`. Disambiguates false entity matches and produces a 0–100 risk
score with a source-grounded explanation.

**AGENTS_016 — Vendor risk (`vendor_agent`)**
Only runs when AGENTS_001 found a medium/high/critical finding. Traverses the NetworkX
supplier graph to find neighbouring vendor nodes. Assesses each vendor relationship in
parallel and recommends monitor/engage/diversify/exit based on exposure level and
retrieved contract clauses.

### LangGraph flow

Two conditional edges gate the expensive LLM calls:

```
social ──(no spike)──────────────────────────────▶ END
       ──(spike detected)──▶ adverse ──(all low)──▶ synthesise ──▶ END
                                     ──(material)──▶ vendor ────▶ synthesise ──▶ END
```

An entity with no negative social signal never reaches the LLM. This keeps the
pipeline cheap enough to run continuously.

### Data contracts

Every agent boundary uses a Pydantic v2 model:

```
Entity              → who we monitor (brand / executive / vendor)
SocialPost          → one post with VADER sentiment score
TrendSignal         → a detected spike for one entity
AdverseFinding      → adverse media result with risk score 0–100
VendorRisk          → one vendor relationship assessment
ReputationDossier   → final artifact combining all three agents
```

---

## App pages

| Page | Purpose |
|------|---------|
| Dashboard | Streaming pipeline view, dossier cards, vendor graph, PDF/JSON export |
| Entity Detail | Single-entity focused scan, risk history chart, heatmap, PDF report |
| Alert Rules | Per-entity spike thresholds, Slack/email notification config |
| Documents | Upload watchlist/edges/posts CSV, contracts, playbook; RSS feed config |
| Chat | Natural language queries over current dossiers or SQLite history |
| Run History | All past pipeline runs with Δ delta column and dossier drill-down |
| Settings | Live edit of LLM endpoint, API keys, and alert credentials |
| Compare | Side-by-side risk comparison for any two entities |

---

## Project layout

```
brand_risk/
  schemas.py            Pydantic contracts between agents
  llm.py                ChatOpenAI singleton with reset_llm() for runtime reconfiguration
  agents.py             AGENTS_039 / 001 / 016 (parallel LLM calls via ThreadPoolExecutor)
  orchestrator.py       LangGraph StateGraph with two conditional edges
  store.py              SQLite persistence — run history, dossier JSON, delta queries
  analytics.py          Risk attribution model + source credibility weighting
  rag.py                ChromaDB RAG — historical incident retrieval
  rag_contracts.py      Contract clause retrieval (per vendor_id filter)
  rag_playbook.py       Crisis response template retrieval
  doc_ingestor.py       PDF/DOCX ingestion into ChromaDB
  social_connectors.py  RSS feed adapter → list[SocialPost]
  upload_parser.py      CSV/JSON upload parsing for watchlist, edges, posts
  watchlist_store.py    Persistent watchlist override (JSON at project root)
  rss_store.py          RSS config persistence (JSON at project root)
  alert_config_loader.py Alert threshold config loader
  scheduler.py          APScheduler background pipeline runner
  notifier.py           Slack webhook + SMTP email alert delivery
  pdf_export.py         ReportLab board report generator
  embeddings.py         HuggingFace embedding model wrapper
  smoke.py              Deterministic spine test — no LLM, no network

pages/
  01_dashboard.py       Streaming pipeline dashboard
  02_entity_detail.py   Entity drill-down
  03_alerts.py          Alert rule configuration
  04_documents.py       Data upload hub
  05_chat.py            Natural language chat interface
  06_history.py         Run history browser
  07_settings.py        Environment and LLM settings
  08_compare.py         Entity comparison view

examples/
  watchlist.csv         5-entity tech supply chain demo watchlist
  vendor_edges.csv      Supply chain graph edges
  social_posts.csv      60 posts from real 2024–2026 brand risk events

app.py                  Streamlit entry point + session state initialisation
```

---

## Configuration

All runtime config lives in environment variables (`.env`). Settings can also be
edited live on the **Settings page** — the LLM client rebuilds automatically on save.

| Variable | Purpose |
|----------|---------|
| `LLM_BASE_URL` | vLLM / llama-server endpoint |
| `LLM_MODEL` | Model name (any OpenAI-compatible model) |
| `JUPYTER_TOKEN` | Auth token for cloud vLLM behind Jupyter proxy |
| `SERP_API_KEY` | Enables live Google News via SerpAPI |
| `SLACK_WEBHOOK_URL` | Slack incoming webhook for alert delivery |
| `SMTP_*` | SMTP credentials for email alerts |
| `REDDIT_CLIENT_ID/SECRET` | Reddit API credentials for live social ingestion |
| `MONITOR_INTERVAL` | Scheduler run interval in minutes (default: 15) |

---

## Key design decisions

| Decision | Reason |
|----------|--------|
| VADER for per-post scoring, LLM only for reasoning | Scoring hundreds of posts with an LLM is slow and expensive. VADER runs in microseconds on CPU. The LLM is reserved for narrative clustering, disambiguation, and risk explanation. |
| Two conditional edges in LangGraph | Entities with no spike never reach the LLM. Entities with only low adverse findings skip the vendor agent. This gating makes the pipeline cheap enough to run continuously at scale. |
| ThreadPoolExecutor for parallel LLM calls | `adverse_agent` and `vendor_agent` submit all per-entity assessments concurrently. For a 5-entity watchlist this cuts LLM wall-clock time by ~4x. |
| Pydantic v2 at every agent boundary | Forces structured LLM output. A malformed reply triggers Pydantic validation which re-prompts with the error rather than crashing. |
| `--guided-decoding-backend outlines` on vLLM | Enforces JSON schema at the token level for `chat_json` calls. Has zero effect on free-form chat calls. |
| All runtime JSON configs at project root | `alert_config.json`, `watchlist_override.json`, `rss_config.json` live at the root, not inside the package, so they cannot be accidentally committed. All three are gitignored. |
