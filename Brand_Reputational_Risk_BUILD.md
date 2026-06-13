# Brand & Reputational Risk Intelligence — Build Brief

**TCS-AMD AI Hackathon · Track 1 (Agents) · Combined Flow #2 (49/60)**
**Pipeline:** `AGENTS_039` (Social Media Insights) → `AGENTS_001` (Adverse Media) → `AGENTS_016` (Third-Party / Vendor Risk)

> **Status as of 2026-06-13:** All four build milestones complete. Pipeline runs
> end-to-end locally (Qwen3-14B via llama-server). Ready for AMD Dev Cloud
> deployment with vLLM on MI300X.

---

# PART 1 — CONTEXT & ANALYSIS

## 1.1 What we are building

An entity-centric pipeline that **detects** a negative social signal about a
monitored brand/executive/vendor, **confirms** it against news with explainable
risk scoring, and **traces** the exposure to the vendors and partners it
touches — producing a single reputation dossier.

```
watchlist ─▶ AGENTS_039 ─(spike?)─▶ AGENTS_001 ─(material?)─▶ AGENTS_016 ─▶ dossier
            social signal          adverse media            vendor risk
```

## 1.2 Why it ranked #2

| Dimension | Score | Note |
|-----------|-------|------|
| **Business value** | **10/10** | Reputation/social-listening (~$8.7B) + third-party risk (~$8B); highest BV of any flow |
| Demo-ability | 9/10 | Spike → adverse hit → vendor impact is a clear, dramatic story |
| Technical feasibility | 7/10 | Entity disambiguation is the one non-trivial part |
| Scope clarity | 7/10 | Three agents, one entity-centric data model |
| AMD platform fit | 7/10 | LLM reasoning + optional GPU sentiment model |
| Synergy bonus | 9/10 | Shared entity model; tight detect→confirm→trace pipeline |

## 1.3 Target users / stakeholders

CMO / communications, brand protection, trust & safety, and procurement risk
functions. The buyer cares about **catching a reputational crisis early** and
**knowing which supplier relationships it threatens**.

## 1.4 Mapped hackathon challenges

- `AGENTS_039` — Social Media insights and triggers — trend prediction
- `AGENTS_001` — Adverse Media / Negative News Screening (Explainable)
- `AGENTS_016` — Third Party & Vendor Risk Management

---

# PART 2 — BUILD SPECIFICATION

## 2.1 Core design principles

1. **VADER for per-post sentiment, LLM only for reasoning.** VADER detects the
   spike instantly on CPU; the LLM is spent only on narrative clustering, entity
   disambiguation, risk explanation, and vendor reasoning.
   Upgrade path: `cardiffnlp/twitter-roberta-base-sentiment` on ROCm — same call site.
2. **Conditional gating.** An entity only triggers an expensive adverse-media
   screen if a sentiment spike was detected; a vendor screen only runs if the
   adverse finding was material (not "low"). Modelled with LangGraph conditional edges.
3. **Validated contracts.** Every agent returns a Pydantic-validated object; the
   LLM client requests JSON and re-prompts on validation failure.
4. **Planted scenarios + local news corpus for demo safety.** The *data* is
   planted; the *reasoning* is real. `search_news()` keeps a stable signature so a
   live API (ddgs/SerpAPI) swaps in by replacing one function.
5. **Ratio-based spike detection, not mean-based.** Neutral posts dilute a mean
   and cause spikes to fail silently. Use `negatives/total >= NEG_RATIO` with an
   absolute floor `MIN_VOLUME`.

## 2.2 Target file tree

```
brand_risk/               ← project root
  app.py                  # Streamlit entry point (root-level; avoids import issues)
  requirements.txt
  Brand_Reputational_Risk_BUILD.md
  README.md
  brand_risk/             ← Python package (pure library code)
    __init__.py
    schemas.py            # Pydantic v2 contracts between agents
    llm.py               # LangChain ChatOpenAI client (vLLM / llama-server)
    embeddings.py         # sentence-transformers semantic search (NEW)
    synthetic_data.py     # watchlist, social stream, news corpus, vendor graph
    agents.py             # AGENTS_039 / 001 / 016
    orchestrator.py       # LangGraph 1.x state machine with conditional edges
    smoke.py              # deterministic spine test (no LLM / GPU)
```

> **Note on `app.py` location:** Streamlit adds the *script's directory* to
> `sys.path`, not the project root. Keeping `app.py` inside the package causes
> `ModuleNotFoundError`. The entry point lives at the project root so the CWD
> (which contains the `brand_risk/` package) is on the path automatically.

## 2.3 Module contracts

### `schemas.py` — unchanged from spec
Pydantic v2 models:

```python
class Entity(BaseModel):
    entity_id: str
    name: str
    kind: Literal["brand", "executive", "vendor"]
    aliases: list[str] = []

class SocialPost(BaseModel):
    post_id: str; entity_id: str; text: str; timestamp: str
    sentiment: float = 0.0          # -1..+1, filled by social_agent

class TrendSignal(BaseModel):       # output of AGENTS_039
    entity_id: str; entity_name: str
    sentiment_delta: float; volume: int
    narrative_cluster: str          # one-line LLM summary of the complaint
    sample_posts: list[str]; detected_at: str

class MediaHit(BaseModel):
    title: str; url: str; snippet: str
    relevant: bool; relevance_reason: str

class AdverseFinding(BaseModel):    # output of AGENTS_001
    entity_id: str; entity_name: str
    risk_score: int                 # 0..100
    risk_category: Literal["low", "medium", "high", "critical"]
    hits: list[MediaHit]; explanation: str

class VendorRisk(BaseModel):        # output of AGENTS_016
    vendor_id: str; vendor_name: str
    exposure: Literal["none", "indirect", "direct"]
    risk_drivers: list[str]
    recommended_action: Literal["monitor", "engage", "diversify", "exit"]
    rationale: str

class ReputationDossier(BaseModel): # final artifact
    entity_id: str; entity_name: str; headline: str
    overall_risk: Literal["low", "medium", "high", "critical"]
    trend: TrendSignal; adverse: AdverseFinding
    vendor_impacts: list[VendorRisk]; generated_at: str
```

### `llm.py` — **updated: LangChain + LangSmith**

**Replaces the Ollama client** with `langchain_openai.ChatOpenAI`, which speaks
the OpenAI-compatible `/v1/chat/completions` API served by both vLLM (AMD Cloud)
and llama-server (local dev). No code changes needed when switching environments
— only env vars change.

Configuration:
```
LLM_BASE_URL  (default: http://localhost:8080/v1)  # llama-server local
              (AMD Cloud: http://localhost:8000/v1) # vLLM
LLM_MODEL     (default: bartowski/Qwen_Qwen3-14B-GGUF:Q4_K_M)
```

Public interface unchanged: `chat(system, user, temperature)` and
`chat_json(system, user, schema, retries=2)`.

**Why manual JSON retry instead of `with_structured_output`:**
LangChain's `with_structured_output` is confirmed broken for nested Pydantic
models with local Llama/Qwen models (langchain-ai/langchain#28412). The manual
strip-fence → `model_validate_json` → re-prompt loop is more robust.

**LangSmith observability is zero-code:** set `LANGCHAIN_TRACING_V2=true` and
`LANGCHAIN_API_KEY`. Every `invoke()` call and the full LangGraph run are
captured automatically as a trace tree with per-node token counts and latencies.

### `embeddings.py` — **new module**

Semantic similarity via `sentence-transformers`. Lazy-loads on first use so
`smoke.py` (zero GPU/ML deps) is unaffected.

```
EMBED_MODEL (default: BAAI/bge-small-en-v1.5)
            33M params, <200 MB, fast on CPU and ROCm GPU
```

Device selected automatically: `cuda` if `torch.cuda.is_available()` (ROCm on
MI300X), else `cpu`. Exposes `embed()`, `top_k_similar()`, `cosine_scores()`.

### `synthetic_data.py` — **updated: semantic `search_news()`**

`search_news(entity_name, aliases)` now uses `cosine_scores()` from
`embeddings.py` to rank articles by semantic similarity, replacing substring
matching. Threshold: `SIM_THRESHOLD = 0.25`. Falls back to substring matching
on `ImportError` (keeps `smoke.py` dependency-free).

Signature is **unchanged** — a live API (ddgs/SerpAPI) swaps in by replacing
this one function.

**Planted scenarios (2 crises):**
- `e_acme` — Acme Foods product recall / contamination (4 negative posts)
- `e_rivera` — Dana Rivera (CEO) executive controversy (4 negative posts, reworded
  for VADER detectability — original phrasing scored near 0.0)

### `agents.py` — **updated: NEG_POST threshold**

- **`social_agent`** (AGENTS_039): VADER ratio spike detection.
  `NEG_POST = -0.20` (was -0.30 in spec). The lower threshold is required to
  fire the executive scenario; VADER scores indirect language like "tone-deaf" and
  "dodging" close to 0.0 even though they are semantically negative.
  Knobs: `NEG_POST = -0.20`, `NEG_RATIO = 0.30`, `MIN_VOLUME = 3`.
- **`adverse_agent`** (AGENTS_001): unchanged — `search_news()` + `chat_json(AdverseFinding)`.
- **`vendor_agent`** (AGENTS_016): unchanged — graph traversal + `chat_json(VendorRisk)`, skips "low" findings.

### `orchestrator.py` — **bug fix: vendor risk scoping**

`_synthesise()` previously put ALL vendor risks into every dossier. Fixed to
filter by graph neighbourhood so each entity's dossier only contains vendor risks
for its own supply-chain neighbours:

```python
g = state["graph"]
neighbor_ids = set(g.neighbors(sig.entity_id))
impacts = [v for v in state.get("vendor_risks", []) if v.vendor_id in neighbor_ids]
```

LangGraph 1.x: `StateGraph` / `add_conditional_edges` / `END` API is
backward-compatible. No forced rewrites; `graph.stream(stream_mode="updates")`
used for synchronous progressive streaming to Streamlit.

### `app.py` — **updated: location + streaming + richer UI**

Entry point moved to project root (see §2.2 note).

Features:
- **Progressive rendering** via `graph.stream(stream_mode="updates")` — each
  column lights up the moment its agent node completes.
- **3-column layout** (Social signal / Adverse media / Vendor impact) with
  `st.empty()` placeholders updated live.
- **Dossier summary cards** with risk score metric, headline, vendor impact list,
  and timestamps.
- **Vendor graph visualisation** via NetworkX + matplotlib (`st.pyplot()`); nodes
  coloured red if flagged by a VendorRisk entry.
- **JSON download button** — full dossier export via `st.download_button`.
- **LangSmith indicator** — shows active project name if `LANGCHAIN_TRACING_V2=true`.

Run: `streamlit run app.py` from the project root.

### `smoke.py` — **updated: imports thresholds from `agents.py`**

Previously had hardcoded `NEG_POST = -0.30`. Now imports `NEG_POST, NEG_RATIO,
MIN_VOLUME` from `agents.py` so thresholds never drift between test and runtime.
Both Acme Foods and Dana Rivera spikes fire and assert True.

## 2.4 Tech stack

| Layer | Choice |
|-------|--------|
| LLM serving (AMD Cloud) | **vLLM** on MI300X (OpenAI-compatible API) |
| LLM serving (local dev) | **llama-server** (llama.cpp, OpenAI-compatible) |
| LLM client | `langchain_openai.ChatOpenAI` — same code for both backends |
| Model (local) | `bartowski/Qwen_Qwen3-14B-GGUF:Q4_K_M` (llama-server) |
| Model (AMD Cloud) | `meta-llama/Llama-3.1-8B-Instruct` or equivalent via vLLM |
| Orchestration | **LangGraph ≥ 1.2** (was 0.2) |
| Observability | **LangSmith** — zero-code tracing via env vars |
| Schemas | Pydantic ≥ 2.6 |
| Sentiment | vaderSentiment (CPU) — upgrade path: RoBERTa on ROCm |
| Embeddings | **sentence-transformers ≥ 5.5** (`BAAI/bge-small-en-v1.5`) — **new** |
| Entity graph | NetworkX ≥ 3.2 |
| Dashboard | Streamlit ≥ 1.35 |

`requirements.txt`:
```
langchain-openai>=1.3.0
langchain-core>=1.4.0
langchain>=1.3.0
langsmith>=0.8.0
langgraph>=1.2.5
langgraph-prebuilt>=1.1.0
pydantic>=2.13
networkx>=3.4
vaderSentiment>=3.3.2
streamlit>=1.58
sentence-transformers>=5.5.0
numpy>=2.0
matplotlib>=3.10
```

## 2.5 AMD Developer Cloud setup

```bash
# 1. ROCm 7.x-enabled torch (must precede sentence-transformers)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.2

# 2. Project dependencies
pip install -r requirements.txt

# 3. vLLM + model
pip install vllm
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000 --dtype float16 &

# 4. App env vars
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
export EMBED_MODEL=BAAI/bge-small-en-v1.5   # optional, this is the default

# 5. LangSmith (token usage + observability for submission report)
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your key>
export LANGCHAIN_PROJECT=brand-risk-hackathon

# 6. Run
rocm-smi                   # capture GPU/VRAM for submission Slide 5
python -m brand_risk.smoke # verify deterministic spine
streamlit run app.py       # launch dashboard
```

**Local dev (llama-server):**
```bash
llama-server -m model.gguf --port 8080
export LLM_BASE_URL=http://localhost:8080/v1
streamlit run app.py
```

## 2.6 Build milestones — all complete ✓

1. ✅ **Spine** — `schemas.py`, `synthetic_data.py`, VADER spike logic, `smoke.py`.
   Both Acme Foods and Dana Rivera spikes fire. Vendor graph and news retrieval print correctly.
2. ✅ **Reasoning** — `llm.py` (LangChain), `embeddings.py` (sentence-transformers),
   `adverse_agent` (disambiguation + risk scoring), `vendor_agent` (graph traversal).
3. ✅ **Loop** — `orchestrator.py` LangGraph 1.x with two conditional edges,
   vendor risk scoping fix, and `synthesise`. Brand + executive scenarios flow
   end-to-end to `ReputationDossier`.
4. ✅ **Show** — `app.py` 3-panel dashboard with live streaming, dossier cards,
   vendor graph visualisation, and JSON download. LangSmith tracing active.

## 2.7 Acceptance criteria

- [x] `smoke.py` deterministically fires both planted spikes (brand + executive).
- [x] Each agent returns a schema-valid object; malformed LLM output self-corrects via retry.
- [x] Disambiguation rejects the irrelevant sports article; keeps on-entity articles.
- [x] Only material findings trigger the vendor screen (gating works).
- [x] A dossier correctly links social spike → adverse finding → impacted vendors (scoping fix applied).
- [ ] `rocm-smi` confirms LLM + embeddings run on AMD GPU (AMD Cloud step).

## 2.8 Success metrics (for submission Slide 6)

| Metric | Target |
|--------|--------|
| Negative-narrative detection latency | < 2 min from injected spike |
| Adverse-media precision (entity-correct) | > 80% |
| Entity disambiguation accuracy | > 85% |
| Vendor-mapping recall | > 75% |
| LLM throughput on MI300X | > 50 tok/s |
| Embedding throughput on MI300X | > 500 sentences/s |

## 2.9 Architecture decisions & rationale

| Decision | Rationale |
|----------|-----------|
| vLLM over Ollama on AMD Cloud | vLLM has first-class ROCm support, higher throughput, and OpenAI-compatible API — no client code changes vs. local dev |
| llama-server for local dev | Lightweight, same OpenAI-compatible API; no Ollama daemon required |
| LangChain `ChatOpenAI` as unified client | Works against both vLLM and llama-server; unlocks LangSmith tracing at zero instrumentation cost |
| LangGraph 1.2 | Backward-compatible StateGraph API; `graph.stream(stream_mode="updates")` enables synchronous progressive UI updates without async/event-loop conflicts in Streamlit |
| Manual `chat_json` retry loop | `with_structured_output` confirmed broken for nested Pydantic with local Llama/Qwen (langchain-ai/langchain#28412) |
| `BAAI/bge-small-en-v1.5` embeddings | 33M params, <200 MB; fast on CPU (fallback) and ROCm GPU; replaces fragile substring matching in `search_news()` |
| `app.py` at project root | Streamlit adds the script's directory to `sys.path`, not the project root; placing the entry point at root makes `from brand_risk import ...` resolve correctly without `PYTHONPATH` hacks |
| `NEG_POST = -0.20` | VADER scores indirect negative language (e.g. "tone-deaf", "dodging") near 0.0; -0.20 threshold is required for the executive scenario to fire |
