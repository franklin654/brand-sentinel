# Brand & Reputational Risk Intelligence — Implementation Scaffold

**TCS-AMD AI Hackathon · Track 1 (Agents) · Flow: AGENTS_039 → AGENTS_001 → AGENTS_016**

A runnable three-agent pipeline that detects a negative social signal, confirms
it against news with explainable risk scoring, and traces the exposure to the
vendors and partners it touches.

```
watchlist ─▶ AGENTS_039 ─(spike?)─▶ AGENTS_001 ─(material?)─▶ AGENTS_016 ─▶ dossier
            social signal          adverse media            vendor risk
```

---

## Why it is built this way

| Decision | Reason |
|----------|--------|
| **VADER for per-post sentiment, LLM only for reasoning** | Scoring hundreds of posts with an LLM is slow and wasteful. VADER detects the *spike* instantly on CPU; the LLM is spent only on clustering, disambiguation, risk explanation, and vendor reasoning — where it earns its latency. |
| **Two conditional edges in LangGraph** | An entity only triggers an expensive adverse-media screen if a spike was detected, and a vendor screen only runs if the adverse finding was material. This gating is what makes the pipeline cheap enough to run continuously. |
| **Pydantic-validated JSON between agents** | Every LLM call returns a validated object, so the dashboard renders off known fields and a malformed reply self-corrects instead of crashing the demo. |
| **Planted synthetic scenarios + local news corpus** | A reproducible demo cannot fail because a live web search returned nothing. The *data* is planted; the *reasoning* over it is real. Swap `search_news()` for a live API by keeping its signature. |

---

## Setup on AMD Developer Cloud

```bash
# 1. Serve an LLM on the AMD GPU via ROCm
curl -fsSL https://ollama.com/install.sh | sh
OLLAMA_ROCM=1 ollama serve &                 # uses AMD Instinct via ROCm
ollama pull llama3.1:8b-instruct-q4_K_M      # ~5 GB, fits one MI300X easily

# 2. Python deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Confirm the GPU is in use
rocm-smi                                      # watch utilisation during a run
```

---

## Run

```bash
# Sanity-check the deterministic spine (no GPU needed)
python -m brand_risk.smoke

# Full pipeline + dashboard
streamlit run brand_risk/app.py
```

Click **Run monitoring cycle**. You should see Acme Foods light up as a spike,
an adverse-media dossier with a risk score and cited sources, and Nimbus +
Orchard inherit vendor exposure.

---

## Layout

```
brand_risk/
  schemas.py          Pydantic contracts between agents
  llm.py              Ollama/ROCm client + JSON-validated chat
  synthetic_data.py   watchlist, planted social stream, news corpus, vendor graph
  agents.py           AGENTS_039 / 001 / 016 — the three agents
  orchestrator.py     LangGraph state machine with conditional edges
  app.py              Streamlit control tower
  smoke.py            deterministic spine test (no LLM)
```

---

## 3-day build order (mapped to this scaffold)

**Day 1 — spine.** Stand up Ollama on ROCm. Get `synthetic_data.py` and the
VADER spike detection in `social_agent` working; verify with `smoke.py`. This is
the deterministic backbone — once it fires reliably, nothing downstream is blind.

**Day 2 — reasoning.** Implement `adverse_agent` (disambiguation + risk) and
`vendor_agent` (graph traversal + action). Wire them into `orchestrator.py` with
the two conditional edges. Unit-test each agent on one planted scenario.

**Day 3 — show.** Build out `app.py` (three panels mirroring the three agents),
add a second planted scenario (executive scandal), tune thresholds, capture the
`rocm-smi` numbers for the submission's Slide 5, and rehearse the demo.

---

## Known tuning knobs (the threshold work the smoke test surfaces)

- `NEG_POST` (agents.py): per-post negativity bar. Lower to −0.20 to make the
  milder executive scenario fire; keep at −0.30 for clear product-recall spikes.
- `NEG_RATIO` / `MIN_VOLUME`: how much of the chatter must be negative before a
  spike is called. Raise to cut false positives on noisy entities.
- `MODEL` (llm.py): swap to `mistral:7b-instruct` if you want lower latency at a
  small reasoning-quality cost.

---

## Upgrade paths (if you have time)

- **GPU sentiment** — replace VADER with `cardiffnlp/twitter-roberta-base-sentiment`
  on ROCm for real throughput on a high-volume stream (same call site in `social_agent`).
- **Live news** — point `search_news()` at `ddgs` or SerpAPI; the return shape is
  already the contract the adverse agent expects.
- **Persisted watchlist + history** — back the entity graph with SQLite so trend
  signals accumulate over time rather than per-cycle.
