# UI Improvement Plan — 3 Phases: Bug Fixes → Visual Polish → New Features

## Context

All 5 pipeline phases are implemented and working. This plan improves the Streamlit
frontend across three incremental phases: correctness bugs first, then visual polish,
then net-new features. The goal is a demo-ready UI that showcases AMD hardware
performance and pipeline intelligence clearly to a hackathon jury.

---

## Phase 1 — Bug Fixes

### 1a. `_helpers.py` line-count violation (205 lines → under 200)

Extract the three chat-page helpers into a new `pages/_chat_helpers.py`:
- `build_system_prompt(dossiers, entity_filter)` → move to `_chat_helpers.py`
- `format_history_context(entity_id)` → move to `_chat_helpers.py`
- `render_starter_buttons(questions)` → move to `_chat_helpers.py`

Update `pages/05_chat.py` import from `pages._helpers` → `pages._chat_helpers`.
`_helpers.py` drops to ~165 lines. `_chat_helpers.py` is ~45 lines. Both under 200.

### 1b. Entity Detail ignores uploaded watchlist

**File:** `pages/02_entity_detail.py` line 15

```python
# Before
entities = data.WATCHLIST
# After
entities = st.session_state.get("uploaded_watchlist") or data.WATCHLIST
```

### 1c. Alert Rules ignores uploaded watchlist

**File:** `pages/03_alerts.py` line 42

```python
# Before
for entity in data.WATCHLIST:
# After
watchlist = st.session_state.get("uploaded_watchlist") or data.WATCHLIST
for entity in watchlist:
```

### 1d. Dashboard vendor graph visualization ignores uploaded graph

**File:** `pages/01_dashboard.py` line 122

```python
# Before
g_viz = data.vendor_graph()
# After
g_viz = st.session_state.get("uploaded_graph") or data.vendor_graph()
```

### Verification
- Upload a custom watchlist CSV → Entity Detail selector and Alert Rules both show the new entities
- Build a custom vendor graph via Documents → Dashboard graph reflects it
- `wc -l pages/_helpers.py pages/_chat_helpers.py` — both under 200

---

## Phase 2 — Visual Polish

### 2a. Dark theme

**New file:** `.streamlit/config.toml`

```toml
[theme]
primaryColor = "#e74c3c"
backgroundColor = "#0f1117"
secondaryBackgroundColor = "#1a1f2e"
textColor = "#fafafa"
font = "sans serif"
```

No code changes — Streamlit picks this up automatically on restart.

### 2b. KPI strip at dashboard top

**File:** `pages/01_dashboard.py` — insert after `st.caption(...)`, before `ctrl1, ctrl2`:

```python
if result := st.session_state.get("result"):
    k1, k2, k3 = st.columns(3)
    dossiers_kpi = result.get("dossiers", [])
    k1.metric("Entities monitored", len(dossiers_kpi))
    k2.metric("Highest risk score", max((d.adverse.risk_score for d in dossiers_kpi), default=0))
    k3.metric("Alerts fired", len(st.session_state.get("alerts", [])))
```

Shows on re-render after first run; empty state shows nothing (no confusing zeros).

### 2c. Per-agent progress with `st.status()`

**File:** `pages/01_dashboard.py` — replace the `with st.spinner(...)` block:

```python
with st.status("Agents running…", expanded=True) as status:
    result = _stream_run(st.session_state.graph,
                         {"watchlist": watchlist, "graph": g, "posts": posts},
                         status)
    status.update(label="Pipeline complete", state="complete", expanded=False)
```

Update `_stream_run()` to accept `status` and call `status.write(f"✓ {node_name}")` after each node completes. `st.status()` is available in Streamlit ≥ 1.28; requirements pin `>=1.58` so this is safe.

### Verification
- Restart Streamlit → dark theme loads
- Run pipeline → KPI strip shows after run completes
- During run → st.status expander shows "✓ social", "✓ adverse", "✓ vendor" as each node finishes

---

## Phase 3 — New Features

### 3a. Interactive vendor graph (pyvis)

**New dep:** `pyvis>=0.3.2` added to `requirements.txt`

**File:** `pages/01_dashboard.py` — replace the matplotlib vendor graph block (~20 lines) with:

```python
from pyvis.network import Network
net = Network(height="400px", width="100%", bgcolor="#1a1f2e", font_color="white")
for node in g_viz.nodes():
    color = "#e74c3c" if node in flagged else "#95a5a6"
    net.add_node(node, label=g_viz.nodes[node]["name"], color=color)
for src, tgt, data in g_viz.edges(data=True):
    net.add_edge(src, tgt, title=data.get("relation", ""))
net.set_options('{"physics": {"stabilization": {"iterations": 50}}}')
html = net.generate_html()
st.components.v1.html(html, height=420)
```

Draggable, zoomable, hover shows relation labels. Replaces static matplotlib graph.
Dashboard stays under 200 lines because we remove ~20 matplotlib lines and add ~10 pyvis lines.

### 3b. Risk heatmap

**File:** `pages/02_entity_detail.py` — add after the existing history chart section:

```python
# Risk heatmap — entity × run
all_entities = risk_store.get_all_entities()
if len(all_entities) > 1:
    st.subheader("Risk score heatmap")
    import numpy as np
    matrix, row_labels, col_labels = _build_heatmap_data(all_entities)
    fig, ax = plt.subplots(figsize=(max(6, len(col_labels)), len(row_labels) * 0.8 + 1))
    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=8)
    plt.colorbar(im, ax=ax, label="Risk score")
    st.pyplot(fig); plt.close(fig)
```

`_build_heatmap_data()` is a small private helper (~15 lines) using `store.get_history()` per entity; aligns runs by timestamp. Add to `02_entity_detail.py` (currently 118 lines; stays under 200).

### 3c. Executive summary button

**File:** `pages/01_dashboard.py` — add a third download column next to JSON + PDF:

```python
dl1, dl2, dl3 = st.columns(3)
# ... existing JSON and PDF buttons in dl1/dl2 ...
if dl3.button("Generate executive summary"):
    with st.spinner("Drafting summary…"):
        from brand_risk.llm import chat
        payload = "\n\n".join(d.model_dump_json(indent=2) for d in dossiers)
        summary = chat(
            "You are a risk analyst. Write a concise 3-paragraph executive summary "
            "of these brand risk dossiers for a board audience. Plain text only.",
            payload, temperature=0.3,
        )
    st.session_state.exec_summary = summary

if exec_sum := st.session_state.get("exec_summary"):
    with st.expander("Executive summary", expanded=True):
        st.write(exec_sum)
```

Reuses existing `llm.chat()` — no new modules.

### 3d. Run history browser (new page)

**New file:** `pages/06_history.py` (~80 lines)

Shows a table of all pipeline runs from SQLite with columns: timestamp, entity, score,
risk level. Clicking a row expands the full dossier JSON in an expander.

```python
from brand_risk import store as risk_store
entities = risk_store.get_all_entities()
rows = []
for eid, ename in entities:
    for h in risk_store.get_history(eid):
        rows.append({"Entity": ename, "Run": h["run_ts"][:16], "Score": h["risk_score"], "Level": h["risk_category"]})
df = pd.DataFrame(rows).sort_values("Run", ascending=False)
st.dataframe(df, use_container_width=True)
```

**New deps:**
- `pyvis>=0.3.2`
- `pandas>=3.0.3`

**File:** `app.py` — add the new page to `st.navigation()`:

```python
st.Page("pages/06_history.py", title="Run History", icon="📈"),
```

### Verification
- `conda run -n cuda_env pip install pyvis pandas` → import check
- Dashboard graph is draggable and shows edge labels on hover
- Entity Detail shows heatmap when ≥ 2 entities have ≥ 2 runs in SQLite
- "Generate executive summary" button produces a 3-paragraph text block
- Run History page shows a sortable table of all past runs

---

## Files modified per phase

| Phase | Files touched |
|---|---|
| 1 | `pages/_helpers.py`, `pages/_chat_helpers.py` (new), `pages/05_chat.py`, `pages/02_entity_detail.py`, `pages/03_alerts.py`, `pages/01_dashboard.py` |
| 2 | `.streamlit/config.toml` (new), `pages/01_dashboard.py` |
| 3 | `pages/01_dashboard.py`, `pages/02_entity_detail.py`, `pages/06_history.py` (new), `app.py`, `requirements.txt` |

## Line count constraints (CLAUDE.md: 200-line max)

| File | Current | After plan |
|---|---|---|
| `pages/_helpers.py` | 205 | ~165 (after chat helpers extracted) |
| `pages/_chat_helpers.py` | — | ~45 (new) |
| `pages/01_dashboard.py` | 191 | ~195 (KPI + status + exec summary + pyvis) |
| `pages/02_entity_detail.py` | 118 | ~175 (heatmap + _build_heatmap_data) |
| `pages/06_history.py` | — | ~80 (new) |
