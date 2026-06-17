"""Page 5 — Analyst Chat.

Interactive multi-turn conversation grounded in the latest pipeline run.
The system prompt is built once from session_state dossiers and stored in
st.session_state.chat_system; it is rebuilt only when the entity scope changes
or the user resets the conversation. Prior turns are concatenated into each
user message so the LLM maintains context across turns.
"""
from __future__ import annotations

import streamlit as st

from brand_risk import store as risk_store
from brand_risk.llm import chat_stream
from pages._chat_helpers import build_system_prompt, format_history_context, render_starter_buttons

_HISTORY_KEYWORDS = ("before", "history", "trend", "last time", "previous", "past", "ever been")

_STARTER_QUESTIONS = [
    "What is the highest-risk entity this run and why?",
    "Should any of these risks be escalated to the board?",
    "Draft a press response for {top_entity}'s situation.",
    "Give me a step-by-step vendor action plan for {vendor_name}.",
]

st.title("Analyst Chat")
st.caption("Ask questions about the latest pipeline findings. Responses are grounded in the run data.")

# ── Collect dossiers: live run → focused scans → SQLite fallback ──────────────
result = st.session_state.get("result")
dossiers = list(result.get("dossiers", [])) if result else []

# Merge focused entity scans that are not already in the full run (Phase 1d)
for _fres in (st.session_state.get("entity_focus_results") or {}).values():
    for _d in _fres.get("dossiers", []):
        if not any(x.entity_id == _d.entity_id for x in dossiers):
            dossiers.append(_d)

if not dossiers:
    _db_entities = risk_store.get_all_entities()
    if not _db_entities:
        st.warning(
            "No pipeline results found — run the monitoring cycle on the **Dashboard** first, "
            "then return here."
        )
        st.stop()

    # Historical mode: answer questions from last-known DB scores only
    st.info(
        "Historical mode — showing last known scores. "
        "Run a pipeline cycle on the Dashboard for full dossier detail."
    )
    if "chat_system" not in st.session_state or st.session_state.get("_chat_mode") != "history":
        _lines = [
            "You are a brand and reputational risk analyst. The following entities have "
            "historical run data from the database. Full narrative and media details are "
            "not available — answer based on these score summaries.\n"
        ]
        for _eid, _ename in _db_entities:
            _latest = risk_store.get_latest(_eid)
            if _latest:
                _hist = risk_store.get_history(_eid)
                _lines.append(f"--- ENTITY: {_ename} ---")
                _lines.append(
                    f"Latest score: {_latest['risk_score']}/100 "
                    f"({_latest['overall_risk'].upper()}) at {_latest['run_ts'][:16]} UTC"
                )
                if len(_hist) >= 2:
                    _trend = "improving" if _hist[-1]["risk_score"] < _hist[-2]["risk_score"] else "worsening"
                    _lines.append(f"Trend: {_trend}")
                _lines.append("")
        _lines.append(
            "Answer analyst questions based on these scores. "
            "Recommend running a fresh pipeline cycle for full detail."
        )
        st.session_state.chat_system = "\n".join(_lines)
        st.session_state.chat_messages = []
        st.session_state._chat_mode = "history"

    for msg in st.session_state.get("chat_messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if _user_input := st.chat_input("Ask about historical risk data…"):
        st.session_state.chat_messages.append({"role": "user", "content": _user_input})
        with st.chat_message("user"):
            st.markdown(_user_input)
        with st.chat_message("assistant"):
            _response = st.write_stream(chat_stream(st.session_state.chat_system, _user_input))
        st.session_state.chat_messages.append({"role": "assistant", "content": _response})
    st.stop()

# ── Entity scope selector ─────────────────────────────────────────────────────
scope_col, reset_col = st.columns([4, 1])
entity_options = ["All entities"] + [d.entity_name for d in dossiers]
selected_scope = scope_col.selectbox(
    "Context scope",
    options=entity_options,
    key="chat_scope",
    help="Narrow to one entity for focused questions; 'All entities' enables comparison.",
)
entity_filter = None if selected_scope == "All entities" else next(
    (d.entity_id for d in dossiers if d.entity_name == selected_scope), None
)

# Rebuild system prompt and clear history on scope change or first load
if st.session_state.get("_chat_scope_key") != selected_scope or "chat_system" not in st.session_state:
    st.session_state.chat_system = build_system_prompt(dossiers, entity_filter)
    st.session_state.chat_messages = []
    st.session_state._chat_scope_key = selected_scope

if reset_col.button("Reset", help="Clear conversation history"):
    st.session_state.chat_messages = []
    st.session_state.chat_system = build_system_prompt(dossiers, entity_filter)
    st.rerun()

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# ── Render existing message history ───────────────────────────────────────────
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Starter question buttons (shown only when history is empty) ───────────────
if not st.session_state.chat_messages:
    top = max(dossiers, key=lambda d: d.adverse.risk_score) if dossiers else None
    vendor = next((v.vendor_name for d in dossiers for v in d.vendor_impacts), None)
    questions = [
        q.replace("{top_entity}", top.entity_name if top else "this entity")
         .replace("{vendor_name}", vendor or "the primary vendor")
        for q in _STARTER_QUESTIONS
    ]
    st.markdown("**Suggested questions:**")
    clicked = render_starter_buttons(questions)
    if clicked:
        st.session_state._pending_question = clicked
        st.rerun()

# ── Chat input + streaming response ──────────────────────────────────────────
pending = st.session_state.pop("_pending_question", None)
user_input = st.chat_input("Ask about the pipeline findings…") or pending

if user_input:
    # Inject historical context when the question seems trend-related
    enriched = user_input
    if any(kw in user_input.lower() for kw in _HISTORY_KEYWORDS):
        for d in dossiers:
            if entity_filter == d.entity_id or d.entity_name.lower() in user_input.lower():
                enriched = f"{user_input}\n\n[HISTORY CONTEXT]\n{format_history_context(d.entity_id)}"
                break

    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Concatenate prior turns into the user message for multi-turn context
    prior = "\n\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in st.session_state.chat_messages[:-1]
    )
    full_user = f"{prior}\n\nUSER: {enriched}" if prior else enriched

    with st.chat_message("assistant"):
        response = st.write_stream(chat_stream(st.session_state.chat_system, full_user))

    st.session_state.chat_messages.append({"role": "assistant", "content": response})
