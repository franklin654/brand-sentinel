"""Alert rules — per-entity spike thresholds and notification preferences.

Config is persisted to brand_risk/alert_config.json and read at runtime by
social_agent() to override the module-level NEG_RATIO default.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from brand_risk import synthetic_data as data

CONFIG_PATH = Path(__file__).parent.parent / "brand_risk" / "alert_config.json"
NEG_RATIO_DEFAULT = 0.30
CEILING_DEFAULT   = 100


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


st.title("Alert Rules")
st.caption(
    "Customize per-entity spike thresholds. "
    "Changes take effect on the next pipeline run."
)

cfg = _load_config()
new_cfg: dict = {}

for entity in (st.session_state.get("uploaded_watchlist") or data.WATCHLIST):
    eid  = entity.entity_id
    ecfg = cfg.get(eid, {})

    with st.expander(f"{entity.name}  ({entity.kind})", expanded=False):
        col_ratio, col_ceil = st.columns(2)

        neg_ratio = col_ratio.slider(
            "Spike threshold (neg_ratio)",
            min_value=0.05, max_value=0.60, step=0.05,
            value=float(ecfg.get("neg_ratio", NEG_RATIO_DEFAULT)),
            key=f"ratio_{eid}",
            help="Fraction of negative posts that triggers a spike alert.",
        )
        risk_ceil = col_ceil.slider(
            "Risk score ceiling",
            min_value=20, max_value=100, step=5,
            value=int(ecfg.get("risk_score_ceiling", CEILING_DEFAULT)),
            key=f"ceil_{eid}",
            help="If adverse risk_score exceeds this, flag for escalation (Phase 5).",
        )

        notify_opts = ecfg.get("notify", [])
        notify_slack = st.checkbox("Notify via Slack", value="slack" in notify_opts, key=f"slack_{eid}")
        notify_email = st.checkbox("Notify via Email", value="email" in notify_opts, key=f"email_{eid}")

        notify = []
        if notify_slack:
            notify.append("slack")
        if notify_email:
            notify.append("email")

        new_cfg[eid] = {
            "neg_ratio":         neg_ratio,
            "risk_score_ceiling": risk_ceil,
            "notify":            notify,
        }

st.divider()
if st.button("Save alert rules", type="primary"):
    _save_config(new_cfg)
    st.success(f"Alert rules saved to `{CONFIG_PATH.name}`.")

if CONFIG_PATH.exists():
    st.caption(f"Config file: `{CONFIG_PATH}`")
else:
    st.caption("No config saved yet — defaults apply (neg_ratio=0.30).")
