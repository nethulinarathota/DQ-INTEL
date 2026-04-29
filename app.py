"""
DQ·INTEL — Data Quality Intelligence
Streamlit app. Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
import os
from pathlib import Path
from analysis import analyze_dataset, generate_recommendations, compare_schemas
from ml_advanced import run_advanced_ml
from agent import get_response, build_system_prompt

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DQ·INTEL",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── INJECT CSS ───────────────────────────────────────────────────────────────

css_path = Path(__file__).parent / "styles.css"
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── HELPERS ──────────────────────────────────────────────────────────────────

def score_color(s):
    if s >= 80: return "#16a34a"
    if s >= 60: return "#b45309"
    return "#991b1b"

def score_label(s):
    if s >= 80: return "Good"
    if s >= 60: return "Moderate"
    return "Poor"

def score_banner_color(s):
    if s >= 80: return "#86efac"
    if s >= 60: return "#fcd34d"
    return "#fca5a5"

def miss_color(pct):
    if pct > 30: return "#dc2626"
    if pct > 10: return "#d97706"
    return "#16a34a"

def impact_order(r):
    return {"High": 3, "Medium": 2, "Low": 1}[r["impact"]]

def read_css():
    with open(css_path) as f:
        return f.read()

# ── SESSION STATE ────────────────────────────────────────────────────────────

if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "df" not in st.session_state:
    st.session_state.df = None
if "recs" not in st.session_state:
    st.session_state.recs = []
if "ml" not in st.session_state:
    st.session_state.ml = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_key" not in st.session_state:
    st.session_state.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
if "rule_history" not in st.session_state:
    st.session_state.rule_history = []
if "drift_analysis" not in st.session_state:
    st.session_state.drift_analysis = None

# ── UPLOAD SCREEN ────────────────────────────────────────────────────────────

if st.session_state.analysis is None:
    st.markdown("""
    <div style="min-height:100vh;display:flex;align-items:center;justify-content:center;flex-direction:column;padding:2rem">
      <div style="text-align:center;max-width:560px;width:100%;padding-top:3rem">
        <div style="font-family:'Geist Mono',monospace;font-size:11px;letter-spacing:.2em;color:#9a9590;text-transform:uppercase;margin-bottom:3rem;display:flex;align-items:center;gap:.6rem;justify-content:center">
          <span style="width:5px;height:5px;background:#1a1612;border-radius:50%;display:inline-block"></span>
          DQ·INTEL
        </div>
        <div class="upload-headline">Understand your<br>data <em>quality</em></div>
        <div style="font-family:'Geist Mono',monospace;font-size:12px;color:#9a9590;margin:1rem 0 2rem;letter-spacing:.04em">Drop a CSV. Get a full quality audit in seconds.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # API key input
        with st.expander("⚙ API Key (required for AI agent)"):
            api_input = st.text_input(
                "Anthropic API Key",
                value=st.session_state.api_key,
                type="password",
                placeholder="sk-ant-...",
                help="Your key stays local — never sent anywhere except api.anthropic.com",
            )
            if api_input:
                st.session_state.api_key = api_input

        uploaded = st.file_uploader(
            "Drop your CSV here",
            type=["csv"],
            label_visibility="collapsed",
        )

        if uploaded:
            with st.spinner("Analysing dataset…"):
                try:
                    # Try common encodings
                    for enc in ("utf-8", "latin-1", "cp1252"):
                        try:
                            df = pd.read_csv(uploaded, encoding=enc, low_memory=False)
                            break
                        except UnicodeDecodeError:
                            uploaded.seek(0)
                            continue

                    # Auto-detect delimiter if needed
                    if len(df.columns) <= 1:
                        uploaded.seek(0)
                        df = pd.read_csv(uploaded, sep=None, engine="python", low_memory=False)

                    analysis = analyze_dataset(df, uploaded.name)
                    recs = generate_recommendations(analysis)
                    # ── ADVANCED ML (replaces old ml_readiness) ──────────────
                    ml = run_advanced_ml(df, analysis)

                    st.session_state.df = df
                    st.session_state.analysis = analysis
                    st.session_state.recs = recs
                    st.session_state.ml = ml
                    st.session_state.chat_history = []
                    st.session_state.rule_history = []
                    st.session_state.drift_analysis = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not parse CSV: {e}")

        st.markdown("""
        <div style="display:flex;gap:.5rem;justify-content:center;flex-wrap:wrap;margin-top:1.5rem">
          <span class="sec-lbl" style="margin:0">Supports</span>
          <span style="font-family:'Geist Mono',monospace;font-size:10.5px;padding:.3rem .8rem;border-radius:20px;border:1px solid #dbd6cf;color:#4a4540;background:#fff">CSV</span>
          <span style="font-family:'Geist Mono',monospace;font-size:10.5px;padding:.3rem .8rem;border-radius:20px;border:1px solid #dbd6cf;color:#4a4540;background:#fff">Up to 1 GB</span>
          <span style="font-family:'Geist Mono',monospace;font-size:10.5px;padding:.3rem .8rem;border-radius:20px;border:1px solid #dbd6cf;color:#4a4540;background:#fff">Auto-encoding detection</span>
          <span style="font-family:'Geist Mono',monospace;font-size:10.5px;padding:.3rem .8rem;border-radius:20px;border:1px solid #dbd6cf;color:#4a4540;background:#fff">Any delimiter</span>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ── DASHBOARD ────────────────────────────────────────────────────────────────

A   = st.session_state.analysis
df  = st.session_state.df
recs = st.session_state.recs
ml  = st.session_state.ml
# Shorthand for the readiness sub-dict used throughout Tab 5
mlr = ml["readiness"] if ml else {}

# Header
badge_color = score_banner_color(A["DQS"])
if A["DQS"] >= 80:
    badge_bg = "rgba(134,239,172,.15)"; badge_border = "rgba(134,239,172,.3)"
elif A["DQS"] >= 60:
    badge_bg = "rgba(253,211,77,.15)"; badge_border = "rgba(253,211,77,.3)"
else:
    badge_bg = "rgba(252,165,165,.15)"; badge_border = "rgba(252,165,165,.3)"

dims = [
    ("Completeness", "30%", A["C"], "Missing values"),
    ("Consistency",  "25%", A["Co"], "Format issues"),
    ("Validity",     "25%", A["V"], "Outliers / rules"),
    ("Uniqueness",   "20%", A["U"], "Duplicates"),
]
dim_cards_html = "".join(f"""
<div class="dim-card">
  <div class="dc-lbl">{l} <span style="opacity:.5">{w}</span></div>
  <div class="dc-score" style="color:{score_banner_color(s)}">{s:.0f}</div>
  <div class="dc-bar"><div class="dc-fill" style="width:{s}%;background:{score_banner_color(s)}"></div></div>
  <div class="dc-desc">{d}</div>
</div>""" for l, w, s, d in dims)

st.markdown(f"""
<div class="score-banner">
  <div class="inner">
    <div>
      <div class="score-big">{A['DQS']:.0f}</div>
      <div class="score-lbl">Data Quality Score</div>
      <div class="score-badge" style="background:{badge_bg};color:{badge_color};border:1px solid {badge_border}">{score_label(A['DQS'])}</div>
    </div>
    <div class="formula-box">
      <div style="opacity:.5;margin-bottom:2px">DQS = 0.30·C + 0.25·Co + 0.25·V + 0.20·U</div>
      <div style="color:rgba(245,243,239,.7)">= 0.30·{A['C']:.0f} + 0.25·{A['Co']:.0f} + 0.25·{A['V']:.0f} + 0.20·{A['U']:.0f}</div>
      <div style="color:rgba(245,243,239,.9)">= {A['DQS']:.2f}</div>
    </div>
    <div class="dim-cards">{dim_cards_html}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Nav bar
nav_col1, nav_col2 = st.columns([6, 1])
with nav_col1:
    st.markdown(f"""
    <div style="padding:.6rem 1.5rem;border-bottom:1px solid var(--rule);background:rgba(245,243,239,.92);backdrop-filter:blur(12px);display:flex;align-items:center;gap:1.25rem">
      <span style="font-family:'Geist Mono',monospace;font-size:12px;font-weight:600;letter-spacing:.15em">DQ·INTEL</span>
      <span style="width:1px;height:18px;background:var(--rule);display:inline-block"></span>
      <span style="font-family:'Geist Mono',monospace;font-size:11px;color:#9a9590"><strong style="color:#4a4540">{A['file_name']}</strong></span>
      <span style="font-family:'Geist Mono',monospace;font-size:11px;color:#9a9590">{A['total_rows']:,} rows · {A['num_cols']} cols</span>
    </div>
    """, unsafe_allow_html=True)
with nav_col2:
    if st.button("↩ New file", type="secondary"):
        for key in ["analysis", "df", "recs", "ml", "chat_history", "rule_history", "drift_analysis"]:
            st.session_state[key] = None if key not in ("recs", "chat_history", "rule_history") else []
        st.rerun()

st.markdown("<div style='padding:0 1.5rem'>", unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────────────────────────────

tab_labels = [
    "Overview",
    "Issues",
    "Root Cause",
    "Correlation",
    "ML Readiness",
    f"Recommendations ({len(recs)})",
    "Custom Rules",
    "Schema Drift",
    "AI Agent",
]
tabs = st.tabs(tab_labels)

# ════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

    if A["total_rows"] > 50_000:
        st.markdown(f"""<div class="note-box mb15"><strong>ⓘ Large dataset —</strong> Analysis performed on a representative sample of {A['sample_size']:,} rows. Scores and estimates reflect full dataset patterns.</div>""", unsafe_allow_html=True)

    miss_cols = sum(1 for c in A["columns"] if c["missing_pct"] > 0)
    outlier_cols = sum(1 for c in A["columns"] if c["outliers"] > 0)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    metrics = [
        (c1, "▦", f"{A['total_rows']:,}", "", "Total Rows"),
        (c2, "▤", str(A["num_cols"]), "", "Columns"),
        (c3, "◻", f"{A['tot_missing']:,}", f"{A['tot_missing']/max(A['total_cells'],1)*100:.1f}% of sample", "Missing Cells"),
        (c4, "⧉", f"~{A['dupes_estimated']:,}", f"{A['dupes_estimated']/max(A['total_rows'],1)*100:.2f}% est.", "Duplicate Rows"),
        (c5, "◈", str(outlier_cols), "", "Cols w/ Outliers"),
        (c6, "◻", str(miss_cols), "", "Cols w/ Missing"),
    ]
    for col, icon, val, sub, label in metrics:
        with col:
            st.markdown(f"""
            <div class="stat-chip">
              <div class="si">{icon}</div>
              <div class="sv">{val}</div>
              {'<div class="ss">'+sub+'</div>' if sub else ''}
              <div class="sl">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

    # Schema table
    st.markdown('<div class="dq-card"><div class="dq-card-hdr"><span class="dq-card-title">Schema & Column Summary</span></div><div class="dq-card-body">', unsafe_allow_html=True)

    rows_html = ""
    for c in A["columns"]:
        pct = c["missing_pct"]
        bar_w = min(pct, 100)
        bar_color = miss_color(pct)
        miss_label = f"{pct:.1f}%"

        health = "◉ Good" if pct < 5 and c["outliers"] == 0 else ("◎ Watch" if pct < 20 else "○ Poor")
        health_color = "#16a34a" if "Good" in health else ("#b45309" if "Watch" in health else "#991b1b")

        top_val = ""
        if c["type"] in ("categorical", "boolean") and c["stats"].get("top_values"):
            top_val = list(c["stats"]["top_values"].keys())[0][:20]
        elif c["type"] == "numeric" and c["stats"].get("mean") is not None:
            top_val = f"μ={c['stats']['mean']:.2f}"

        rows_html += f"""
        <tr>
          <td style="font-weight:500;color:var(--ink)">{c['col']}</td>
          <td><span class="type-badge {c['type']}">{c['type']}</span></td>
          <td>
            <div style="display:flex;align-items:center;gap:.5rem">
              <div style="width:64px;height:4px;background:var(--bg-sunken);border-radius:2px;overflow:hidden">
                <div style="width:{bar_w}%;height:100%;background:{bar_color};border-radius:2px"></div>
              </div>
              <span style="font-family:'Geist Mono',monospace;font-size:11px;color:{bar_color}">{miss_label}</span>
            </div>
          </td>
          <td style="font-family:'Geist Mono',monospace;font-size:11px;color:{'#d97706' if c['outliers'] > 0 else 'var(--ink-3)'}">{c['outliers'] or '—'}</td>
          <td style="font-family:'Geist Mono',monospace;font-size:11px;color:var(--ink-3)">{top_val or '—'}</td>
          <td style="font-size:11px;color:{health_color}">{health}</td>
        </tr>"""

    st.markdown(f"""
    <table class="dtable">
      <thead><tr>
        <th>Column</th><th>Type</th><th>Missing</th><th>Outliers</th><th>Sample</th><th>Health</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>""", unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

    left, right = st.columns([1, 1])
    with left:
        # Missing value bars
        st.markdown('<div class="sec-lbl">Missing Values by Column</div>', unsafe_allow_html=True)
        miss_cols_data = sorted(
            [c for c in A["columns"] if c["missing_pct"] > 0],
            key=lambda c: c["missing_pct"], reverse=True
        )
        if miss_cols_data:
            for c in miss_cols_data[:12]:
                color = miss_color(c["missing_pct"])
                st.markdown(f"""
                <div class="miss-row">
                  <span class="miss-name">{c['col']}</span>
                  <div class="miss-bar"><div class="miss-fill" style="width:{min(c['missing_pct'],100)}%;background:{color}"></div></div>
                  <span class="miss-pct" style="color:{color}">{c['missing_pct']:.1f}%</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-family:\'Geist Mono\',monospace;font-size:12px;color:#16a34a;padding:.5rem 0">✓ No missing values</div>', unsafe_allow_html=True)

    with right:
        # Outliers
        st.markdown('<div class="sec-lbl">Outliers by Column</div>', unsafe_allow_html=True)
        outlier_cols_data = sorted(
            [c for c in A["columns"] if c["outliers"] > 0 and c["type"] == "numeric"],
            key=lambda c: c["outliers"], reverse=True
        )
        if outlier_cols_data:
            for c in outlier_cols_data[:10]:
                pct = c["outliers"] / max(A["sample_size"], 1) * 100
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;padding:.6rem .5rem;border-radius:6px;margin-bottom:2px">
                  <div>
                    <div style="font-size:12px;font-weight:500;color:var(--ink)">{c['col']}</div>
                    <div style="font-family:'Geist Mono',monospace;font-size:10px;color:var(--ink-3)">
                      fence: [{c['stats'].get('lower_fence', 0):.2f}, {c['stats'].get('upper_fence', 0):.2f}]
                    </div>
                  </div>
                  <div style="text-align:right">
                    <div style="font-family:'Instrument Serif',serif;font-size:1.6rem;font-weight:400;line-height:1;color:#b45309">{c['outliers']}</div>
                    <div style="font-family:'Geist Mono',monospace;font-size:10px;color:var(--ink-3)">{pct:.1f}% of rows</div>
                  </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-family:\'Geist Mono\',monospace;font-size:12px;color:#16a34a;padding:.5rem 0">✓ No outliers detected</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # Consistency issues
        st.markdown('<div class="sec-lbl">Consistency Issues</div>', unsafe_allow_html=True)
        incon_cols = [c for c in A["columns"] if c["inconsistent"] > 0]
        if incon_cols:
            for c in incon_cols[:10]:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;padding:.5rem 0;border-bottom:1px solid #f0ece6;font-size:12px">
                  <span style="font-weight:500;color:var(--ink)">{c['col']}</span>
                  <span style="font-family:'Geist Mono',monospace;font-size:11px;color:#b45309">{c['inconsistent']} — {c.get('inconsistent_desc','')}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-family:\'Geist Mono\',monospace;font-size:12px;color:#16a34a;padding:.5rem 0">✓ No consistency issues detected</div>', unsafe_allow_html=True)

    # Column detail
    st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sec-lbl">Column Deep Dive</div>', unsafe_allow_html=True)
    col_names = [c["col"] for c in A["columns"]]
    selected_col = st.selectbox("Select column", col_names, label_visibility="collapsed")

    col_data = next(c for c in A["columns"] if c["col"] == selected_col)
    s = col_data["stats"]

    detail_left, detail_right = st.columns([1, 1])
    with detail_left:
        st.markdown(f"""
        <div class="dq-card">
          <div class="dq-card-hdr">
            <span class="dq-card-title">{selected_col}</span>
            <span class="type-badge {col_data['type']}">{col_data['type']}</span>
          </div>
          <div class="dq-card-body">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:.5rem;margin-bottom:1rem">
              <div class="mstat" style="background:var(--bg-sunken);border-radius:7px;padding:.6rem .75rem">
                <div style="font-family:'Geist Mono',monospace;font-size:9.5px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.08em">Missing</div>
                <div style="font-size:13px;font-weight:500;color:var(--ink);margin-top:2px">{col_data['missing_pct']:.1f}%</div>
              </div>
              <div class="mstat" style="background:var(--bg-sunken);border-radius:7px;padding:.6rem .75rem">
                <div style="font-family:'Geist Mono',monospace;font-size:9.5px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.08em">Non-null</div>
                <div style="font-size:13px;font-weight:500;color:var(--ink);margin-top:2px">{col_data['non_missing']:,}</div>
              </div>
              <div class="mstat" style="background:var(--bg-sunken);border-radius:7px;padding:.6rem .75rem">
                <div style="font-family:'Geist Mono',monospace;font-size:9.5px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.08em">Outliers</div>
                <div style="font-size:13px;font-weight:500;color:var(--ink);margin-top:2px">{col_data['outliers']}</div>
              </div>
              <div class="mstat" style="background:var(--bg-sunken);border-radius:7px;padding:.6rem .75rem">
                <div style="font-family:'Geist Mono',monospace;font-size:9.5px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.08em">Inconsistent</div>
                <div style="font-size:13px;font-weight:500;color:var(--ink);margin-top:2px">{col_data['inconsistent']}</div>
              </div>
            </div>
        """, unsafe_allow_html=True)

        if col_data["type"] == "numeric" and s:
            stats_rows = [
                ("Mean", f"{s.get('mean',0):.4f}"),
                ("Median", f"{s.get('median',0):.4f}"),
                ("Std Dev", f"{s.get('std',0):.4f}"),
                ("Min", f"{s.get('min',0):.4f}"),
                ("Max", f"{s.get('max',0):.4f}"),
                ("Q1", f"{s.get('q1',0):.4f}"),
                ("Q3", f"{s.get('q3',0):.4f}"),
                ("Skewness", f"{s.get('skewness',0):.3f}"),
                ("Kurtosis", f"{s.get('kurtosis',0):.3f}"),
            ]
            rows_html = "".join(f"""
            <div style="display:flex;justify-content:space-between;padding:.35rem 0;border-bottom:1px solid #f0ece6;font-size:12px">
              <span style="font-family:'Geist Mono',monospace;color:var(--ink-3)">{k}</span>
              <span style="color:var(--ink-2);font-weight:500">{v}</span>
            </div>""" for k, v in stats_rows)
            st.markdown(rows_html, unsafe_allow_html=True)

        elif col_data["type"] in ("categorical", "boolean") and s.get("top_values"):
            st.markdown('<div style="font-family:\'Geist Mono\',monospace;font-size:9.5px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.5rem">Top Values</div>', unsafe_allow_html=True)
            tv = s["top_values"]
            total_tv = sum(tv.values())
            for val, count in list(tv.items())[:8]:
                pct = count / max(total_tv, 1) * 100
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:5px">
                  <span style="font-family:'Geist Mono',monospace;font-size:11px;color:var(--ink-2);width:100px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{str(val)[:20]}</span>
                  <div style="flex:1;height:6px;background:var(--bg-sunken);border-radius:2px;overflow:hidden">
                    <div style="width:{pct}%;height:100%;background:var(--ink);border-radius:2px;opacity:.25"></div>
                  </div>
                  <span style="font-family:'Geist Mono',monospace;font-size:10px;color:var(--ink-3);width:36px;text-align:right">{count}</span>
                </div>""", unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

    with detail_right:
        if col_data["type"] == "numeric" and s.get("histogram"):
            hist = s["histogram"]
            fig = go.Figure(go.Bar(
                x=[(hist["edges"][i] + hist["edges"][i+1]) / 2 for i in range(len(hist["counts"]))],
                y=hist["counts"],
                width=[(hist["edges"][i+1] - hist["edges"][i]) * 0.9 for i in range(len(hist["counts"]))],
                marker_color="#1a1612",
                marker_opacity=0.7,
            ))
            fig.update_layout(
                height=260, margin=dict(l=0, r=0, t=20, b=0),
                paper_bgcolor="white", plot_bgcolor="white",
                xaxis=dict(showgrid=False, color="#9a9590", tickfont=dict(family="Geist Mono", size=10)),
                yaxis=dict(showgrid=True, gridcolor="#f0ece6", color="#9a9590", tickfont=dict(family="Geist Mono", size=10)),
                title=dict(text="Distribution", font=dict(family="Geist Mono", size=11, color="#9a9590")),
            )
            if s.get("lower_fence") is not None:
                fig.add_vline(x=s["lower_fence"], line_dash="dash", line_color="#991b1b", line_width=1, opacity=0.5)
                fig.add_vline(x=s["upper_fence"], line_dash="dash", line_color="#991b1b", line_width=1, opacity=0.5)
            st.plotly_chart(fig, use_container_width=True)
        elif col_data["type"] in ("categorical", "boolean") and s.get("top_values"):
            tv = s["top_values"]
            labels = [str(k)[:15] for k in list(tv.keys())[:8]]
            values = list(tv.values())[:8]
            fig = go.Figure(go.Bar(
                x=labels, y=values,
                marker_color="#1a1612", marker_opacity=0.7,
            ))
            fig.update_layout(
                height=260, margin=dict(l=0, r=0, t=20, b=0),
                paper_bgcolor="white", plot_bgcolor="white",
                xaxis=dict(showgrid=False, color="#9a9590", tickfont=dict(family="Geist Mono", size=9)),
                yaxis=dict(showgrid=True, gridcolor="#f0ece6", color="#9a9590", tickfont=dict(family="Geist Mono", size=10)),
                title=dict(text="Top Values", font=dict(family="Geist Mono", size=11, color="#9a9590")),
            )
            st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2: ISSUES
# ════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

    issue_cols = [c for c in A["columns"] if c["missing_pct"] > 0 or c["outliers"] > 0 or c["inconsistent"] > 0]
    if not issue_cols:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:var(--ink-3)">
          <div style="font-size:2.5rem;margin-bottom:.75rem">✦</div>
          <div style="font-family:'Instrument Serif',serif;font-size:18px;color:#16a34a;margin-bottom:.4rem">No issues found</div>
          <div style="font-family:'Geist Mono',monospace;font-size:11px">Your dataset looks clean across all dimensions.</div>
        </div>""", unsafe_allow_html=True)
    else:
        for c in sorted(issue_cols, key=lambda x: x["missing_pct"] + x["outliers"] + x["inconsistent"], reverse=True):
            issues = []
            if c["missing_pct"] > 0:
                issues.append(f"Missing: {c['missing_pct']:.1f}%")
            if c["outliers"] > 0:
                issues.append(f"Outliers: {c['outliers']}")
            if c["inconsistent"] > 0:
                issues.append(f"Inconsistent: {c['inconsistent']}")
            severity = "high" if c["missing_pct"] > 30 or c["outliers"] > 10 else "medium" if c["missing_pct"] > 5 else "low"
            st.markdown(f"""
            <div class="insight-card {severity}" style="margin-bottom:.6rem">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <span class="rec-col-n">{c['col']}</span>
                <span class="type-badge {c['type']}">{c['type']}</span>
              </div>
              <div class="rec-issue">{' · '.join(issues)}</div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3: ROOT CAUSE
# ════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown('<div class="sec-lbl">Issue Insights</div>', unsafe_allow_html=True)
        if recs:
            for r in recs[:8]:
                level = r["impact"].lower()
                icon = "🔴" if level == "high" else ("🟡" if level == "medium" else "🟢")
                st.markdown(f"""
                <div class="insight-card {level}">
                  <div class="ii {level}">{icon}</div>
                  <div class="it">{r['col']} — {r['issue']}</div>
                  <div class="id">{r['reason']}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-family:\'Geist Mono\',monospace;font-size:12px;color:#16a34a;padding:.5rem 0">✓ No major issues found</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="sec-lbl">Impact Ranking</div>', unsafe_allow_html=True)

        def col_impact(c):
            return c["missing_pct"] * 0.3 + (c["outliers"] / max(A["sample_size"], 1)) * 25 + (c["inconsistent"] / max(A["sample_size"], 1)) * 25

        ranked = sorted(A["columns"], key=col_impact, reverse=True)[:10]
        max_impact = max(col_impact(c) for c in ranked) or 1

        for i, c in enumerate(ranked):
            imp = col_impact(c)
            pct = imp / max_impact * 100
            color = score_color(100 - pct)
            st.markdown(f"""
            <div class="hs-row">
              <div class="hs-rank">{i+1}</div>
              <div class="hs-wrap">
                <div class="hs-col">{c['col']}</div>
                <div class="hs-track"><div class="hs-fill" style="width:{pct}%;background:{color}"></div></div>
              </div>
              <div class="hs-score">impact {imp:.1f}</div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4: CORRELATION
# ════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)
    num_names = A.get("num_col_names", [])

    if len(num_names) < 2:
        st.markdown('<div class="note-box">Need at least 2 numeric columns for a correlation heatmap.</div>', unsafe_allow_html=True)
    else:
        corr = A["corr_matrix"]
        matrix = [[corr[c1].get(c2, 0) for c2 in num_names] for c1 in num_names]

        fig = go.Figure(go.Heatmap(
            z=matrix,
            x=num_names,
            y=num_names,
            colorscale=[[0, "#991b1b"], [0.5, "#f5f3ef"], [1, "#1a6641"]],
            zmid=0, zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in matrix],
            texttemplate="%{text}",
            textfont=dict(family="Geist Mono", size=9),
            hoverongaps=False,
        ))
        fig.update_layout(
            height=max(300, len(num_names) * 40 + 100),
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor="white",
            xaxis=dict(tickfont=dict(family="Geist Mono", size=9), side="bottom"),
            yaxis=dict(tickfont=dict(family="Geist Mono", size=9), autorange="reversed"),
            coloraxis_showscale=True,
        )
        st.plotly_chart(fig, use_container_width=True)

        high_pairs = []
        for i, c1 in enumerate(num_names):
            for j, c2 in enumerate(num_names):
                if j > i:
                    r = corr[c1].get(c2, 0)
                    if abs(r) >= 0.7:
                        high_pairs.append((c1, c2, r))

        if high_pairs:
            st.markdown('<div class="sec-lbl" style="margin-top:1rem">High Correlation Pairs (|r| ≥ 0.7)</div>', unsafe_allow_html=True)
            for c1, c2, r in sorted(high_pairs, key=lambda x: abs(x[2]), reverse=True):
                color = "#991b1b" if r < 0 else "#1a6641"
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;padding:.5rem 0;border-bottom:1px solid #f0ece6;font-size:12px">
                  <span style="font-family:'Geist Mono',monospace;color:var(--ink-2)">{c1} × {c2}</span>
                  <span style="font-family:'Geist Mono',monospace;font-weight:600;color:{color}">r = {r:.3f}</span>
                </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 5: ML READINESS (Advanced)
# ════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

    if not ml or not mlr:
        st.markdown('<div class="note-box">ML analysis unavailable — scikit-learn or scipy may not be installed.</div>', unsafe_allow_html=True)
    else:
        ml_color_val = score_color(mlr["score"])

        # ── Row 1: Score + Factors ────────────────────────────────────────────
        top_left, top_right = st.columns([1, 1])
        with top_left:
            st.markdown(f"""
            <div style="text-align:center;padding:2rem;background:var(--bg-card);border:1px solid var(--rule);border-radius:var(--radius)">
              <div style="font-family:'Geist Mono',monospace;font-size:10px;letter-spacing:.15em;text-transform:uppercase;color:var(--ink-3);margin-bottom:.6rem">ML Readiness Score</div>
              <div style="font-family:'Instrument Serif',serif;font-size:3.8rem;font-weight:400;line-height:1;color:{ml_color_val}">{mlr['score']:.0f}</div>
              <div style="font-family:'Geist Mono',monospace;font-size:10px;color:var(--ink-3);margin-top:3px">{mlr['label']}</div>
              <div style="height:5px;background:var(--bg-sunken);border-radius:3px;margin:1rem 0 .6rem;overflow:hidden">
                <div style="width:{mlr['score']}%;height:100%;background:{ml_color_val};border-radius:3px"></div>
              </div>
              <div style="font-family:'Geist Mono',monospace;font-size:10px;color:var(--ink-3)">{mlr['num_features']} numeric · {mlr['cat_features']} categorical</div>
            </div>
            """, unsafe_allow_html=True)

            # Sub-scores breakdown
            st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)
            st.markdown('<div class="sec-lbl">Sub-score Breakdown</div>', unsafe_allow_html=True)
            weights_labels = {
                "completeness":     ("Completeness",    "30%"),
                "sample_size":      ("Sample Size",     "20%"),
                "feature_quality":  ("Feature Quality", "20%"),
                "cardinality":      ("Cardinality",     "10%"),
                "outliers":         ("Outliers",        "10%"),
                "duplicates":       ("Duplicates",      "5%"),
                "type_consistency": ("Type Consistency","5%"),
            }
            for key, (label, weight) in weights_labels.items():
                sub_val = mlr["sub_scores"].get(key, 0)
                bar_color = score_color(sub_val)
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:.75rem;padding:.4rem 0;border-bottom:1px solid #f0ece6">
                  <span style="font-family:'Geist Mono',monospace;font-size:10px;color:var(--ink-3);width:120px">{label} <span style="opacity:.5">{weight}</span></span>
                  <div style="flex:1;height:6px;background:var(--bg-sunken);border-radius:3px;overflow:hidden">
                    <div style="width:{sub_val}%;height:100%;background:{bar_color};border-radius:3px"></div>
                  </div>
                  <span style="font-family:'Geist Mono',monospace;font-size:11px;color:{bar_color};width:32px;text-align:right">{sub_val:.0f}</span>
                </div>""", unsafe_allow_html=True)

        with top_right:
            st.markdown('<div class="sec-lbl">Readiness Factors</div>', unsafe_allow_html=True)
            severity_color = {"high": "#991b1b", "medium": "#b45309", "low": "#1a6641"}
            severity_icon  = {"high": "🔴", "medium": "🟡", "low": "🟢"}

            if mlr.get("factors"):
                for f in mlr["factors"]:
                    sev = f["severity"]
                    st.markdown(f"""
                    <div style="display:flex;gap:1rem;padding:.55rem 0;border-bottom:1px solid #f0ece6;font-size:12px;align-items:flex-start">
                      <span>{severity_icon[sev]}</span>
                      <div>
                        <div style="font-weight:500;color:var(--ink)">{f['label']}</div>
                        <div style="color:var(--ink-3);font-size:11.5px;line-height:1.65;margin-top:2px">{f['detail']}</div>
                      </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown('<div style="font-family:\'Geist Mono\',monospace;font-size:12px;color:#16a34a;padding:.5rem 0">✓ No ML readiness blockers found</div>', unsafe_allow_html=True)

        if mlr.get("high_missing_cols"):
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="note-box">
              <strong>Columns requiring imputation before training:</strong> {', '.join(mlr['high_missing_cols'])}.<br>
              Most sklearn estimators will raise an error on NaN values. Use a <code>SimpleImputer</code> or <code>Pipeline</code>.
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

        # ── Row 2: Target Detection ───────────────────────────────────────────
        st.markdown('<div class="sec-lbl">Target Column Detection</div>', unsafe_allow_html=True)
        td = ml.get("target_detection", {})
        if td.get("detected"):
            conf_color = "#16a34a" if td["confidence"] >= 70 else "#b45309"
            task_badge_colors = {
                "binary_classification":    ("Binary Classification",    "#dbeafe", "#1d4ed8"),
                "multiclass_classification":("Multiclass Classification","#ede9ff", "#5b21b6"),
                "regression":               ("Regression",               "#dcfce7", "#1a6641"),
            }
            task_label, task_bg, task_fg = task_badge_colors.get(
                td.get("task_type", ""), (td.get("task_label","?"), "#f0ece6", "#4a4540")
            )
            ranked_html = "".join(
                f'<span style="font-family:\'Geist Mono\',monospace;font-size:10px;padding:2px 8px;border-radius:4px;border:1px solid var(--rule);color:var(--ink-3);margin-right:.4rem">{r["col"]} ({r["score"]})</span>'
                for r in td.get("ranked", [])[1:4]
            )
            st.markdown(f"""
            <div class="dq-card">
              <div class="dq-card-body" style="display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap">
                <div>
                  <div style="font-family:'Geist Mono',monospace;font-size:9.5px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.3rem">Detected Target</div>
                  <div style="font-size:15px;font-weight:600;color:var(--ink)">{td['col']}</div>
                </div>
                <span style="background:{task_bg};color:{task_fg};font-family:'Geist Mono',monospace;font-size:10px;font-weight:600;padding:3px 10px;border-radius:5px">{task_label}</span>
                <div>
                  <div style="font-family:'Geist Mono',monospace;font-size:9.5px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.3rem">Confidence</div>
                  <div style="font-family:'Instrument Serif',serif;font-size:1.6rem;line-height:1;color:{conf_color}">{td['confidence']}%</div>
                </div>
                <div>
                  <div style="font-family:'Geist Mono',monospace;font-size:9.5px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.4rem">Other Candidates</div>
                  <div>{ranked_html if ranked_html else '<span style="font-family:\'Geist Mono\',monospace;font-size:10px;color:var(--ink-3)">None</span>'}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="note-box">
              <strong>No target column detected.</strong> No column matched target heuristics strongly enough.
              If you have a target, rename it to something like <code>target</code>, <code>label</code>, or <code>y</code> for better detection.
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

        # ── Row 3: Feature Importance ─────────────────────────────────────────
        importance = ml.get("importance", [])
        st.markdown('<div class="sec-lbl">Feature Importance (3-Method Ensemble)</div>', unsafe_allow_html=True)
        if not importance:
            st.markdown("""
            <div class="note-box">
              Feature importance requires a detected target column and at least 30 non-null rows.
              scikit-learn and scipy must also be installed.
            </div>""", unsafe_allow_html=True)
        else:
            rec_colors = {
                "high":   ("#dcfce7", "#1a6641", "Keep — strong signal"),
                "medium": ("#fef3c7", "#b45309", "Keep — moderate signal"),
                "low":    ("#f0ece6", "#9a9590", "Consider — weak signal"),
                "drop":   ("#fee2e2", "#991b1b", "Drop — near-zero"),
            }
            feat_rows = ""
            for f in importance[:20]:
                level = f.get("rec_level", "low")
                bg, fg, _ = rec_colors.get(level, ("#f0ece6", "#9a9590", ""))
                bar_w = min(f["ensemble"] * 100, 100)
                feat_rows += f"""
                <tr>
                  <td style="font-family:'Geist Mono',monospace;font-size:11px;font-weight:500;color:var(--ink)">{f['rank']}. {f['col']}</td>
                  <td style="font-family:'Geist Mono',monospace;font-size:11px">{f['mi']:.3f}</td>
                  <td style="font-family:'Geist Mono',monospace;font-size:11px">{f['rf']:.3f}</td>
                  <td style="font-family:'Geist Mono',monospace;font-size:11px">{f['spearman']:.3f}</td>
                  <td>
                    <div style="display:flex;align-items:center;gap:.5rem">
                      <div style="width:60px;height:5px;background:var(--bg-sunken);border-radius:2px;overflow:hidden">
                        <div style="width:{bar_w}%;height:100%;background:var(--ink);border-radius:2px"></div>
                      </div>
                      <span style="font-family:'Geist Mono',monospace;font-size:11px;font-weight:600">{f['ensemble']:.3f}</span>
                    </div>
                  </td>
                  <td><span style="background:{bg};color:{fg};font-family:'Geist Mono',monospace;font-size:9.5px;padding:2px 7px;border-radius:4px;font-weight:600">{f['recommendation']}</span></td>
                </tr>"""

            st.markdown(f"""
            <div class="dq-card">
              <div class="dq-card-body" style="padding:0">
                <table class="dtable">
                  <thead><tr>
                    <th>Feature</th><th>MI</th><th>RF</th><th>Spearman</th><th>Ensemble ▾</th><th>Verdict</th>
                  </tr></thead>
                  <tbody>{feat_rows}</tbody>
                </table>
              </div>
            </div>""", unsafe_allow_html=True)

            # Feature importance chart
            top_feats = importance[:12]
            fig = go.Figure(go.Bar(
                x=[f["ensemble"] for f in reversed(top_feats)],
                y=[f["col"] for f in reversed(top_feats)],
                orientation="h",
                marker_color="#1a1612", marker_opacity=0.75,
            ))
            fig.update_layout(
                height=max(220, len(top_feats) * 28 + 60),
                margin=dict(l=0, r=0, t=20, b=0),
                paper_bgcolor="white", plot_bgcolor="white",
                xaxis=dict(showgrid=True, gridcolor="#f0ece6", color="#9a9590", tickfont=dict(family="Geist Mono", size=9), title="Ensemble Score"),
                yaxis=dict(showgrid=False, color="#9a9590", tickfont=dict(family="Geist Mono", size=9)),
                title=dict(text="Feature Importance (Ensemble)", font=dict(family="Geist Mono", size=11, color="#9a9590")),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

        # ── Row 4: Class Imbalance ────────────────────────────────────────────
        imbalance = ml.get("imbalance")
        if imbalance:
            st.markdown('<div class="sec-lbl">Class Imbalance Analysis</div>', unsafe_allow_html=True)

            if imbalance.get("task_type") == "regression":
                skew_val = imbalance.get("skew", 0)
                sev = imbalance.get("severity", "none")
                sev_color = {"none": "#16a34a", "low": "#b45309", "medium": "#b45309", "high": "#991b1b"}.get(sev, "#9a9590")
                st.markdown(f"""
                <div class="dq-card">
                  <div class="dq-card-body">
                    <div style="display:flex;gap:2rem;align-items:flex-start;flex-wrap:wrap">
                      <div>
                        <div style="font-family:'Geist Mono',monospace;font-size:9.5px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.3rem">Target Skew</div>
                        <div style="font-family:'Instrument Serif',serif;font-size:2rem;color:{sev_color}">{skew_val:.3f}</div>
                      </div>
                      <div>
                        <div style="font-family:'Geist Mono',monospace;font-size:9.5px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.3rem">Kurtosis</div>
                        <div style="font-family:'Instrument Serif',serif;font-size:2rem;color:var(--ink)">{imbalance.get('kurtosis', 0):.3f}</div>
                      </div>
                      <div style="flex:1;min-width:200px">
                        <div style="font-family:'Geist Mono',monospace;font-size:9.5px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.5rem">Recommendation</div>
                        <div style="font-size:12px;color:var(--ink-2);line-height:1.6">{imbalance.get('recommendation','')}</div>
                      </div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                sev = imbalance.get("severity", "none")
                sev_color = {"none": "#16a34a", "medium": "#b45309", "high": "#b45309", "critical": "#991b1b"}.get(sev, "#9a9590")
                classes = imbalance.get("classes", [])
                balance_score = imbalance.get("balance_score", 100)
                ratio = imbalance.get("imbalance_ratio", 1)

                imb_left, imb_right = st.columns([1, 1])
                with imb_left:
                    st.markdown(f"""
                    <div class="dq-card">
                      <div class="dq-card-body">
                        <div style="display:flex;gap:1.5rem;align-items:center;margin-bottom:1rem">
                          <div>
                            <div style="font-family:'Geist Mono',monospace;font-size:9.5px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.3rem">Imbalance Ratio</div>
                            <div style="font-family:'Instrument Serif',serif;font-size:2.2rem;line-height:1;color:{sev_color}">{ratio:.1f}:1</div>
                          </div>
                          <div>
                            <div style="font-family:'Geist Mono',monospace;font-size:9.5px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.3rem">Balance Score</div>
                            <div style="font-family:'Instrument Serif',serif;font-size:2.2rem;line-height:1;color:{score_color(balance_score)}">{balance_score:.0f}</div>
                          </div>
                          <div>
                            <div style="font-family:'Geist Mono',monospace;font-size:9.5px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.3rem">Severity</div>
                            <span style="background:{'#fee2e2' if sev in ('high','critical') else '#fef3c7' if sev=='medium' else '#dcfce7'};color:{sev_color};font-family:'Geist Mono',monospace;font-size:10px;padding:3px 9px;border-radius:4px;font-weight:600">{sev.upper()}</span>
                          </div>
                        </div>
                        {''.join(f"""<div style="display:flex;align-items:center;gap:.75rem;margin-bottom:5px"><span style="font-family:'Geist Mono',monospace;font-size:10px;color:var(--ink-2);width:90px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{cls['class'][:12]}</span><div style="flex:1;height:8px;background:var(--bg-sunken);border-radius:2px;overflow:hidden"><div style="width:{cls['pct']}%;height:100%;background:var(--ink);border-radius:2px;opacity:.6"></div></div><span style="font-family:'Geist Mono',monospace;font-size:10px;color:var(--ink-3);width:50px;text-align:right">{cls['pct']:.1f}%</span></div>""" for cls in classes[:8])}
                      </div>
                    </div>""", unsafe_allow_html=True)

                with imb_right:
                    strategies = imbalance.get("strategies", [])
                    if strategies:
                        st.markdown('<div class="sec-lbl">Recommended Strategies</div>', unsafe_allow_html=True)
                        for strat in strategies:
                            st.markdown(f"""
                            <div style="display:flex;gap:.75rem;padding:.5rem 0;border-bottom:1px solid #f0ece6;font-size:12px;align-items:flex-start">
                              <span style="color:#b45309;flex-shrink:0">→</span>
                              <span style="color:var(--ink-2);line-height:1.55">{strat}</span>
                            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

        # ── Row 5: Data Leakage ───────────────────────────────────────────────
        leakage = ml.get("leakage", [])
        st.markdown('<div class="sec-lbl">Data Leakage Detection</div>', unsafe_allow_html=True)
        if not leakage:
            st.markdown('<div style="font-family:\'Geist Mono\',monospace;font-size:12px;color:#16a34a;padding:.5rem 0">✓ No leakage suspects detected</div>', unsafe_allow_html=True)
        else:
            for leak in leakage:
                sev = leak.get("severity", "medium")
                card_class = "high" if sev == "high" else "medium"
                flags_html = "".join(f'<div style="font-size:11px;color:var(--ink-3);padding:.2rem 0;border-bottom:1px solid #f0ece6;line-height:1.55">⚠ {flag}</div>' for flag in leak["flags"])
                st.markdown(f"""
                <div class="insight-card {card_class}" style="margin-bottom:.6rem">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.4rem">
                    <span style="font-weight:600;font-size:13px;color:var(--ink)">{leak['col']}</span>
                    <span style="background:{'#fee2e2' if sev=='high' else '#fef3c7'};color:{'#991b1b' if sev=='high' else '#b45309'};font-family:'Geist Mono',monospace;font-size:9.5px;padding:2px 8px;border-radius:4px;font-weight:600">{sev.upper()} RISK</span>
                  </div>
                  {flags_html}
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

        # ── Row 6: PCA / Dimensionality ───────────────────────────────────────
        pca = ml.get("pca", {})
        st.markdown('<div class="sec-lbl">Dimensionality Reduction (PCA)</div>', unsafe_allow_html=True)
        if not pca.get("available"):
            reason = pca.get("reason", "scikit-learn not installed")
            st.markdown(f'<div class="note-box"><strong>PCA unavailable —</strong> {reason}</div>', unsafe_allow_html=True)
        else:
            pca_left, pca_right = st.columns([1, 1])
            with pca_left:
                st.markdown(f"""
                <div class="dq-card">
                  <div class="dq-card-body">
                    <div style="display:flex;gap:1.5rem;flex-wrap:wrap;margin-bottom:1rem">
                      <div>
                        <div style="font-family:'Geist Mono',monospace;font-size:9.5px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.3rem">Features In</div>
                        <div style="font-family:'Instrument Serif',serif;font-size:2rem;line-height:1;color:var(--ink)">{pca['n_features']}</div>
                      </div>
                      <div>
                        <div style="font-family:'Geist Mono',monospace;font-size:9.5px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.3rem">For 95% Variance</div>
                        <div style="font-family:'Instrument Serif',serif;font-size:2rem;line-height:1;color:#1d4ed8">{pca['components_for_95pct']} components</div>
                      </div>
                      <div>
                        <div style="font-family:'Geist Mono',monospace;font-size:9.5px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.3rem">Redundancy</div>
                        <div style="font-family:'Instrument Serif',serif;font-size:2rem;line-height:1;color:{score_color(pca['redundancy_score'])}">{pca['redundancy_score']:.0f}%</div>
                      </div>
                    </div>
                    <div style="font-family:'Geist Mono',monospace;font-size:10px;color:var(--ink-3);margin-bottom:.3rem">Components for 80% variance: {pca['components_for_80pct']} &nbsp;·&nbsp; 90%: {pca['components_for_90pct']} &nbsp;·&nbsp; 95%: {pca['components_for_95pct']}</div>
                    <div style="font-size:11.5px;color:var(--ink-2);margin-top:.6rem;line-height:1.55"><strong>t-SNE:</strong> {pca['tsne_recommendation']}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

            with pca_right:
                # Explained variance chart
                exp_var = pca.get("explained_variance", [])
                cum_var = pca.get("cumulative_variance", [])
                if exp_var:
                    x_vals = list(range(1, len(exp_var) + 1))
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=x_vals, y=exp_var,
                        name="Component variance",
                        marker_color="#1a1612", marker_opacity=0.5,
                    ))
                    fig.add_trace(go.Scatter(
                        x=x_vals, y=cum_var,
                        name="Cumulative",
                        line=dict(color="#1d4ed8", width=2),
                        mode="lines+markers",
                        marker=dict(size=5),
                    ))
                    fig.add_hline(y=95, line_dash="dash", line_color="#16a34a", line_width=1, opacity=0.6)
                    fig.update_layout(
                        height=220, margin=dict(l=0, r=0, t=20, b=0),
                        paper_bgcolor="white", plot_bgcolor="white",
                        legend=dict(font=dict(family="Geist Mono", size=9), orientation="h", y=-0.2),
                        xaxis=dict(showgrid=False, color="#9a9590", tickfont=dict(family="Geist Mono", size=9), title="Component"),
                        yaxis=dict(showgrid=True, gridcolor="#f0ece6", color="#9a9590", tickfont=dict(family="Geist Mono", size=9), title="% Variance", range=[0, 105]),
                        title=dict(text="Explained Variance per Component", font=dict(family="Geist Mono", size=11, color="#9a9590")),
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Top component loadings
            loadings = pca.get("component_loadings", [])
            if loadings:
                st.markdown('<div class="sec-lbl" style="margin-top:1rem">Top Feature Loadings per Component</div>', unsafe_allow_html=True)
                loading_cols = st.columns(min(len(loadings), 5))
                for i, comp in enumerate(loadings[:5]):
                    with loading_cols[i]:
                        feats_html = "".join(
                            f'<div style="font-family:\'Geist Mono\',monospace;font-size:10px;padding:.3rem 0;border-bottom:1px solid #f0ece6;display:flex;justify-content:space-between"><span style="color:var(--ink-2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:80px">{fl["feature"][:12]}</span><span style="color:{"#1a6641" if fl["loading"] > 0 else "#991b1b"};font-weight:500">{fl["loading"]:+.2f}</span></div>'
                            for fl in comp["top_features"]
                        )
                        st.markdown(f"""
                        <div style="background:var(--bg-card);border:1px solid var(--rule);border-radius:8px;padding:.75rem">
                          <div style="font-family:'Geist Mono',monospace;font-size:9px;color:var(--ink-3);letter-spacing:.1em;text-transform:uppercase;margin-bottom:.4rem">PC{comp['component']} · {comp['explained_variance_pct']:.1f}%</div>
                          {feats_html}
                        </div>""", unsafe_allow_html=True)

        # sklearn/scipy availability notice
        if not ml.get("sklearn_available") or not ml.get("scipy_available"):
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div class="note-box">
              <strong>ⓘ Partial analysis —</strong> Some advanced ML features require
              <code>scikit-learn</code> and <code>scipy</code>. Run <code>pip install scikit-learn scipy</code> for the full experience.
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 6: RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════════════

with tabs[5]:
    st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

    if not recs:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:var(--ink-3)">
          <div style="font-size:2.5rem;margin-bottom:.75rem">✦</div>
          <div style="font-family:'Instrument Serif',serif;font-size:18px;color:#16a34a;margin-bottom:.4rem">No major issues found</div>
          <div style="font-family:'Geist Mono',monospace;font-size:11px">Your dataset appears clean. Keep monitoring as data grows.</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="font-family:\'Geist Mono\',monospace;font-size:11px;color:var(--ink-3);margin-bottom:1.25rem">{len(recs)} issue(s) · prioritized by impact</div>', unsafe_allow_html=True)

        for i, r in enumerate(recs):
            level = r["impact"].lower()
            badge_class = {"high": "rb-h", "medium": "rb-m", "low": "rb-l"}[level]
            with st.expander(f"{'🔴' if level=='high' else '🟡' if level=='medium' else '🟢'}  {r['col']} — {r['issue']}", expanded=(i < 3)):
                st.markdown(f"""
                <div>
                  <div style="margin-bottom:.5rem"><span class="rec-col-n">{r['col']}</span><span class="rec-badge {badge_class}">{r['impact']}</span></div>
                  <div class="rec-strat">Strategy: {r['strategy']}</div>
                  <div class="rec-reason">{r['reason']}</div>
                </div>""", unsafe_allow_html=True)
                if r.get("code"):
                    st.code(r["code"], language="python")

        st.markdown("""
        <div class="note-box" style="margin-top:1rem">
          <strong>ⓘ Confidence note —</strong> Recommendations are based on statistical heuristics (IQR 1.5× for outliers, threshold-based missingness). Domain knowledge may warrant different strategies.
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 7: CUSTOM RULES
# ════════════════════════════════════════════════════════════════════════════

with tabs[6]:
    st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sec-lbl">Define Custom Validation Rules</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'Geist Mono\',monospace;font-size:11px;color:var(--ink-3);margin-bottom:1.25rem">Run domain-specific rules beyond the default statistical checks.</div>', unsafe_allow_html=True)

    col_names = [c["col"] for c in A["columns"]]
    rule_col1, rule_col2, rule_col3 = st.columns(3)

    with rule_col1:
        rule_col = st.selectbox("Column", col_names, key="rule_col")
    with rule_col2:
        rule_type = st.selectbox("Rule type", [
            "not_null", "min_value", "max_value", "range", "one_of", "regex", "unique"
        ], key="rule_type")
    with rule_col3:
        needs_param = rule_type not in ("not_null", "unique")
        param_help = {
            "min_value": "e.g. 0", "max_value": "e.g. 100",
            "range": "e.g. 0, 100", "one_of": "e.g. M, F, N/A",
            "regex": "e.g. ^\\d{4}$"
        }
        rule_param = st.text_input(
            "Parameter(s)",
            placeholder=param_help.get(rule_type, "—"),
            disabled=not needs_param,
            key="rule_param"
        )

    if st.button("▶ Run Rule", type="primary"):
        vals = df[rule_col].tolist()
        passes, fails, fail_examples = 0, 0, []

        for v in vals:
            ok = True
            sv = str(v).strip() if v is not None else ""
            is_miss_val = sv == "" or sv.lower() in ("nan", "none", "null", "na", "n/a")

            if rule_type == "not_null":
                ok = not is_miss_val
            elif rule_type == "min_value":
                try: ok = not is_miss_val and float(sv) >= float(rule_param)
                except: ok = False
            elif rule_type == "max_value":
                try: ok = not is_miss_val and float(sv) <= float(rule_param)
                except: ok = False
            elif rule_type == "range":
                try:
                    mn, mx = [float(x.strip()) for x in rule_param.split(",")]
                    ok = not is_miss_val and mn <= float(sv) <= mx
                except: ok = False
            elif rule_type == "one_of":
                allowed = [x.strip().lower() for x in rule_param.split(",")]
                ok = sv.lower() in allowed
            elif rule_type == "regex":
                import re
                try: ok = bool(re.match(rule_param, sv))
                except: ok = False
            elif rule_type == "unique":
                ok = True  # handled separately

            if ok: passes += 1
            else:
                fails += 1
                if len(fail_examples) < 3: fail_examples.append(v)

        if rule_type == "unique":
            unique_vals = set(str(v) for v in vals if v is not None)
            fails = len(vals) - len(unique_vals)
            passes = len(unique_vals)

        total = passes + fails
        pct = passes / max(total, 1) * 100
        passed = fails == 0

        result = {
            "col": rule_col, "type": rule_type, "param": rule_param,
            "passes": passes, "fails": fails, "pct": round(pct, 1),
            "examples": fail_examples, "passed": passed,
        }
        st.session_state.rule_history.insert(0, result)

        if passed:
            st.success(f"✓ PASS — {passes:,} passed, 0 failed ({pct:.1f}% pass rate)")
        else:
            ex = ", ".join(f'"{e}"' for e in fail_examples)
            st.error(f"✗ FAIL — {passes:,} passed, {fails:,} failed ({pct:.1f}% pass rate){' · Examples: ' + ex if ex else ''}")

    if st.session_state.rule_history:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="dq-card"><div class="dq-card-hdr"><span class="dq-card-title">Rule Run History</span></div><div class="dq-card-body">', unsafe_allow_html=True)
        for r in st.session_state.rule_history[:10]:
            color = "#16a34a" if r["passed"] else "#dc2626"
            label = "✓ PASS" if r["passed"] else "✗ FAIL"
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;padding:.5rem 0;border-bottom:1px solid #f0ece6;font-size:11px">
              <span style="font-family:'Geist Mono',monospace;color:var(--ink-2)">{r['col']} · {r['type']}{' ('+r['param']+')' if r['param'] else ''}</span>
              <span style="font-family:'Geist Mono',monospace;font-weight:500;color:{color}">{label} {r['pct']}%</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

        if st.button("Clear history", type="secondary"):
            st.session_state.rule_history = []
            st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# TAB 8: SCHEMA DRIFT
# ════════════════════════════════════════════════════════════════════════════

with tabs[7]:
    st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sec-lbl">Schema Drift Detection</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'Geist Mono\',monospace;font-size:11px;color:var(--ink-3);margin-bottom:1.25rem">Compare this dataset\'s schema and distributions against a second CSV to detect structural changes.</div>', unsafe_allow_html=True)

    drift_file = st.file_uploader("Upload comparison CSV", type=["csv"], key="drift_upload")

    if drift_file:
        with st.spinner("Comparing schemas…"):
            try:
                df2 = pd.read_csv(drift_file, encoding="utf-8", low_memory=False)
                analysis2 = analyze_dataset(df2, drift_file.name)
                drift = compare_schemas(A, analysis2, A["file_name"], drift_file.name)
                st.session_state.drift_analysis = drift
            except Exception as e:
                st.error(f"Could not parse comparison CSV: {e}")

    if st.session_state.drift_analysis:
        drift = st.session_state.drift_analysis

        d1, d2, d3 = st.columns(3)
        with d1:
            st.markdown(f"""<div class="stat-chip"><div class="si">+</div><div class="sv" style="color:#16a34a">{len(drift['added'])}</div><div class="sl">Added Columns</div></div>""", unsafe_allow_html=True)
        with d2:
            st.markdown(f"""<div class="stat-chip"><div class="si">−</div><div class="sv" style="color:#dc2626">{len(drift['removed'])}</div><div class="sl">Removed Columns</div></div>""", unsafe_allow_html=True)
        with d3:
            st.markdown(f"""<div class="stat-chip"><div class="si">~</div><div class="sv" style="color:#b45309">{len(drift['changes'])}</div><div class="sl">Changed Columns</div></div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown(f'<div class="dq-card"><div class="dq-card-hdr"><span class="dq-card-title">Drift Report — {drift["name_a"]} vs {drift["name_b"]}</span></div><div class="dq-card-body">', unsafe_allow_html=True)

        all_cols = (
            [(c, "added", "New column") for c in drift["added"]] +
            [(c, "removed", "Column no longer exists") for c in drift["removed"]] +
            [(c["col"], "changed", " · ".join(c["diffs"])) for c in drift["changes"]] +
            [(c, "ok", "No changes detected") for c in drift["stable"]]
        )

        for col_name, status, detail in all_cols:
            st.markdown(f"""
            <div class="diff-row">
              <span class="diff-col">{col_name}</span>
              <span class="diff-chip diff-{status}">{status}</span>
              <span style="font-size:11.5px;color:var(--ink-3);margin-left:.5rem">{detail}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 9: AI AGENT
# ════════════════════════════════════════════════════════════════════════════

with tabs[8]:
    st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

    if not st.session_state.api_key:
        st.markdown("""
        <div class="note-box">
          <strong>ⓘ API key required —</strong> Go back to the upload screen and enter your Anthropic API key to use the AI agent.
          Alternatively, set the <code>ANTHROPIC_API_KEY</code> environment variable before launching the app.
        </div>""", unsafe_allow_html=True)
        st.text_input("Anthropic API Key", type="password", key="inline_key",
                      placeholder="sk-ant-...", on_change=lambda: st.session_state.update(api_key=st.session_state.inline_key))
    else:
        suggestions = [
            "What does my DQS score mean?",
            "Which columns need the most attention?",
            "Is this data ready for ML?",
            "Explain the missing values pattern",
            "Generate a cleaning priority list",
            "What could cause the outliers?",
            "Which features are most important?",
            "Any data leakage risks?",
        ]

        sug_cols = st.columns(4)
        for i, s in enumerate(suggestions):
            with sug_cols[i % 4]:
                if st.button(s, key=f"sug_{i}", type="secondary"):
                    st.session_state.chat_history.append({"role": "user", "content": s})
                    with st.spinner("Thinking…"):
                        reply = get_response(A, recs, st.session_state.chat_history, st.session_state.api_key, ml=ml)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    st.rerun()

        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

        if not st.session_state.chat_history:
            st.markdown("""
            <div class="msg ai" style="max-width:80%">
              👋 Hi! I'm your data quality analyst. I've read your dataset — ask me anything about it.
              I can explain scores, identify risks, prioritize fixes, assess ML readiness, and interpret feature importance.
            </div>""", unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                role_class = "user" if msg["role"] == "user" else "ai"
                st.markdown(f'<div class="msg {role_class}">{msg["content"]}</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        user_input = st.text_input(
            "Ask about your dataset…",
            key="chat_input",
            label_visibility="collapsed",
            placeholder="Ask about your dataset…",
        )

        send_col, clear_col = st.columns([5, 1])
        with send_col:
            if st.button("Send", type="primary", use_container_width=True) and user_input.strip():
                st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
                with st.spinner("Thinking…"):
                    reply = get_response(A, recs, st.session_state.chat_history, st.session_state.api_key, ml=ml)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.rerun()
        with clear_col:
            if st.button("Clear", type="secondary", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

# ── EXPORT ───────────────────────────────────────────────────────────────────

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

export_col1, export_col2, export_col3 = st.columns([1, 1, 4])

with export_col1:
    date_str = pd.Timestamp.now().strftime("%B %d, %Y")

    column_rows = ""
    for c in A['columns']:
        color = "#d97706" if c["outliers"] > 0 else "#9a9590"
        column_rows += f"<tr><td>{c['col']}</td><td class='meta'>{c['type']}</td><td style='color:{miss_color(c['missing_pct'])}'>{c['missing_pct']:.1f}%</td><td style='color:{color}'>{c['outliers'] or '—'}</td></tr>"

    recs_rows = ""
    if recs:
        for r in recs:
            badge_class = "poor" if r["impact"] == "High" else "mod" if r["impact"] == "Medium" else "good"
            recs_rows += f"<tr><td style='font-weight:500'>{r['col']}</td><td style='color:#9a9590'>{r['issue']}</td><td><span class='badge {badge_class}'>{r['impact']}</span></td><td>{r['strategy']}</td></tr>"
        recs_section = f"<h2>Recommendations</h2><table><tr><th>Column</th><th>Issue</th><th>Impact</th><th>Strategy</th></tr>{recs_rows}</table>"
    else:
        recs_section = ""

    # ML section for export
    ml_export_section = ""
    if ml and mlr:
        td = ml.get("target_detection", {})
        target_line = f"<p class='meta'>Detected target: <strong>{td['col']}</strong> ({td.get('task_label','?')}, {td.get('confidence',0)}% confidence)</p>" if td.get("detected") else ""
        ml_export_section = f"""
        <h2>ML Readiness</h2>
        <p>Score: <strong>{mlr['score']:.0f}/100</strong> — {mlr['label']}</p>
        {target_line}
        """

    report_html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>DQ Report — {A['file_name']}</title>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist+Mono:wght@300;400;500;600&family=Geist:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>body{{font-family:'Geist',sans-serif;max-width:900px;margin:40px auto;color:#1a1612;padding:0 2rem;background:#f5f3ef}}
h1{{font-family:'Instrument Serif',serif;font-size:2.5rem;letter-spacing:-.02em;margin-bottom:.25rem;font-weight:400}}
h2{{font-size:1rem;letter-spacing:.12em;text-transform:uppercase;color:#9a9590;font-weight:400;margin:2rem 0 .75rem;border-bottom:1px solid #dbd6cf;padding-bottom:.4rem;font-family:'Geist Mono',monospace}}
table{{width:100%;border-collapse:collapse;font-size:12px;margin-bottom:1.5rem}}
th{{padding:.5rem .75rem;text-align:left;font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:#9a9590;border-bottom:1px solid #dbd6cf;font-family:'Geist Mono',monospace}}
td{{padding:.5rem .75rem;border-bottom:1px solid #f0ece6}}
.badge{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:10px;font-family:'Geist Mono',monospace;font-weight:600}}
.good{{background:#dcfce7;color:#1a6641}}.mod{{background:#fef3c7;color:#b45309}}.poor{{background:#fee2e2;color:#991b1b}}
.meta{{font-family:'Geist Mono',monospace;font-size:11px;color:#9a9590}}</style></head><body>
<p class="meta">{date_str} · Generated by DQ·INTEL</p>
<h1>{A['DQS']:.0f} <span style="font-size:1.5rem;color:#9a9590">/ 100</span></h1>
<p class="meta">Dataset: {A['file_name']} · {A['total_rows']:,} rows · {A['num_cols']} columns</p>
<span class="badge {'good' if A['DQS']>=80 else 'mod' if A['DQS']>=60 else 'poor'}">{score_label(A['DQS'])}</span>
<h2>Dimension Scores</h2>
<table><tr><th>Dimension</th><th>Weight</th><th>Score</th><th>Meaning</th></tr>
{''.join(f"<tr><td>{l}</td><td class='meta'>{w}</td><td style='font-weight:600;color:{score_color(s)}'>{s:.0f}</td><td style='color:#9a9590'>{d}</td></tr>" for l,w,s,d in dims)}
</table>
<h2>Column Summary</h2>
<table><tr><th>Column</th><th>Type</th><th>Missing %</th><th>Outliers</th></tr>
{column_rows}
</table>
{recs_section}
{ml_export_section}
<p class="meta" style="margin-top:2rem;border-top:1px solid #dbd6cf;padding-top:1rem">DQS = 0.30·C + 0.25·Co + 0.25·V + 0.20·U = {A['DQS']:.2f} · Sample: {A['sample_size']:,} rows</p>
</body></html>"""

    st.download_button(
        "↓ Export Report",
        data=report_html,
        file_name=f"DQ_Report_{A['file_name'].replace('.csv','')}_{pd.Timestamp.now().strftime('%Y-%m-%d')}.html",
        mime="text/html",
        type="secondary",
    )

with export_col2:
    if recs:
        recs_df = pd.DataFrame([{k: v for k, v in r.items() if k != "code"} for r in recs])
        csv_buf = io.StringIO()
        recs_df.to_csv(csv_buf, index=False)
        st.download_button(
            "↓ Recs as CSV",
            data=csv_buf.getvalue(),
            file_name=f"DQ_Recs_{A['file_name'].replace('.csv','')}_{pd.Timestamp.now().strftime('%Y-%m-%d')}.csv",
            mime="text/csv",
            type="secondary",
        )