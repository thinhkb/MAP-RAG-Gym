"""
MAP-RAG-Gym — Streamlit Demo Dashboard
"""
import sys, os, json, pathlib

# Ensure map_rag_gym is importable even without editable install
_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = pathlib.Path(__file__).resolve().parent
OUT  = ROOT / "outputs"
MET  = OUT / "metrics"
VIZ  = OUT / "report_visualizations"
ARCH_IMG = ROOT / "Baocao" / "ChatGPT Image 20_37_09 4 thg 6, 2026.png"

# ── helpers ────────────────────────────────────────────────────────
@st.cache_data
def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_csv(p):
    return pd.read_csv(p)

def safe_csv(name):
    p = MET / name
    return load_csv(p) if p.exists() else None

def safe_json(name):
    p = OUT / name
    return load_json(p) if p.exists() else None


def _infer_experiment_meta(exp_dir: pathlib.Path):
    """Infer dataset/model/provider from output folder name and logs."""
    if exp_dir == OUT:
        return {
            "experiment": "hotpotqa_large_llama3.2_baseline",
            "dataset": "hotpotqa_large",
            "provider": "ollama",
            "model": "llama3.2",
            "output_dir": str(exp_dir.relative_to(ROOT)),
        }

    name = exp_dir.name
    dataset = "hotpotqa_large" if "hotpotqa" in name else ("2wikimultihopqa" if "2wiki" in name else "unknown")
    provider = "ollama"
    model = "unknown"
    if "gemini" in name:
        provider, model = "gemini", "gemini-2.0-flash"
    elif "gemma3.4b" in name or "gemma3_4b" in name or "genma3.4b" in name:
        provider, model = "ollama", "gemma3:4b"
    elif "gemma3" in name:
        provider, model = "ollama", "gemma3"
    elif "llama" in name:
        provider, model = "ollama", "llama3.2"

    log = exp_dir / "run_all.log"
    train_limit, eval_limit = None, None
    if log.exists():
        try:
            text = log.read_text(encoding="utf-8", errors="ignore")[:3000]
            import re
            m = re.search(r"Dataset:\s*([^|\n]+)", text)
            if m:
                dataset = m.group(1).strip()
            m = re.search(r"(?:LLM Model\s*:|ollama/)([\w.:-]+)", text)
            if m and model == "unknown":
                model = m.group(1).strip()
            m = re.search(r"Train(?: limit)?\s*[=:]\s*(\d+)", text)
            if m:
                train_limit = int(m.group(1))
            m = re.search(r"Eval(?: limit)?\s*[=:]\s*(\d+)", text)
            if m:
                eval_limit = int(m.group(1))
        except Exception:
            pass

    return {
        "experiment": name,
        "dataset": dataset,
        "provider": provider,
        "model": model,
        "train_limit": train_limit,
        "eval_limit": eval_limit,
        "output_dir": str(exp_dir.relative_to(ROOT)),
    }


@st.cache_data
def discover_experiments():
    """Scan outputs/ and collect completed experiment metrics."""
    rows, configs, critic_rows, gate_rows = [], [], [], []
    candidates = []
    if (OUT / "metrics" / "metrics_macro_budget_summary.csv").exists():
        candidates.append(OUT)
    if OUT.exists():
        for d in sorted([p for p in OUT.iterdir() if p.is_dir()]):
            if (d / "metrics" / "metrics_macro_budget_summary.csv").exists():
                candidates.append(d)

    for exp_dir in candidates:
        meta = _infer_experiment_meta(exp_dir)
        metrics_dir = exp_dir / "metrics"
        budget_path = metrics_dir / "metrics_macro_budget_summary.csv"
        try:
            bdf = pd.read_csv(budget_path)
            for _, r in bdf.iterrows():
                row = dict(meta)
                row.update(r.to_dict())
                rows.append(row)

            best = bdf.loc[bdf["avg_utility"].idxmax()].to_dict()
            cfg = dict(meta)
            cfg.update({
                "best_budget": best.get("budget_mode"),
                "best_utility": best.get("avg_utility"),
                "best_em": best.get("avg_em"),
                "best_f1": best.get("avg_f1_proxy"),
                "best_tokens": best.get("avg_tokens"),
                "best_latency_ms": best.get("avg_latency_ms"),
                "num_budgets": len(bdf),
                "completed": True,
            })
            configs.append(cfg)
        except Exception:
            continue

        cp = metrics_dir / "metrics_micro_critic_summary.csv"
        if cp.exists():
            cdf = pd.read_csv(cp)
            for _, r in cdf.iterrows():
                row = dict(meta)
                row.update(r.to_dict())
                critic_rows.append(row)

        sp = metrics_dir / "metrics_system_overview.csv"
        if sp.exists():
            sdf = pd.read_csv(sp)
            for _, r in sdf.iterrows():
                row = dict(meta)
                row.update(r.to_dict())
                gate_rows.append(row)

    return {
        "budget": pd.DataFrame(rows),
        "configs": pd.DataFrame(configs),
        "critics": pd.DataFrame(critic_rows),
        "gates": pd.DataFrame(gate_rows),
    }


def fmt_num(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return x

COLORS = {
    "low": "#6366f1", "medium": "#f59e0b", "high": "#10b981",
    "bg": "#0e1117", "card": "#1a1d23", "accent": "#818cf8",
}

# ── page config ────────────────────────────────────────────────────
st.set_page_config(page_title="MAP-RAG-Gym", page_icon="🧠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.metric-card {
    background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
    border-radius: 16px; padding: 24px; border: 1px solid #333;
    text-align: center; margin-bottom: 8px;
}
.metric-card h3 { color: #a5b4fc; font-size: 14px; margin: 0 0 8px; text-transform: uppercase; letter-spacing: 1px; }
.metric-card .value { color: #e0e7ff; font-size: 32px; font-weight: 700; }
.metric-card .sub { color: #94a3b8; font-size: 12px; margin-top: 4px; }
.status-pass { color: #34d399; font-weight: 700; }
.status-fail { color: #f87171; font-weight: 700; }
.hero { text-align: center; padding: 20px 0 10px; }
.hero h1 { font-size: 2.4rem; background: linear-gradient(90deg, #818cf8, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero p { color: #94a3b8; font-size: 1.05rem; max-width: 700px; margin: 0 auto; }
div[data-testid="stTabs"] button { font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

# ── hero ───────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
<h1>🧠 MAP-RAG-Gym</h1>
<p>Adaptive Multi-Workflow RAG with Process Evaluation<br/>
Kết hợp <b>MAO-ARAG</b> (macro routing) và <b>RAG-Gym</b> (micro critic)</p>
</div>
""", unsafe_allow_html=True)

# ── sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Navigation")
    st.markdown("Use the tabs below to explore different layers of the system.")
    st.divider()
    st.markdown("### 📊 Quick Stats")
    regate = safe_json("regate_report.json")
    if regate:
        online = regate.get("ready_for_online_rl", False)
        st.markdown(f"**Online RL:** {'✅ READY' if online else '❌ Not ready'}")
    budget_df = safe_csv("metrics_macro_budget_summary.csv")
    if budget_df is not None:
        best = budget_df.loc[budget_df["avg_utility"].idxmax()]
        st.markdown(f"**Best Budget:** {best['budget_mode'].upper()} ({best['avg_utility']:.4f})")
    st.divider()
    st.markdown("### 🏗️ Workflow Library")
    wf_data = {
        "W1": "AG → Direct answer",
        "W2": "QR→RA→AG → Rewrite+retrieve",
        "W3": "RA→DS→AG → Retrieve+select",
        "W4": "QDP→RA→AS → Parallel decomp",
        "W5": "QDS→QR→RA→AS → Serial decomp",
        "W6": "DRAFT→REFLECT→RA→AG → Reflective",
    }
    for wf, desc in wf_data.items():
        st.markdown(f"**{wf}**: {desc}")

# ── tabs ───────────────────────────────────────────────────────────
tabs = st.tabs([
    "📈 Overview",
    "🏗️ Architecture",
    "🏆 Experiment Comparison",
    "🔀 Macro Layer",
    "🔬 Micro Layer",
    "🚦 System Gates",
    "📊 Visualizations",
    "🏃 Live Demo",
])

# ━━━━ TAB 0: Overview ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[0]:
    st.subheader("System Overview")
    budget_df = safe_csv("metrics_macro_budget_summary.csv")
    if budget_df is not None:
        cols = st.columns(3)
        for i, (_, row) in enumerate(budget_df.iterrows()):
            bm = row["budget_mode"].upper()
            c = [COLORS["low"], COLORS["medium"], COLORS["high"]][i % 3]
            with cols[i]:
                st.markdown(f"""<div class="metric-card">
                    <h3 style="color:{c}">{bm} Budget</h3>
                    <div class="value">{row['avg_utility']:.4f}</div>
                    <div class="sub">EM: {row['avg_em']:.4f} | F1: {row['avg_f1_proxy']:.4f} | Tokens: {row['avg_tokens']:.0f}</div>
                </div>""", unsafe_allow_html=True)

    # Architecture diagram
    st.markdown("#### System Architecture")
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                     MAP-RAG-Gym System                      │
    ├──────────────────────┬──────────────────────────────────────┤
    │   MACRO LAYER        │   MICRO LAYER                       │
    │                      │                                      │
    │  Question + Budget   │   Process Critic (QR, AG)           │
    │       ↓              │       ↓                              │
    │  Router / Bandit     │   Candidate Reranking               │
    │       ↓              │       ↓                              │
    │  Workflow Selection   │   Selective Deployment (gate=0.70)  │
    │  (W1–W6)            │                                      │
    ├──────────────────────┴──────────────────────────────────────┤
    │   SYSTEM LAYER: Offline RL → Promotion Gate → Online RL    │
    └─────────────────────────────────────────────────────────────┘
    ```
    """)

    # Gate summary
    sys_df = safe_csv("metrics_system_overview.csv")
    if sys_df is not None:
        st.markdown("#### Deployment Gates")
        gate_rows = sys_df[sys_df["category"].isin(["Bandit Gate", "Critic Gate", "RL Gate"])]
        c1, c2 = st.columns(2)
        with c1:
            for _, r in gate_rows.iterrows():
                val = str(r["value"])
                icon = "✅" if val.lower() == "true" else ("❌" if val.lower() == "false" else "ℹ️")
                st.markdown(f"{icon} **{r['category']}** — {r['metric']}: `{val}`")
        with c2:
            promo_df = safe_csv("metrics_system_promotion.csv")
            if promo_df is not None:
                st.markdown("**Promotion Status**")
                st.dataframe(promo_df[["budget_mode","passed","candidate_utility","frozen_utility","utility_delta"]], hide_index=True)

# ━━━━ TAB 1: Architecture ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[1]:
    st.subheader("🏗️ MAP-RAG-Gym System Architecture")
    if ARCH_IMG.exists():
        st.image(str(ARCH_IMG), caption="MAP-RAG-Gym: Adaptive Multi-Workflow RAG with Process Evaluation", use_container_width=True)
    else:
        st.warning(f"Architecture image not found: `{ARCH_IMG}`")

    st.markdown("""
    ### Main Processing Flow
    1. **Input question + budget** enters the macro router.
    2. **Macro layer** selects workflow W1/W2/W3/W6 based on budget constraints.
    3. **Micro layer** applies process critic (QR/AG) when the deployment gate allows.
    4. **System layer** checks Offline RL, Promotion Gate, Bandit Gate, Critic Gate.
    5. Final output includes answer, utility, EM/F1 proxy, token cost, and latency.
    """)

    st.markdown("#### Workflow Library")
    wf_table = pd.DataFrame([
        {"Workflow": "W1", "Pipeline": "AG", "Role": "Direct answer / lowest cost"},
        {"Workflow": "W2", "Pipeline": "QR → RA → AG", "Role": "Rewrite + retrieve + answer"},
        {"Workflow": "W3", "Pipeline": "RA → DS → AG", "Role": "Retrieve + document selection + answer"},
        {"Workflow": "W4", "Pipeline": "QDP → RA → AS", "Role": "Parallel question decomposition"},
        {"Workflow": "W5", "Pipeline": "QDS → QR → RA → AS", "Role": "Serial decomposition"},
        {"Workflow": "W6", "Pipeline": "DRAFT → REFLECT → RA → AG", "Role": "Reflective RAG"},
    ])
    st.dataframe(wf_table, hide_index=True, use_container_width=True)

# ━━━━ TAB 2: Experiment Comparison ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[2]:
    st.subheader("🏆 Experiment Results — All Trained Runs in `outputs/`")
    exp = discover_experiments()
    budget_all = exp["budget"]
    cfg_all = exp["configs"]
    critic_all = exp["critics"]
    gate_all = exp["gates"]

    if budget_all.empty:
        st.warning("No completed experiments found in `outputs/**/metrics/metrics_macro_budget_summary.csv`.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Experiments", cfg_all["experiment"].nunique())
        c2.metric("Datasets", cfg_all["dataset"].nunique())
        c3.metric("Models", cfg_all["model"].nunique())
        best_idx = budget_all["avg_utility"].astype(float).idxmax()
        best_row = budget_all.loc[best_idx]
        c4.metric("Best Utility", f"{best_row['avg_utility']:.4f}", f"{best_row['experiment']} / {best_row['budget_mode']}")

        st.markdown("#### Training Configurations")
        show_cfg = cfg_all[[
            "experiment", "dataset", "provider", "model", "train_limit", "eval_limit",
            "best_budget", "best_utility", "best_em", "best_f1", "best_tokens", "best_latency_ms", "output_dir"
        ]].copy()
        st.dataframe(show_cfg.sort_values("best_utility", ascending=False), hide_index=True, use_container_width=True)

        st.markdown("#### Budget-Level Comparison Table")
        show_budget = budget_all[[
            "experiment", "dataset", "provider", "model", "budget_mode", "recommended_method",
            "num_runs", "avg_utility", "avg_em", "avg_f1_proxy", "avg_process_score",
            "avg_tokens", "avg_latency_ms", "workflow_distribution", "zero_utility_rate"
        ]].copy()
        st.dataframe(show_budget.sort_values(["avg_utility"], ascending=False), hide_index=True, use_container_width=True)

        st.markdown("#### Utility by Experiment & Budget")
        fig_cmp = px.bar(
            budget_all,
            x="experiment",
            y="avg_utility",
            color="budget_mode",
            barmode="group",
            hover_data=["dataset", "model", "avg_em", "avg_f1_proxy", "avg_tokens", "avg_latency_ms"],
            color_discrete_map={"low": "#6366f1", "medium": "#f59e0b", "high": "#10b981"},
            template="plotly_dark",
        )
        fig_cmp.update_layout(height=520, xaxis_tickangle=-25, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cmp, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig_cost = px.scatter(
                budget_all,
                x="avg_tokens",
                y="avg_utility",
                color="model",
                symbol="budget_mode",
                size="avg_latency_ms",
                hover_name="experiment",
                hover_data=["dataset", "avg_em", "avg_f1_proxy"],
                title="Utility vs Token Cost",
                template="plotly_dark",
            )
            fig_cost.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_cost, use_container_width=True)
        with c2:
            fig_lat = px.scatter(
                budget_all,
                x="avg_latency_ms",
                y="avg_utility",
                color="dataset",
                symbol="model",
                size="avg_tokens",
                hover_name="experiment",
                title="Utility vs Latency",
                template="plotly_dark",
            )
            fig_lat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_lat, use_container_width=True)

        st.markdown("#### Micro Critic Summary")
        if not critic_all.empty:
            critic_cols = ["experiment", "dataset", "model", "module", "eval_spearman", "eval_pearson", "eval_mae", "eval_rmse", "ready_as_offline_reward_model"]
            st.dataframe(critic_all[critic_cols].sort_values(["eval_spearman"], ascending=False), hide_index=True, use_container_width=True)
        else:
            st.info("No `metrics_micro_critic_summary.csv` found.")

        st.markdown("#### System Gates")
        if not gate_all.empty:
            gate_pivot = gate_all.pivot_table(index=["experiment", "dataset", "model"], columns=["category", "metric"], values="value", aggfunc="first")
            st.dataframe(gate_pivot, use_container_width=True)
        else:
            st.info("No `metrics_system_overview.csv` found.")

        with st.expander("Notes on data scanning"):
            st.markdown("""
            - App auto-scans `outputs/metrics/` and every `outputs/<experiment>/metrics/`.
            - Root `outputs/metrics/` is labeled `hotpotqa_large_llama3.2_baseline` (legacy run, may not be strictly comparable).
            - Directories without `metrics_macro_budget_summary.csv` are skipped.
            - `hotpotqa_large_genma3.4b` retains its original directory name including the `genma3` typo.
            """)

        # ── Paper Reference Comparison ──────────────────────────────
        st.divider()
        st.markdown("#### 📄 Comparison with Original Paper Results")
        st.markdown("""
        The charts below show how our trained system compares to published baselines
        from the original HotpotQA paper reference. **Note:** the paper reports official
        HotpotQA EM/F1, while this project uses a local HotpotQA-large split, local LLM
        settings, and `f1_proxy` + utility/cost metrics — so direct comparison is approximate.
        """)

        _viz = ROOT / "outputs" / "report_visualizations"
        _paper_imgs = [
            ("09_hotpotqa_paper_reference.png", "HotpotQA Paper Reference — EM/F1 Comparison"),
            ("10_hotpotqa_paper_reference_table.png", "Paper Reference — Detailed Results Table"),
            ("11_budget_summary_table.png", "Budget Summary — Our System"),
            ("12_system_gate_table.png", "System Gate Status — Our System"),
        ]
        for fname, caption in _paper_imgs:
            p = _viz / fname
            if p.exists():
                st.image(str(p), caption=caption, use_container_width=True)

        st.markdown("#### 📊 Detailed Pipeline Visualizations (Baseline Run)")
        _detail_imgs = [
            ("01_budget_quality.png", "Quality Metrics by Budget Mode"),
            ("02_budget_costs.png", "Cost Metrics by Budget Mode"),
            ("03_macro_quality_cost_tradeoff.png", "Quality vs Cost Tradeoff"),
            ("04_workflow_distribution.png", "Workflow Distribution per Budget"),
            ("05_micro_critic_summary.png", "Micro Critic Performance Summary"),
            ("06_critic_deployment_tradeoff.png", "Critic Deployment Tradeoff"),
            ("07_selective_critic_gate.png", "Selective Critic Gate Analysis"),
            ("08_high_bandit_cv.png", "High-Budget Bandit Cross-Validation"),
        ]
        cols_per_row = 2
        for i in range(0, len(_detail_imgs), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(_detail_imgs):
                    fname, caption = _detail_imgs[idx]
                    p = _viz / fname
                    if p.exists():
                        with col:
                            st.image(str(p), caption=caption, use_container_width=True)

# ━━━━ TAB 3: Macro Layer ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[3]:
    st.subheader("Macro Layer — Workflow Routing & Budget Policies")
    budget_df = safe_csv("metrics_macro_budget_summary.csv")
    if budget_df is not None:
        # Quality bar chart
        fig = go.Figure()
        for metric, color in [("avg_utility","#818cf8"),("avg_em","#f59e0b"),("avg_f1_proxy","#34d399")]:
            fig.add_trace(go.Bar(
                x=budget_df["budget_mode"].str.upper(), y=budget_df[metric],
                name=metric.replace("avg_","").upper(), marker_color=color,
                text=budget_df[metric].round(4), textposition="outside",
            ))
        fig.update_layout(
            title="Quality Metrics by Budget", barmode="group",
            template="plotly_dark", height=420,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch")

        # Cost chart
        c1, c2 = st.columns(2)
        with c1:
            fig2 = px.bar(budget_df, x=budget_df["budget_mode"].str.upper(), y="avg_tokens",
                          color=budget_df["budget_mode"].str.upper(),
                          color_discrete_sequence=["#6366f1","#f59e0b","#10b981"],
                          title="Avg Tokens per Budget", template="plotly_dark")
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig2, width="stretch")
        with c2:
            fig3 = px.bar(budget_df, x=budget_df["budget_mode"].str.upper(), y="avg_latency_ms",
                          color=budget_df["budget_mode"].str.upper(),
                          color_discrete_sequence=["#6366f1","#f59e0b","#10b981"],
                          title="Avg Latency (ms) per Budget", template="plotly_dark")
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig3, width="stretch")

        # Workflow distribution
        st.markdown("#### Workflow Distribution")
        for _, row in budget_df.iterrows():
            try:
                dist = eval(row["workflow_distribution"]) if isinstance(row["workflow_distribution"], str) else {}
                if dist:
                    st.markdown(f"**{row['budget_mode'].upper()}**: {dict(dist)}")
            except:
                pass

    # Bandit CV configs
    st.divider()
    st.markdown("#### Bandit Cross-Validation (5-fold, 101 configs)")
    cv_df = safe_csv("cv_ensemble_bandit_configs.csv")
    if cv_df is not None:
        fig4 = px.scatter(cv_df, x="cv_avg_regret", y="cv_avg_best_rate",
                          color="type", hover_name="name",
                          size=cv_df["cv_avg_best_rate"]*10,
                          color_discrete_sequence=["#818cf8","#34d399","#f59e0b"],
                          title="Regret vs Best-Rate (lower-left = better regret, higher = better best_rate)",
                          template="plotly_dark")
        fig4.add_hline(y=0.70, line_dash="dash", line_color="#f87171", annotation_text="min best_rate=0.70")
        fig4.add_vline(x=0.04, line_dash="dash", line_color="#f87171", annotation_text="max regret=0.04")
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=500)
        st.plotly_chart(fig4, width="stretch")

        with st.expander("Top 10 Bandit Configs"):
            st.dataframe(cv_df.head(10), hide_index=True)

# ━━━━ TAB 2: Micro Layer ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[4]:
    st.subheader("Micro Layer — Process Critic")
    critic_df = safe_csv("metrics_micro_critic_summary.csv")
    if critic_df is not None:
        c1, c2 = st.columns(2)
        for i, (_, row) in enumerate(critic_df.iterrows()):
            col = c1 if i == 0 else c2
            mod = row["module"]
            with col:
                st.markdown(f"""<div class="metric-card">
                    <h3>{mod} Critic</h3>
                    <div class="value">ρ = {row['eval_spearman']:.4f}</div>
                    <div class="sub">Pearson: {row['eval_pearson']:.4f} | MAE: {row['eval_mae']:.4f} | RMSE: {row['eval_rmse']:.4f}</div>
                    <div class="sub">Train: {int(row['train_examples'])} | Eval: {int(row['eval_examples'])}</div>
                </div>""", unsafe_allow_html=True)

    # Critic deployment
    deploy_df = safe_csv("metrics_micro_critic_deployment.csv")
    if deploy_df is not None:
        st.markdown("#### Method Comparison (Base vs Critic)")
        base_rows = deploy_df[deploy_df["is_critic"] == "no"].copy()
        critic_rows = deploy_df[deploy_df["is_critic"] == "yes"].copy()
        if len(base_rows) > 0 and len(critic_rows) > 0:
            fig5 = go.Figure()
            fig5.add_trace(go.Bar(x=base_rows["method"], y=base_rows["avg_utility"],
                                  name="Base", marker_color="#818cf8"))
            fig5.add_trace(go.Bar(x=critic_rows["method"], y=critic_rows["avg_utility"],
                                  name="Critic", marker_color="#34d399"))
            fig5.update_layout(title="Utility: Base vs Critic", barmode="group",
                              template="plotly_dark", height=400,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig5, use_container_width=True)

    # Selective critic verification
    sel_df = safe_csv("metrics_selective_critic_verification.csv")
    if sel_df is not None:
        st.markdown("#### Selective Critic Gate Verification")
        c1, c2 = st.columns(2)
        with c1:
            fig6 = go.Figure()
            fig6.add_trace(go.Scatter(x=sel_df["gate_threshold"], y=sel_df["token_multiplier"],
                                      mode="lines+markers", name="Token Multiplier",
                                      line=dict(color="#f59e0b", width=3)))
            fig6.add_hline(y=1.25, line_dash="dash", line_color="#f87171", annotation_text="Max 1.25x")
            fig6.update_layout(title="Token Multiplier vs Gate", template="plotly_dark",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig6, use_container_width=True)
        with c2:
            fig7 = go.Figure()
            fig7.add_trace(go.Scatter(x=sel_df["gate_threshold"], y=sel_df["utility_vs_base"],
                                      mode="lines+markers", name="Utility Gap",
                                      line=dict(color="#818cf8", width=3)))
            fig7.add_hline(y=-0.001, line_dash="dash", line_color="#f87171", annotation_text="Min -0.001")
            fig7.update_layout(title="Utility Gap vs Gate", template="plotly_dark",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig7, use_container_width=True)

        with st.expander("Full Gate Table"):
            st.dataframe(sel_df, hide_index=True)

# ━━━━ TAB 3: System Gates ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[5]:
    st.subheader("System Gates & RL Status")
    regate = safe_json("regate_report.json")
    if regate:
        ready = regate.get("ready_for_online_rl", False)
        st.markdown(f"""<div class="metric-card">
            <h3>Online RL Readiness</h3>
            <div class="value {'status-pass' if ready else 'status-fail'}">{'✅ ALL GATES PASS' if ready else '❌ BLOCKED'}</div>
            <div class="sub">{regate.get('deployment_recommendation','')}</div>
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        bc = regate.get("bandit_check", {})
        cc = regate.get("critic_check", {})
        with c1:
            st.markdown("#### 🎰 Bandit Gate")
            items = [
                ("Avg Regret", bc.get("avg_regret"), f"≤ {bc.get('max_regret_threshold')}"),
                ("Best Rate", bc.get("exact_best_rate"), f"≥ {bc.get('min_best_rate_threshold')}"),
            ]
            for label, val, threshold in items:
                passed = bc.get("meets_online_threshold", False)
                st.markdown(f"{'✅' if passed else '❌'} **{label}**: `{val}` (threshold: {threshold})")
            cfg = bc.get("best_config", {})
            if cfg:
                st.markdown(f"**Best Config**: `{cfg.get('name','?')}`")
                folds = cfg.get("per_fold_regret", [])
                if folds:
                    fig_f = px.bar(x=[f"Fold {i+1}" for i in range(len(folds))], y=folds,
                                   title="Per-Fold Regret", template="plotly_dark",
                                   color_discrete_sequence=["#818cf8"])
                    fig_f.add_hline(y=0.04, line_dash="dash", line_color="#f87171")
                    fig_f.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                       showlegend=False, height=300)
                    st.plotly_chart(fig_f, use_container_width=True)

        with c2:
            st.markdown("#### 🔬 Critic Gate")
            cp = cc.get("meets_online_threshold", False)
            st.markdown(f"{'✅' if cp else '❌'} **Token Multiplier**: `{cc.get('token_multiplier')}x` (≤ {cc.get('max_token_multiplier_threshold')})")
            st.markdown(f"{'✅' if cp else '❌'} **Utility Gap**: `{cc.get('utility_gap')}` (≥ -{cc.get('max_utility_loss_threshold')})")
            st.markdown(f"**Strategy**: `{cc.get('strategy','?')}`")
            st.markdown(f"**Verified on holdout**: {'✅' if cc.get('verified_on_holdout') else '❌'}")

            # Gauge chart
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=cc.get("token_multiplier", 1.0),
                title={"text": "Token Multiplier"},
                gauge={"axis": {"range": [0.8, 2.5]},
                       "bar": {"color": "#34d399" if cc.get("token_multiplier",1) <= 1.25 else "#f87171"},
                       "threshold": {"line": {"color": "#f87171", "width": 3}, "value": 1.25}}
            ))
            fig_g.update_layout(template="plotly_dark", height=300,
                               paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_g, use_container_width=True)
    else:
        st.info("No regate report found. Run `python scripts/regate_online_rl.py` first.")

    # Promotion details
    promo = safe_json("promotion_report.json")
    if promo:
        st.divider()
        st.markdown("#### Promotion Report")
        st.json(promo)

# ━━━━ TAB 4: Visualizations ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[6]:
    st.subheader("Report Visualizations")
    if VIZ.exists():
        pngs = sorted(VIZ.glob("*.png"))
        if pngs:
            cols_per_row = 2
            for i in range(0, len(pngs), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(pngs):
                        with col:
                            st.image(str(pngs[idx]), caption=pngs[idx].stem.replace("_"," ").title(), use_container_width=True)
        else:
            st.info("No visualization PNGs found.")
    else:
        st.info("Visualization directory not found.")

# ━━━━ TAB 5: Live Demo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[7]:
    st.subheader("🏃 Interactive Q&A Demo")
    st.markdown("Nhập câu hỏi → hệ thống **tự động chọn workflow** tối ưu dựa trên trained bandit router + budget.")

    # Row 1: LLM settings
    lc1, lc2, lc3 = st.columns([1, 1, 2])
    with lc1:
        llm_provider = st.selectbox("LLM Provider", ["ollama", "gemini", "dummy"], index=0,
                                     help="ollama = local LLM | gemini = Google API | dummy = placeholder")
    with lc2:
        default_models = {"ollama": "llama3.2", "gemini": "gemini-2.0-flash", "dummy": "dummy-free"}
        llm_model = st.text_input("Model", value=default_models.get(llm_provider, "llama3.2"))
    with lc3:
        dataset = st.selectbox("Dataset (corpus)", ["sample", "hotpotqa_large", "hotpotqa"],
                               help="sample = 3 QA pairs | hotpotqa_large = 600 QA | hotpotqa = 100 QA")

    # Row 2: Question + budget (no manual workflow selection by default)
    c1, c2 = st.columns([4, 1])
    with c1:
        question = st.text_input("Question", value="Which novel by the author of Pride and Prejudice was published posthumously?")
    with c2:
        budget = st.selectbox("Budget", ["low", "medium", "high"], index=2)

    gold = st.text_input("Gold Answer (for scoring)", value="Persuasion")

    # Manual override option
    with st.expander("⚙️ Chế độ nâng cao — ghi đè workflow thủ công"):
        manual_override = st.checkbox("Ghi đè workflow thủ công (bỏ qua auto router)", value=False)
        manual_workflow = st.selectbox("Chọn workflow", ["W1","W2","W3","W4","W5","W6"], index=2,
                                       disabled=not manual_override)

    # Provider status indicator
    if llm_provider == "ollama":
        st.info("🟢 **Ollama** local LLM. Đảm bảo `ollama serve` đang chạy.")
    elif llm_provider == "gemini":
        st.info("🔑 **Gemini API**. Cần `GEMINI_API_KEY` trong `.env`.")
    else:
        st.warning("🧪 **Dummy** provider — kết quả placeholder.")

    if st.button("🚀 Run Pipeline", type="primary"):
        import time as _time
        t_start = _time.time()
        with st.spinner(f"Đang xử lý câu hỏi với {llm_provider}/{llm_model}..."):
            try:
                from map_rag_gym.core.pipeline import MAPRAGGym
                corpus_path = str(ROOT / "data" / dataset / "corpus.json")

                # ── Auto Router: load bandit model & select workflow ──
                router_info = {}
                if manual_override:
                    selected_workflow = manual_workflow
                    planner_reason = f"manual_override:{manual_workflow}"
                    router_info = {"mode": "manual", "workflow": manual_workflow}
                else:
                    # Try loading the trained bandit router for this budget
                    bundle_path = OUT / "final_budget_policy_bundle_rl_ready.json"
                    if not bundle_path.exists():
                        bundle_path = OUT / "final_budget_policy_bundle.json"

                    selected_workflow = "W3"  # fallback
                    planner_reason = "fallback:W3"

                    if bundle_path.exists():
                        bundle = load_json(str(bundle_path))
                        bp = bundle.get("budget_policies", {}).get(budget, {})
                        method = bp.get("recommended_method", "fixed_W3")
                        rs = bp.get("router_settings", {})
                        bandit_model_path = rs.get("bandit_router_model")

                        if bandit_model_path and (ROOT / bandit_model_path).exists():
                            from map_rag_gym.router.bandit import BanditRouter
                            from map_rag_gym.retrieval.bm25 import LocalBM25Retriever

                            bandit = BanditRouter()
                            bandit.load(str(ROOT / bandit_model_path))
                            bandit.attach_probe_retriever(LocalBM25Retriever(corpus_path))

                            if "gated" in method:
                                wf, conf, scores, gate_meta = bandit.predict_with_gate(
                                    question, budget_mode=budget,
                                    baseline_workflow=rs.get("bandit_gate_baseline_workflow", "W3"),
                                    minimum_advantage=rs.get("bandit_gate_min_advantage", 0.0),
                                    minimum_confidence=rs.get("bandit_gate_min_confidence", 0.0),
                                    allowed_switch_workflows=rs.get("bandit_gate_allowed_workflows", []),
                                )
                                selected_workflow = wf
                                planner_reason = f"gated_bandit:{conf:.4f}"
                                router_info = {
                                    "mode": "gated_bandit_router",
                                    "workflow": wf, "confidence": round(conf, 4),
                                    "scores": {k: round(v, 4) for k, v in scores.items()},
                                    "gate": gate_meta,
                                }
                            else:
                                wf, conf, scores = bandit.predict_with_scores(question, budget_mode=budget)
                                selected_workflow = wf
                                planner_reason = f"bandit:{conf:.4f}"
                                router_info = {
                                    "mode": "bandit_router",
                                    "workflow": wf, "confidence": round(conf, 4),
                                    "scores": {k: round(v, 4) for k, v in scores.items()},
                                }
                        else:
                            router_info = {"mode": "fallback", "reason": "No bandit model found", "workflow": "W3"}
                    else:
                        router_info = {"mode": "fallback", "reason": "No policy bundle found", "workflow": "W3"}

                # ── Show router decision ──
                if not manual_override:
                    mode_label = router_info.get("mode", "?")
                    scores = router_info.get("scores", {})
                    if scores:
                        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                        score_str = " → ".join(f"**{wf}** ({s:.4f})" for wf, s in ranked)
                        st.markdown(f"🔀 **Router ({mode_label})** đã chọn **{selected_workflow}** "
                                    f"(confidence: {router_info.get('confidence', '?')})")
                        st.caption(f"Workflow scores: {score_str}")
                        if router_info.get("gate", {}).get("gate_applied"):
                            st.caption(f"⚠️ Gate applied — baseline fallback triggered")
                    else:
                        st.markdown(f"🔀 **Router ({mode_label})**: sử dụng **{selected_workflow}**")

                # ── Run pipeline ──
                pipe = MAPRAGGym(corpus_path, llm_provider=llm_provider, llm_model=llm_model)
                result = pipe.run(question, gold, selected_workflow,
                                  planner_reason=planner_reason, budget_mode=budget)
                elapsed = _time.time() - t_start

                st.success(f"**Answer:** {result.final_answer}")
                st.caption(f"⏱️ {elapsed:.1f}s | {llm_provider}/{llm_model} | {selected_workflow} | budget={budget}")

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Utility", f"{result.final_scores.get('utility_total', 0):.4f}")
                mc2.metric("EM", f"{result.final_scores.get('exact_match', 0):.4f}")
                mc3.metric("F1", f"{result.final_scores.get('f1', 0):.4f}")
                mc4.metric("Tokens", f"{result.total_cost.get('tokens', 0):.0f}")

                st.markdown("#### Pipeline Steps")
                for step in result.steps:
                    with st.expander(f"Step {step.step_id}: **{step.module}** — {', '.join(f'{k}={v:.3f}' for k,v in step.scores.items() if v is not None)}"):
                        sc1, sc2 = st.columns(2)
                        with sc1:
                            st.markdown("**Input:**")
                            st.json(step.input_data)
                        with sc2:
                            st.markdown("**Output:**")
                            st.json(step.output_data)
                        st.markdown(f"**Cost:** tokens={step.cost.get('tokens',0):.0f} | retrieval={step.cost.get('retrieval_calls',0)} | latency={step.cost.get('latency_ms',0):.0f}ms")
                        if step.notes:
                            st.caption(f"Notes: {step.notes}")

                # Show full router decision details
                if router_info:
                    with st.expander("🔍 Chi tiết Router Decision"):
                        st.json(router_info)

            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc(), language="text")
                if llm_provider == "ollama":
                    st.info("💡 Kiểm tra: `ollama serve` đang chạy? Model đã pull? (`ollama pull llama3.2`)")
                elif llm_provider == "gemini":
                    st.info("💡 Kiểm tra: `GEMINI_API_KEY` đã set trong `.env`?")
                else:
                    st.info("💡 Lỗi không mong đợi. Kiểm tra lại source code.")

# ── footer ─────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#64748b; font-size:13px; padding:10px 0">
    MAP-RAG-Gym • Adaptive Multi-Workflow RAG with Process Evaluation<br/>
    Combining MAO-ARAG (macro) + RAG-Gym (micro) for budget-aware RAG
</div>
""", unsafe_allow_html=True)
