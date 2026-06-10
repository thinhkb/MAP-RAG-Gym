"""Quick script to generate report figures from streamlit_app helpers."""
import sys, os, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import textwrap
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
OUT  = ROOT / "outputs"
MET  = OUT / "metrics"

def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def load_csv(p):
    return pd.read_csv(p)

COLORS = {
    "low": "#6366f1", "medium": "#f59e0b", "high": "#10b981",
}
REPORT_FIG_DIR = ROOT / "figures"
REPORT_FIG_DIR.mkdir(exist_ok=True)

# Copy helper functions
def _wrap(text, width=58):
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))

def _load_qa_examples(dataset_dir, n=2):
    p = ROOT / "data" / dataset_dir / "splits" / "test.json"
    if not p.exists(): return []
    data = load_json(str(p))
    examples = []
    for item in data[:n]:
        examples.append({
            "id": item.get("id", ""),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "type": item.get("metadata", {}).get("type", "multi-hop"),
        })
    return examples

def _final_policy_rows():
    configs = [
        ("HotpotQA", "Llama 3.2 (3B)", OUT / "metrics" / "metrics_macro_budget_summary.csv", None),
        ("HotpotQA", "Gemma 3 (12B)", None, OUT / "hotpotqa_large_gemma3" / "final_project_report.json"),
        ("HotpotQA", "Gemma 3 (4B)", None, OUT / "hotpotqa_large_genma3.4b" / "final_project_report.json"),
        ("2WikiMultihopQA", "Gemma 3 (12B)", None, OUT / "2wikimultihopqa_gemma3" / "final_project_report.json"),
        ("2WikiMultihopQA", "Llama 3.2 (3B)", None, OUT / "2wikimultihopqa_llama" / "final_project_report.json"),
    ]
    rows = []
    for dataset, model, csv_path, json_path in configs:
        if csv_path and csv_path.exists():
            df = pd.read_csv(csv_path)
            for _, r in df.iterrows():
                rows.append({"dataset": dataset, "model": model, "budget": r["budget_mode"],
                             "utility": float(r["avg_utility"]), "em": float(r["avg_em"]),
                             "f1": float(r["avg_f1_proxy"]), "tokens": float(r["avg_tokens"])})
        elif json_path and json_path.exists():
            report = load_json(str(json_path))
            for r in report.get("final_policy_table", []):
                rows.append({"dataset": dataset, "model": model, "budget": r["budget_mode"],
                             "utility": float(r["avg_utility"]), "em": float(r["avg_em"]),
                             "f1": float(r["avg_f1_proxy"]), "tokens": float(r["avg_tokens"])})
    return pd.DataFrame(rows)

def _load_bandit_cv():
    csv_path = OUT / "metrics" / "cv_ensemble_bandit_configs.csv"
    if csv_path.exists(): return pd.read_csv(csv_path)
    return pd.DataFrame()

def _load_eval_examples(n=3):
    p = OUT / "final_budget_policy_test_eval.json"
    if not p.exists(): return []
    data = load_json(str(p))
    budget_results = data.get("budget_results", {})
    examples = []
    for budget in ["low", "medium", "high"]:
        br = budget_results.get(budget, {})
        per_q = br.get("per_question", [])
        if not per_q: continue
        chosen = None
        for q in per_q:
            if float(q.get("utility_total", 0)) > 0:
                chosen = q
                break
        if chosen is None: chosen = per_q[0]
        examples.append({
            "budget": budget,
            "workflow": chosen.get("workflow_id", chosen.get("selected_method", "?")),
            "question": chosen.get("question", ""),
            "gold": chosen.get("gold_answer", ""),
            "prediction": f"EM={chosen.get('em', 0):.0f}, F1={chosen.get('f1_proxy', 0):.4f}",
            "utility": f"{chosen.get('utility_total', 0):.4f}",
        })
        if len(examples) >= n: break
    return examples


if __name__ == "__main__":
    saved = []

    # Figure 1
    hotpot = _load_qa_examples("hotpotqa_large", n=2)
    wiki = _load_qa_examples("2wikimultihopqa", n=2)
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.axis("off")
    ax.set_title("Figure 1. Representative Examples from HotpotQA and 2WikiMultihopQA", fontsize=16, weight="bold", pad=16)
    y = 0.92
    for title, examples, color in [("HotpotQA", hotpot, "#EEF2FF"), ("2WikiMultihopQA", wiki, "#ECFDF5")]:
        ax.text(0.02, y, title, fontsize=13, weight="bold", color="#111827", transform=ax.transAxes)
        y -= 0.055
        for ex in examples:
            box = patches.FancyBboxPatch((0.02, y-0.145), 0.96, 0.13, boxstyle="round,pad=0.012", facecolor=color, edgecolor="#CBD5E1", transform=ax.transAxes)
            ax.add_patch(box)
            txt = f"Q: {_wrap(ex['question'], 115)}\nA: {ex['answer']}   |   Type: {ex['type']}"
            ax.text(0.04, y-0.03, txt, fontsize=9.5, va="top", color="#111827", transform=ax.transAxes)
            y -= 0.165
        y -= 0.025
    p = REPORT_FIG_DIR / "figure_1_dataset_examples.png"
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig); saved.append(p)

    # Figure 3
    df = _final_policy_rows()
    if not df.empty:
        order = ["low", "medium", "high"]
        pivot = df.pivot_table(index=["dataset", "model"], columns="budget", values="utility", aggfunc="first").reindex(columns=order)
        labels = [f"{idx[0]}\n{idx[1]}" for idx in pivot.index]
        x = np.arange(len(labels))
        width = 0.24
        fig, ax = plt.subplots(figsize=(13, 6.8))
        colors = [COLORS["low"], COLORS["medium"], COLORS["high"]]
        for i, b in enumerate(order):
            bars = ax.bar(x + (i-1)*width, pivot[b], width, label=b.capitalize(), color=colors[i], edgecolor="white", linewidth=0.7)
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
        ax.set_title("Figure 3. Utility across Datasets, Local Models, and Budget Modes", fontsize=15, weight="bold")
        ax.set_ylabel("Average utility")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylim(0, max(0.5, float(df["utility"].max()) + 0.08))
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(title="Budget mode", ncols=3, loc="upper right")
        fig.tight_layout()
        p = REPORT_FIG_DIR / "figure_3_budget_utility.png"
        fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig); saved.append(p)

    # Figure 4
    cv = _load_bandit_cv()
    if not cv.empty and {"cv_avg_regret", "cv_avg_best_rate"}.issubset(cv.columns):
        fig, ax = plt.subplots(figsize=(10.5, 6.5))
        types = cv.get("type", pd.Series(["config"] * len(cv))).fillna("config")
        for t, sub in cv.groupby(types):
            ax.scatter(sub["cv_avg_regret"], sub["cv_avg_best_rate"], s=55, alpha=0.75, label=str(t), edgecolor="white", linewidth=0.4)
        ax.axvline(0.04, color="#EF4444", linestyle="--", linewidth=1.6, label="Regret threshold = 0.04")
        ax.axhline(0.70, color="#10B981", linestyle="--", linewidth=1.6, label="Best-rate threshold = 0.70")
        best = cv.sort_values(["cv_avg_best_rate", "cv_avg_regret"], ascending=[False, True]).head(1)
        if not best.empty:
            bx, by = float(best.iloc[0]["cv_avg_regret"]), float(best.iloc[0]["cv_avg_best_rate"])
            ax.scatter([bx], [by], s=180, marker="*", color="#F59E0B", edgecolor="#111827", zorder=5, label="Best config")
            ax.annotate(f"best\nregret={bx:.4f}\nbest-rate={by:.4f}", (bx, by), xytext=(8, 8), textcoords="offset points", fontsize=9)
        ax.set_title("Figure 4. High-Budget Bandit Cross-Validation", fontsize=15, weight="bold")
        ax.set_xlabel("Cross-validation average regret (lower is better)")
        ax.set_ylabel("Exact best-rate (higher is better)")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        p = REPORT_FIG_DIR / "figure_4_bandit_cv.png"
        fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig); saved.append(p)

    # Figure 5
    examples = _load_eval_examples(n=3)
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.axis("off")
    ax.set_title("Figure 5. Qualitative Routed Workflow Examples", fontsize=16, weight="bold", pad=14)
    y = 0.90
    palette = {"low": "#EEF2FF", "medium": "#FFFBEB", "high": "#ECFDF5"}
    for ex in examples:
        box = patches.FancyBboxPatch((0.02, y-0.23), 0.96, 0.205, boxstyle="round,pad=0.014",
              facecolor=palette.get(ex["budget"], "#F8FAFC"), edgecolor="#CBD5E1", transform=ax.transAxes)
        ax.add_patch(box)
        header = f"Budget: {ex['budget'].upper()}   |   Workflow: {ex['workflow']}   |   Utility: {ex['utility']}"
        body = f"Q: {_wrap(ex['question'], 105)}\nGold: {ex['gold']}\nScores: {ex['prediction']}"
        ax.text(0.04, y-0.045, header, fontsize=11, weight="bold", color="#111827", transform=ax.transAxes)
        ax.text(0.04, y-0.085, body, fontsize=9.5, va="top", color="#111827", transform=ax.transAxes)
        y -= 0.265
    if not examples:
        ax.text(0.5, 0.5, "No qualitative examples found", ha="center", va="center")
    p = REPORT_FIG_DIR / "figure_5_qualitative_examples.png"
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig); saved.append(p)

    print(f"Generated {len(saved)} figures:")
    for s in saved:
        print(f"  {s}")
