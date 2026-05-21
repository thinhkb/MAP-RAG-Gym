from __future__ import annotations

"""
Create report-ready matplotlib figures and tables from the current outputs/.

The script is intentionally read-only with respect to training artifacts. It reads
CSV/JSON metrics produced by run_all.sh and writes PNG + Markdown summaries to:

  outputs/report_visualizations/

Paper reference numbers are manually transcribed from the two PDFs used by the
project. They are included only as HotpotQA reference points because the project
uses a local HotpotQA-large split, local models, and an f1_proxy/utility metric,
so the comparison is not a strict apples-to-apples benchmark.
"""

import argparse
import ast
import json
from pathlib import Path
from textwrap import wrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BUDGET_ORDER = ["low", "medium", "high"]
WORKFLOW_ORDER = ["W1", "W2", "W3", "W4", "W5", "W6"]
COLORS = {
    "low": "#4C78A8",
    "medium": "#F58518",
    "high": "#54A24B",
    "utility": "#4C78A8",
    "em": "#F58518",
    "f1": "#54A24B",
    "tokens": "#B279A2",
    "latency": "#E45756",
    "retrieval": "#72B7B2",
}


# HotpotQA numbers transcribed from the papers.
PAPER_HOTPOTQA_REFERENCE = [
    {
        "source": "MAO-ARAG paper",
        "method": "Vanilla RAG",
        "metric": "F1",
        "value": 49.54,
        "note": "HotpotQA F1, Table 1",
    },
    {
        "source": "MAO-ARAG paper",
        "method": "Search-o1",
        "metric": "F1",
        "value": 53.75,
        "note": "HotpotQA F1, Table 1",
    },
    {
        "source": "MAO-ARAG paper",
        "method": "MAO-ARAG",
        "metric": "F1",
        "value": 53.80,
        "note": "HotpotQA F1, Table 1",
    },
    {
        "source": "RAG-Gym paper",
        "method": "Re2Search++ (Llama-3.1-8B)",
        "metric": "EM",
        "value": 46.50,
        "note": "HotpotQA EM, Table 3",
    },
    {
        "source": "RAG-Gym paper",
        "method": "Re2Search++ (Llama-3.1-8B)",
        "metric": "F1",
        "value": 60.19,
        "note": "HotpotQA F1, Table 3",
    },
    {
        "source": "RAG-Gym paper",
        "method": "Re2Search++ (Qwen-2.5-7B)",
        "metric": "EM",
        "value": 44.40,
        "note": "HotpotQA EM, Table 3",
    },
    {
        "source": "RAG-Gym paper",
        "method": "Re2Search++ (Qwen-2.5-7B)",
        "metric": "F1",
        "value": 56.47,
        "note": "HotpotQA F1, Table 3",
    },
]


def _style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required metrics file: {path}")
    return pd.read_csv(path)


def _maybe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: plt.Figure, out_path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _autolabel_bars(ax: plt.Axes, fmt: str = "{:.2f}", y_pad: float = 0.01) -> None:
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * y_pad
    for container in ax.containers:
        for rect in container:
            height = rect.get_height()
            if np.isnan(height):
                continue
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                height + offset,
                fmt.format(height),
                ha="center",
                va="bottom",
                fontsize=8,
            )


def _wrap_labels(labels: list[str], width: int = 18) -> list[str]:
    return ["\n".join(wrap(str(label), width=width)) for label in labels]


def _parse_distribution(raw: object) -> dict[str, int]:
    if isinstance(raw, dict):
        return {str(k): int(v) for k, v in raw.items()}
    if pd.isna(raw):
        return {}
    try:
        parsed = ast.literal_eval(str(raw))
    except (ValueError, SyntaxError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): int(v) for k, v in parsed.items()}


def plot_budget_quality(macro_df: pd.DataFrame, out_dir: Path) -> Path:
    df = macro_df.set_index("budget_mode").loc[BUDGET_ORDER].reset_index()
    x = np.arange(len(df))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width, df["avg_utility"], width, label="Utility", color=COLORS["utility"])
    ax.bar(x, df["avg_em"], width, label="EM", color=COLORS["em"])
    ax.bar(x + width, df["avg_f1_proxy"], width, label="F1 proxy", color=COLORS["f1"])
    ax.set_title("Final policy quality by budget")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(df["budget_mode"].str.upper())
    ax.set_ylim(0, max(0.55, float(df[["avg_utility", "avg_em", "avg_f1_proxy"]].max().max()) + 0.08))
    ax.legend(loc="upper left", ncols=3)
    _autolabel_bars(ax)
    return _save(fig, out_dir / "01_budget_quality.png")


def plot_budget_costs(macro_df: pd.DataFrame, out_dir: Path) -> Path:
    df = macro_df.set_index("budget_mode").loc[BUDGET_ORDER].reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    metrics = [
        ("avg_tokens", "Avg tokens", COLORS["tokens"]),
        ("avg_retrieval_calls", "Retrieval calls", COLORS["retrieval"]),
        ("avg_latency_ms", "Latency (ms)", COLORS["latency"]),
    ]
    for ax, (col, title, color) in zip(axes, metrics):
        bars = ax.bar(df["budget_mode"].str.upper(), df[col], color=color)
        ax.set_title(title)
        ax.set_xlabel("Budget")
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    fig.suptitle("Cost profile by budget", y=1.03, fontsize=13)
    return _save(fig, out_dir / "02_budget_costs.png")


def plot_macro_tradeoff(macro_df: pd.DataFrame, out_dir: Path) -> Path:
    df = macro_df.set_index("budget_mode").loc[BUDGET_ORDER].reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    sizes = 160 + df["avg_retrieval_calls"].astype(float) * 360
    ax.scatter(
        df["avg_tokens"],
        df["avg_utility"],
        s=sizes,
        c=[COLORS[b] for b in df["budget_mode"]],
        edgecolors="#222222",
        linewidths=0.8,
        alpha=0.9,
    )
    for _, row in df.iterrows():
        label = f"{row['budget_mode'].upper()}\n{row['recommended_method']}"
        ax.annotate(label, (row["avg_tokens"], row["avg_utility"]), xytext=(7, 7), textcoords="offset points")
    ax.set_title("Quality-cost tradeoff of final budget policies")
    ax.set_xlabel("Avg tokens per question")
    ax.set_ylabel("Avg utility")
    ax.text(
        0.01,
        0.02,
        "Bubble size = retrieval calls",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    return _save(fig, out_dir / "03_macro_quality_cost_tradeoff.png")


def plot_workflow_distribution(macro_df: pd.DataFrame, out_dir: Path) -> Path:
    rows = []
    for _, row in macro_df.iterrows():
        dist = _parse_distribution(row.get("workflow_distribution"))
        total = max(1, sum(dist.values()))
        for wf in WORKFLOW_ORDER:
            rows.append(
                {
                    "budget_mode": row["budget_mode"],
                    "workflow": wf,
                    "count": dist.get(wf, 0),
                    "share": dist.get(wf, 0) / total,
                }
            )
    dist_df = pd.DataFrame(rows)
    pivot = dist_df.pivot(index="budget_mode", columns="workflow", values="share").reindex(BUDGET_ORDER).fillna(0.0)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bottom = np.zeros(len(pivot))
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    for color, wf in zip(palette, WORKFLOW_ORDER):
        values = pivot[wf].values
        ax.bar(pivot.index.str.upper(), values, bottom=bottom, label=wf, color=color)
        bottom += values
    ax.set_title("Workflow distribution by budget")
    ax.set_ylabel("Share of questions")
    ax.set_ylim(0, 1)
    ax.legend(ncols=6, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.yaxis.set_major_formatter(lambda x, _pos: f"{x:.0%}")
    return _save(fig, out_dir / "04_workflow_distribution.png")


def plot_critic_summary(critic_df: pd.DataFrame, out_dir: Path) -> Path:
    df = critic_df.copy()
    x = np.arange(len(df))
    width = 0.22

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width, df["eval_pearson"], width, label="Pearson", color="#4C78A8")
    ax.bar(x, df["eval_spearman"], width, label="Spearman", color="#54A24B")
    ax.bar(x + width, df["eval_mae"], width, label="MAE", color="#F58518")
    ax.set_title("Micro critic validation metrics")
    ax.set_ylabel("Metric value")
    ax.set_xticks(x)
    ax.set_xticklabels(df["module"])
    ax.legend(ncols=3, loc="upper center")
    _autolabel_bars(ax)
    return _save(fig, out_dir / "05_micro_critic_summary.png")


def plot_critic_deployment(deploy_df: pd.DataFrame, out_dir: Path) -> Path | None:
    if deploy_df.empty:
        return None
    df = deploy_df[deploy_df["method"].astype(str).str.upper() != "DELTA (CRITIC - BASE)"].copy()
    df = df[df["method"].astype(str).str.upper() != "TOKEN MULTIPLIER"]
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    color_map = {"yes": "#E45756", "no": "#4C78A8"}
    for critic_flag, group in df.groupby("is_critic"):
        ax.scatter(
            group["avg_tokens"],
            group["avg_utility"],
            s=90,
            label=f"critic={critic_flag}",
            color=color_map.get(str(critic_flag), "#999999"),
            alpha=0.85,
            edgecolors="#222222",
            linewidths=0.5,
        )
    for _, row in df.iterrows():
        name = str(row["method"]).replace("_critic", "")
        if name in {"fixed_W1", "fixed_W2", "fixed_W3", "fixed_W6", "learned_router"}:
            ax.annotate(name, (row["avg_tokens"], row["avg_utility"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_title("Critic deployment tradeoff")
    ax.set_xlabel("Avg tokens")
    ax.set_ylabel("Avg utility")
    ax.legend(loc="best")
    return _save(fig, out_dir / "06_critic_deployment_tradeoff.png")


def plot_selective_critic_gate(gate_df: pd.DataFrame, out_dir: Path) -> Path | None:
    if gate_df.empty:
        return None
    df = gate_df.copy()
    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
    ax1.plot(df["gate_threshold"], df["token_multiplier"], marker="o", color="#E45756", label="Token multiplier")
    ax1.axhline(1.25, color="#E45756", linestyle="--", linewidth=1, alpha=0.7, label="Token gate")
    ax1.set_xlabel("Critic gate threshold")
    ax1.set_ylabel("Token multiplier", color="#E45756")
    ax1.tick_params(axis="y", labelcolor="#E45756")

    ax2 = ax1.twinx()
    ax2.plot(df["gate_threshold"], df["utility_vs_base"], marker="s", color="#4C78A8", label="Utility vs base")
    ax2.axhline(-0.001, color="#4C78A8", linestyle="--", linewidth=1, alpha=0.7, label="Utility gate")
    ax2.set_ylabel("Utility vs base", color="#4C78A8")
    ax2.tick_params(axis="y", labelcolor="#4C78A8")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="lower center", ncols=2, bbox_to_anchor=(0.5, -0.28))
    ax1.set_title("Selective critic gate check")
    return _save(fig, out_dir / "07_selective_critic_gate.png")


def plot_bandit_cv(config_df: pd.DataFrame, out_dir: Path) -> Path | None:
    if config_df.empty:
        return None
    df = config_df.copy().head(40)
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    type_colors = {"gbt": "#4C78A8", "ridge": "#F58518", "ensemble": "#54A24B"}
    for model_type, group in df.groupby("type"):
        ax.scatter(
            group["cv_avg_regret"],
            group["cv_avg_best_rate"],
            s=85,
            label=model_type,
            color=type_colors.get(model_type, "#999999"),
            alpha=0.82,
            edgecolors="#222222",
            linewidths=0.5,
        )
    ax.axvline(0.04, color="#E45756", linestyle="--", linewidth=1, label="regret gate")
    ax.axhline(0.70, color="#72B7B2", linestyle="--", linewidth=1, label="best-rate gate")
    for _, row in df.head(5).iterrows():
        ax.annotate(str(row["name"]), (row["cv_avg_regret"], row["cv_avg_best_rate"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_title("High-budget bandit CV model selection")
    ax.set_xlabel("CV avg regret (lower is better)")
    ax.set_ylabel("CV exact-best rate (higher is better)")
    ax.legend(loc="lower left", ncols=2)
    return _save(fig, out_dir / "08_high_bandit_cv.png")


def plot_paper_reference(macro_df: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    high = macro_df.set_index("budget_mode").loc["high"]
    project_f1 = float(high["avg_f1_proxy"]) * 100.0
    project_em = float(high["avg_em"]) * 100.0

    f1_rows = [
        {"source": "This project", "method": "MAP-RAG-Gym high budget", "metric": "F1 proxy", "value": project_f1},
        *[row for row in PAPER_HOTPOTQA_REFERENCE if row["metric"] == "F1"],
    ]
    em_rows = [
        {"source": "This project", "method": "MAP-RAG-Gym high budget", "metric": "EM", "value": project_em},
        *[row for row in PAPER_HOTPOTQA_REFERENCE if row["metric"] == "EM"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, rows, title in [
        (axes[0], f1_rows, "HotpotQA F1 reference"),
        (axes[1], em_rows, "HotpotQA EM reference"),
    ]:
        labels = [f"{r['source']}\n{r['method']}" for r in rows]
        values = [r["value"] for r in rows]
        colors = ["#222222"] + ["#A0A0A0"] * (len(rows) - 1)
        bars = ax.bar(range(len(rows)), values, color=colors)
        ax.set_title(title)
        ax.set_ylabel("Score (%)")
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(_wrap_labels(labels, width=17), rotation=0)
        ax.set_ylim(0, max(values) + 12)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    fig.suptitle("Paper comparison is a reference only: metric, split, model, and corpus differ", y=1.03, fontsize=12)
    chart_path = _save(fig, out_dir / "09_hotpotqa_paper_reference.png")

    table_rows = [
        {
            "Source": "This project",
            "Method": "MAP-RAG-Gym high budget",
            "Dataset": "local HotpotQA-large test",
            "Metric": "F1 proxy",
            "Value": f"{project_f1:.2f}",
            "Note": "Not paper F1",
        },
        {
            "Source": "This project",
            "Method": "MAP-RAG-Gym high budget",
            "Dataset": "local HotpotQA-large test",
            "Metric": "EM",
            "Value": f"{project_em:.2f}",
            "Note": "Exact match",
        },
    ]
    for row in PAPER_HOTPOTQA_REFERENCE:
        table_rows.append(
            {
                "Source": row["source"],
                "Method": row["method"],
                "Dataset": "HotpotQA",
                "Metric": row["metric"],
                "Value": f"{row['value']:.2f}",
                "Note": row["note"],
            }
        )
    table_df = pd.DataFrame(table_rows)
    table_path = make_table_png(table_df, out_dir / "10_hotpotqa_paper_reference_table.png", title="HotpotQA reference metrics")
    return chart_path, table_path


def make_table_png(df: pd.DataFrame, out_path: Path, title: str, max_rows: int = 16) -> Path:
    shown = df.head(max_rows).copy()
    for col in shown.columns:
        shown[col] = shown[col].map(lambda x: "\n".join(wrap(str(x), width=24)))
    fig_height = 1.2 + 0.44 * (len(shown) + 1)
    fig, ax = plt.subplots(figsize=(12.5, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=12)
    table = ax.table(
        cellText=shown.values,
        colLabels=shown.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1, 1.45)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#333333")
        elif row % 2 == 0:
            cell.set_facecolor("#F7F7F7")
    return _save(fig, out_path)


def make_budget_table(macro_df: pd.DataFrame, out_dir: Path) -> Path:
    cols = [
        "budget_mode",
        "recommended_method",
        "avg_utility",
        "avg_em",
        "avg_f1_proxy",
        "avg_tokens",
        "avg_retrieval_calls",
        "avg_latency_ms",
        "workflow_distribution",
    ]
    df = macro_df.set_index("budget_mode").loc[BUDGET_ORDER].reset_index()[cols].copy()
    rename = {
        "budget_mode": "Budget",
        "recommended_method": "Policy",
        "avg_utility": "Utility",
        "avg_em": "EM",
        "avg_f1_proxy": "F1 proxy",
        "avg_tokens": "Tokens",
        "avg_retrieval_calls": "Retrieval calls",
        "avg_latency_ms": "Latency ms",
        "workflow_distribution": "Workflow counts",
    }
    df = df.rename(columns=rename)
    return make_table_png(df, out_dir / "11_budget_summary_table.png", "Final policy summary by budget")


def make_system_gate_table(system_df: pd.DataFrame, out_dir: Path) -> Path:
    df = system_df.copy()
    df.columns = ["Category", "Metric", "Value"]
    return make_table_png(df, out_dir / "12_system_gate_table.png", "System gate summary")


def write_markdown_summary(
    out_dir: Path,
    macro_df: pd.DataFrame,
    critic_df: pd.DataFrame,
    system_df: pd.DataFrame,
    generated: list[Path],
) -> Path:
    high = macro_df.set_index("budget_mode").loc["high"]
    best_budget = macro_df.sort_values("avg_utility", ascending=False).iloc[0]
    qr = critic_df[critic_df["module"] == "QR"]
    ag = critic_df[critic_df["module"] == "AG"]
    qr_s = float(qr["eval_spearman"].iloc[0]) if not qr.empty else np.nan
    ag_s = float(ag["eval_spearman"].iloc[0]) if not ag.empty else np.nan
    overview = {
        (row["category"], row["metric"]): row["value"]
        for _, row in system_df.iterrows()
    }

    lines = [
        "# Report visualization summary",
        "",
        "## Key talking points",
        "",
        f"- Best final budget by utility: `{best_budget['budget_mode']}` with utility `{best_budget['avg_utility']:.4f}`.",
        f"- High-budget result: utility `{high['avg_utility']:.4f}`, EM `{high['avg_em']:.4f}`, F1 proxy `{high['avg_f1_proxy']:.4f}`, tokens `{high['avg_tokens']:.2f}`.",
        f"- Bandit gate: avg regret `{overview.get(('Bandit Gate', 'avg_regret'), 'n/a')}`, exact-best rate `{overview.get(('Bandit Gate', 'exact_best_rate'), 'n/a')}`.",
        f"- Critic validation: QR Spearman `{qr_s:.4f}`, AG Spearman `{ag_s:.4f}`.",
        f"- Online gate: `{overview.get(('RL Gate', 'online_rl_ready'), 'n/a')}`, deployment mode `{overview.get(('RL Gate', 'deployment_mode'), 'n/a')}`.",
        "",
        "## Caveat for paper comparison",
        "",
        "The paper comparison figure is only a reference. The papers report official HotpotQA EM/F1, while this project uses a local HotpotQA-large split, local LLM settings, and `f1_proxy` plus utility/cost metrics.",
        "",
        "## Generated files",
        "",
    ]
    for path in generated:
        lines.append(f"- `{path.name}`")
    lines.append("")

    md_path = out_dir / "report_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report visualizations from MAP-RAG-Gym outputs.")
    parser.add_argument("--outputs-dir", default="outputs", help="Directory containing run outputs.")
    parser.add_argument("--out-dir", default=None, help="Directory for PNG/Markdown artifacts.")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    metrics_dir = outputs_dir / "metrics"
    out_dir = Path(args.out_dir) if args.out_dir else outputs_dir / "report_visualizations"
    _ensure_dir(out_dir)
    _style()

    macro_df = _read_csv(metrics_dir / "metrics_macro_budget_summary.csv")
    critic_df = _read_csv(metrics_dir / "metrics_micro_critic_summary.csv")
    system_df = _read_csv(metrics_dir / "metrics_system_overview.csv")
    deploy_df = _maybe_read_csv(metrics_dir / "metrics_micro_critic_deployment.csv")
    gate_df = _maybe_read_csv(metrics_dir / "metrics_selective_critic_verification.csv")
    config_df = _maybe_read_csv(metrics_dir / "cv_ensemble_bandit_configs.csv")

    generated: list[Path] = []
    generated.append(plot_budget_quality(macro_df, out_dir))
    generated.append(plot_budget_costs(macro_df, out_dir))
    generated.append(plot_macro_tradeoff(macro_df, out_dir))
    generated.append(plot_workflow_distribution(macro_df, out_dir))
    generated.append(plot_critic_summary(critic_df, out_dir))
    maybe = plot_critic_deployment(deploy_df, out_dir)
    if maybe:
        generated.append(maybe)
    maybe = plot_selective_critic_gate(gate_df, out_dir)
    if maybe:
        generated.append(maybe)
    maybe = plot_bandit_cv(config_df, out_dir)
    if maybe:
        generated.append(maybe)
    generated.extend(plot_paper_reference(macro_df, out_dir))
    generated.append(make_budget_table(macro_df, out_dir))
    generated.append(make_system_gate_table(system_df, out_dir))
    generated.append(write_markdown_summary(out_dir, macro_df, critic_df, system_df, generated))

    print(f"Wrote {len(generated)} report artifacts to {out_dir}")
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    main()
