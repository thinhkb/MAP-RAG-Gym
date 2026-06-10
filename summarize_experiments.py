"""Summarize experimental setup and training/evaluation results.

This script scans MAP-RAG-Gym output artifacts and writes a compact report that can
be used in a paper, README, or progress note.

Usage:
    python summarize_experiments.py
    python summarize_experiments.py --output-dir outputs/2wikimultihopqa_gemma3
    python summarize_experiments.py --out reports/experiment_summary.md --json-out reports/experiment_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("summarize_experiments")


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
DEFAULT_REPORT = ROOT / "outputs" / "experimental_setup_and_train_results_summary.md"
DEFAULT_JSON = ROOT / "outputs" / "experimental_setup_and_train_results_summary.json"


def load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return str(value)
    return str(value)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def get_nested(obj: Mapping[str, Any] | None, keys: Sequence[str], default: Any = None) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def infer_dataset_info(output_dir: Path, package: Mapping[str, Any] | None) -> dict[str, Any]:
    dataset = get_nested(package, ["manifest", "dataset"], {})
    settings = get_nested(package, ["manifest", "settings"], {})
    coverage = get_nested(package, ["macro_layer", "counterfactual_rollout_coverage"], {})

    return {
        "name": dataset.get("name") if isinstance(dataset, Mapping) else None,
        "split": dataset.get("split") if isinstance(dataset, Mapping) else None,
        "effective_questions": dataset.get("effective_questions") if isinstance(dataset, Mapping) else None,
        "policy_bundle": settings.get("policy_bundle") if isinstance(settings, Mapping) else None,
        "final_eval": settings.get("final_eval") if isinstance(settings, Mapping) else None,
        "final_report": settings.get("final_report") if isinstance(settings, Mapping) else None,
        "rollout_coverage": coverage if isinstance(coverage, Mapping) else {},
        "output_dir": rel(output_dir),
    }


def summarize_budget_results(output_dir: Path, package: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    metric_rows = load_csv(output_dir / "metrics" / "metrics_macro_budget_summary.csv")
    if metric_rows:
        return [
            {
                "budget_mode": r.get("budget_mode"),
                "method": r.get("recommended_method"),
                "num_runs": r.get("num_runs"),
                "avg_utility": r.get("avg_utility"),
                "avg_em": r.get("avg_em"),
                "avg_f1_proxy": r.get("avg_f1_proxy"),
                "avg_tokens": r.get("avg_tokens"),
                "avg_latency_ms": r.get("avg_latency_ms"),
                "workflow_distribution": r.get("workflow_distribution"),
                "zero_utility_rate": r.get("zero_utility_rate"),
            }
            for r in metric_rows
            if r.get("budget_mode")
        ]

    final_eval = get_nested(package, ["evaluation", "final_budget_eval"], {})
    if not isinstance(final_eval, Mapping):
        return []
    rows: list[dict[str, Any]] = []
    for budget, values in final_eval.items():
        if not isinstance(values, Mapping):
            continue
        rows.append(
            {
                "budget_mode": budget,
                "method": values.get("recommended_method"),
                "num_runs": values.get("num_runs"),
                "avg_utility": values.get("avg_utility"),
                "avg_em": values.get("avg_em"),
                "avg_f1_proxy": values.get("avg_f1_proxy"),
                "avg_tokens": values.get("avg_tokens"),
                "avg_latency_ms": values.get("avg_latency_ms"),
                "workflow_distribution": values.get("workflow_counts"),
                "zero_utility_rate": None,
            }
        )
    return rows


def summarize_critic_results(output_dir: Path, package: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    metric_rows = load_csv(output_dir / "metrics" / "metrics_micro_critic_summary.csv")
    if metric_rows:
        return [dict(r) for r in metric_rows if r.get("module")]

    critics = get_nested(package, ["micro_layer", "critic_models"], {})
    if not isinstance(critics, Mapping):
        return []

    rows: list[dict[str, Any]] = []
    for module, values in critics.items():
        if not isinstance(values, Mapping):
            continue
        train_counts = values.get("train_counts", {})
        eval_counts = values.get("eval_counts", {})
        eval_metrics = values.get("eval_metrics", {})
        gate = values.get("readiness_gate", {})
        rows.append(
            {
                "module": module,
                "train_examples": get_nested(train_counts, ["num_examples"]),
                "eval_examples": get_nested(eval_counts, ["num_examples"]),
                "eval_mae": get_nested(eval_metrics, ["mae"]),
                "eval_rmse": get_nested(eval_metrics, ["rmse"]),
                "eval_pearson": get_nested(eval_metrics, ["pearson"]),
                "eval_spearman": get_nested(eval_metrics, ["spearman"]),
                "ready_as_offline_reward_model": get_nested(gate, ["ready_as_offline_reward_model"]),
            }
        )
    return rows


def summarize_bandit(output_dir: Path, package: Mapping[str, Any] | None) -> dict[str, Any]:
    regate = load_json(output_dir / "regate_report.json")
    cv_report = load_json(output_dir / "cv_ensemble_report.json")

    if isinstance(regate, Mapping):
        check = regate.get("bandit_check", {})
        if isinstance(check, Mapping):
            return {
                "source": rel(output_dir / "regate_report.json"),
                "model": check.get("improved_model"),
                "best_config": get_nested(check, ["best_config", "name"]),
                "avg_regret": check.get("avg_regret"),
                "exact_best_rate": check.get("exact_best_rate"),
                "meets_online_threshold": check.get("meets_online_threshold"),
            }

    if isinstance(cv_report, Mapping):
        best = cv_report.get("best_result", {})
        summary = cv_report.get("cv_summary", {})
        return {
            "source": rel(output_dir / "cv_ensemble_report.json"),
            "model": rel(output_dir / "cv_ensemble_high_bandit.joblib"),
            "best_config": get_nested(best, ["name"]),
            "avg_regret": get_nested(best, ["cv_avg_regret"], get_nested(summary, ["best_cv_regret"])),
            "exact_best_rate": get_nested(best, ["cv_avg_best_rate"], get_nested(summary, ["best_cv_best_rate"])),
            "meets_online_threshold": get_nested(summary, ["meets_online_gate"]),
        }

    improved = get_nested(package, ["macro_layer", "budget_policies", "high", "improved_bandit"], {})
    if isinstance(improved, Mapping):
        config = improved.get("config", {})
        return {
            "source": rel(output_dir / "full_system_rl_package.json"),
            "model": get_nested(package, ["macro_layer", "budget_policies", "high", "router_settings", "bandit_router_model"]),
            "best_config": get_nested(config, ["name"]),
            "avg_regret": improved.get("regret"),
            "exact_best_rate": improved.get("exact_best_rate"),
            "meets_online_threshold": None,
        }

    return {}


def summarize_selective_critic(output_dir: Path) -> dict[str, Any]:
    verification = load_json(output_dir / "selective_critic_verification.json")
    regate = load_json(output_dir / "regate_report.json")

    result: dict[str, Any] = {}
    if isinstance(verification, Mapping):
        result.update(
            {
                "baseline_utility": get_nested(verification, ["baseline", "avg_utility"]),
                "baseline_tokens": get_nested(verification, ["baseline", "avg_tokens"]),
                "full_critic_utility": get_nested(verification, ["full_critic", "avg_utility"]),
                "full_critic_tokens": get_nested(verification, ["full_critic", "avg_tokens"]),
                "best_passing_gate": verification.get("best_passing_gate"),
                "best_critic_using_gate": verification.get("best_critic_using_gate"),
                "source": rel(output_dir / "selective_critic_verification.json"),
            }
        )

    if isinstance(regate, Mapping):
        critic_check = regate.get("critic_check", {})
        if isinstance(critic_check, Mapping):
            result.update(
                {
                    "online_strategy": critic_check.get("strategy"),
                    "online_token_multiplier": critic_check.get("token_multiplier"),
                    "online_utility_gap": critic_check.get("utility_gap"),
                    "meets_online_threshold": critic_check.get("meets_online_threshold"),
                    "verified_on_holdout": critic_check.get("verified_on_holdout"),
                }
            )
    return result


def summarize_rl_status(output_dir: Path, package: Mapping[str, Any] | None) -> dict[str, Any]:
    regate = load_json(output_dir / "regate_report.json")
    if isinstance(regate, Mapping):
        return {
            "offline_rl_ready": get_nested(package, ["stage", "ready_for_offline_full_system_rl"]),
            "online_rl_ready": regate.get("ready_for_online_rl"),
            "deployment_mode": get_nested(package, ["stage", "deployment_mode"]),
            "recommended_next_stage": get_nested(package, ["stage", "recommended_next_stage"]),
            "deployment_recommendation": regate.get("deployment_recommendation"),
            "online_blockers": regate.get("online_blockers", []),
            "offline_blockers": get_nested(package, ["stage", "offline_blockers"], []),
        }

    return {
        "offline_rl_ready": get_nested(package, ["stage", "ready_for_offline_full_system_rl"]),
        "online_rl_ready": get_nested(package, ["stage", "ready_for_online_full_system_rl"]),
        "deployment_mode": get_nested(package, ["stage", "deployment_mode"]),
        "recommended_next_stage": get_nested(package, ["stage", "recommended_next_stage"]),
        "deployment_recommendation": None,
        "online_blockers": get_nested(package, ["stage", "online_blockers"], []),
        "offline_blockers": get_nested(package, ["stage", "offline_blockers"], []),
    }


def build_summary(output_dir: Path) -> dict[str, Any]:
    package_path = first_existing(output_dir / "full_system_rl_package.json")
    package_raw = load_json(package_path) if package_path else None
    package = package_raw if isinstance(package_raw, Mapping) else None

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output_dir": rel(output_dir),
        "source_files": {
            "full_system_rl_package": rel(package_path) if package_path else None,
            "macro_budget_metrics": rel(output_dir / "metrics" / "metrics_macro_budget_summary.csv"),
            "micro_critic_metrics": rel(output_dir / "metrics" / "metrics_micro_critic_summary.csv"),
            "cv_ensemble_report": rel(output_dir / "cv_ensemble_report.json"),
            "selective_critic_verification": rel(output_dir / "selective_critic_verification.json"),
            "regate_report": rel(output_dir / "regate_report.json"),
        },
        "experimental_setup": infer_dataset_info(output_dir, package),
        "macro_budget_results": summarize_budget_results(output_dir, package),
        "micro_critic_results": summarize_critic_results(output_dir, package),
        "bandit_result": summarize_bandit(output_dir, package),
        "selective_critic_result": summarize_selective_critic(output_dir),
        "rl_status": summarize_rl_status(output_dir, package),
    }


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return "_No data found._\n"
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(cell) for cell in row) + " |")
    return "\n".join(out) + "\n"


def render_markdown(summary: Mapping[str, Any]) -> str:
    setup = summary["experimental_setup"]
    macro_rows = summary["macro_budget_results"]
    critic_rows = summary["micro_critic_results"]
    bandit = summary["bandit_result"]
    selective = summary["selective_critic_result"]
    rl = summary["rl_status"]

    lines: list[str] = []
    lines.append("# Experimental Setup and Training Results Summary")
    lines.append("")
    lines.append(f"Generated at UTC: `{summary['generated_at_utc']}`")
    lines.append(f"Output directory: `{summary['output_dir']}`")
    lines.append("")

    lines.append("## 1. Experimental Setup")
    lines.append("")
    lines.append(table(
        ["Item", "Value"],
        [
            ["Dataset/package name", setup.get("name")],
            ["Split/stage", setup.get("split")],
            ["Effective questions", setup.get("effective_questions")],
            ["Policy bundle", setup.get("policy_bundle")],
            ["Final evaluation", setup.get("final_eval")],
            ["Final report", setup.get("final_report")],
        ],
    ))

    coverage = setup.get("rollout_coverage") or {}
    if isinstance(coverage, Mapping) and coverage:
        lines.append("### Counterfactual Rollout Coverage")
        lines.append("")
        coverage_rows = []
        for budget, data in coverage.items():
            if not isinstance(data, Mapping):
                continue
            coverage_rows.append(
                [
                    budget,
                    data.get("num_questions"),
                    data.get("num_runs"),
                    data.get("workflow_counts"),
                    data.get("workflow_avg_utility"),
                    data.get("has_counterfactual_workflows"),
                ]
            )
        lines.append(table(
            ["Budget", "Questions", "Runs", "Workflow counts", "Workflow avg utility", "Has counterfactuals"],
            coverage_rows,
        ))

    lines.append("## 2. Macro Budget Policy Results")
    lines.append("")
    lines.append(table(
        ["Budget", "Method", "Runs", "Utility", "EM", "F1", "Tokens", "Latency ms", "Workflows", "Zero utility"],
        [
            [
                r.get("budget_mode"),
                r.get("method"),
                r.get("num_runs"),
                r.get("avg_utility"),
                r.get("avg_em"),
                r.get("avg_f1_proxy"),
                r.get("avg_tokens"),
                r.get("avg_latency_ms"),
                r.get("workflow_distribution"),
                r.get("zero_utility_rate"),
            ]
            for r in macro_rows
        ],
    ))

    lines.append("## 3. High-Budget Bandit Training")
    lines.append("")
    lines.append(table(
        ["Metric", "Value"],
        [
            ["Source", bandit.get("source")],
            ["Model", bandit.get("model")],
            ["Best config", bandit.get("best_config")],
            ["Average regret", bandit.get("avg_regret")],
            ["Exact best rate", bandit.get("exact_best_rate")],
            ["Meets online threshold", bandit.get("meets_online_threshold")],
        ],
    ))

    lines.append("## 4. Micro Critic Training Results")
    lines.append("")
    lines.append(table(
        ["Module", "Train examples", "Eval examples", "MAE", "RMSE", "Pearson", "Spearman", "Ready offline reward"],
        [
            [
                r.get("module"),
                r.get("train_examples"),
                r.get("eval_examples"),
                r.get("eval_mae"),
                r.get("eval_rmse"),
                r.get("eval_pearson"),
                r.get("eval_spearman"),
                r.get("ready_as_offline_reward_model"),
            ]
            for r in critic_rows
        ],
    ))

    lines.append("## 5. Selective Critic Verification")
    lines.append("")
    best_gate = selective.get("best_passing_gate") if isinstance(selective, Mapping) else None
    lines.append(table(
        ["Metric", "Value"],
        [
            ["Baseline utility", selective.get("baseline_utility")],
            ["Baseline tokens", selective.get("baseline_tokens")],
            ["Full critic utility", selective.get("full_critic_utility")],
            ["Full critic tokens", selective.get("full_critic_tokens")],
            ["Best passing gate", get_nested(best_gate, ["gate_threshold"]) if isinstance(best_gate, Mapping) else None],
            ["Best passing token multiplier", get_nested(best_gate, ["token_multiplier"]) if isinstance(best_gate, Mapping) else None],
            ["Best passing utility vs base", get_nested(best_gate, ["utility_vs_base"]) if isinstance(best_gate, Mapping) else None],
            ["Online strategy", selective.get("online_strategy")],
            ["Meets online threshold", selective.get("meets_online_threshold")],
            ["Verified on holdout", selective.get("verified_on_holdout")],
        ],
    ))

    lines.append("## 6. RL Readiness / Deployment Status")
    lines.append("")
    lines.append(table(
        ["Metric", "Value"],
        [
            ["Offline RL ready", rl.get("offline_rl_ready")],
            ["Online RL ready", rl.get("online_rl_ready")],
            ["Deployment mode", rl.get("deployment_mode")],
            ["Recommended next stage", rl.get("recommended_next_stage")],
            ["Deployment recommendation", rl.get("deployment_recommendation")],
            ["Offline blockers", "; ".join(rl.get("offline_blockers") or [])],
            ["Online blockers", "; ".join(rl.get("online_blockers") or [])],
        ],
    ))

    lines.append("## 7. Source Files")
    lines.append("")
    source_files = summary.get("source_files", {})
    for name, path in source_files.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experimental setup and training results.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing experiment outputs. Default: outputs/",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_REPORT,
        help="Markdown summary output path.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_JSON,
        help="Machine-readable JSON summary output path.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print warnings/errors.",
    )
    parser.add_argument(
        "--no-full-log",
        action="store_true",
        help="Do not print the full summary content to logs.",
    )
    parser.add_argument(
        "--log-format",
        choices=["json", "markdown"],
        default="json",
        help="Full log format. Default: json.",
    )
    return parser.parse_args()


def configure_logging(quiet: bool) -> None:
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def log_source_files(output_dir: Path) -> None:
    sources = {
        "full_system_rl_package": output_dir / "full_system_rl_package.json",
        "macro_budget_metrics": output_dir / "metrics" / "metrics_macro_budget_summary.csv",
        "micro_critic_metrics": output_dir / "metrics" / "metrics_micro_critic_summary.csv",
        "cv_ensemble_report": output_dir / "cv_ensemble_report.json",
        "selective_critic_verification": output_dir / "selective_critic_verification.json",
        "regate_report": output_dir / "regate_report.json",
    }
    for name, path in sources.items():
        status = "FOUND" if path.exists() else "MISSING"
        log.info("source %-30s %s %s", name, status, rel(path))


def main() -> None:
    args = parse_args()
    configure_logging(args.quiet)

    output_dir = args.output_dir.resolve()
    log.info("start summary generation")
    log.info("output_dir=%s", rel(output_dir))
    log.info("markdown_out=%s", rel(args.out))
    log.info("json_out=%s", rel(args.json_out))

    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    log.info("checking source files")
    log_source_files(output_dir)

    log.info("building summary data")
    summary = build_summary(output_dir)

    setup = summary["experimental_setup"]
    macro_rows = summary["macro_budget_results"]
    critic_rows = summary["micro_critic_results"]
    bandit = summary["bandit_result"]
    selective = summary["selective_critic_result"]
    rl = summary["rl_status"]

    log.info(
        "setup dataset=%s split=%s effective_questions=%s",
        setup.get("name"),
        setup.get("split"),
        setup.get("effective_questions"),
    )
    log.info("macro budget rows=%d", len(macro_rows))
    for row in macro_rows:
        log.info(
            "macro budget=%s method=%s utility=%s em=%s f1=%s tokens=%s",
            row.get("budget_mode"),
            row.get("method"),
            row.get("avg_utility"),
            row.get("avg_em"),
            row.get("avg_f1_proxy"),
            row.get("avg_tokens"),
        )

    log.info(
        "bandit config=%s regret=%s best_rate=%s online_gate=%s",
        bandit.get("best_config"),
        bandit.get("avg_regret"),
        bandit.get("exact_best_rate"),
        bandit.get("meets_online_threshold"),
    )

    log.info("critic rows=%d", len(critic_rows))
    for row in critic_rows:
        log.info(
            "critic module=%s train=%s eval=%s spearman=%s ready=%s",
            row.get("module"),
            row.get("train_examples"),
            row.get("eval_examples"),
            row.get("eval_spearman"),
            row.get("ready_as_offline_reward_model"),
        )

    log.info(
        "selective critic baseline_utility=%s full_critic_utility=%s online_strategy=%s",
        selective.get("baseline_utility"),
        selective.get("full_critic_utility"),
        selective.get("online_strategy"),
    )
    log.info(
        "rl status offline_ready=%s online_ready=%s deployment=%s",
        rl.get("offline_rl_ready"),
        rl.get("online_rl_ready"),
        rl.get("deployment_mode"),
    )

    log.info("rendering markdown")
    markdown = render_markdown(summary)

    if not args.no_full_log:
        log.info("full summary dump start format=%s", args.log_format)
        if args.log_format == "markdown":
            for line in markdown.splitlines():
                log.info("%s", line)
        else:
            full_json = json.dumps(summary, indent=2, ensure_ascii=False)
            for line in full_json.splitlines():
                log.info("%s", line)
        log.info("full summary dump end")

    log.info("writing outputs")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8")
    args.json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("done")
    print(f"Wrote Markdown summary: {rel(args.out)}")
    print(f"Wrote JSON summary: {rel(args.json_out)}")


if __name__ == "__main__":
    main()
