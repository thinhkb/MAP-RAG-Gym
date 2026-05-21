from __future__ import annotations

"""
Export evaluation metrics per layer (macro / micro / system) as CSV files.

Outputs:
  outputs/metrics_macro_budget_summary.csv    -- per-budget routing metrics
  outputs/metrics_macro_per_question.csv      -- per-question routing decisions
  outputs/metrics_macro_bandit_holdout.csv    -- per-question bandit holdout diagnostics
  outputs/metrics_micro_critic_summary.csv    -- critic quality per module
  outputs/metrics_micro_critic_predictions.csv-- critic per-example predictions
  outputs/metrics_system_overview.csv         -- system-level gate/readiness
  outputs/metrics_system_promotion.csv        -- promotion comparison table
  outputs/metrics_bandit_configs.csv          -- bandit model config comparison
"""

import argparse
import csv
import os
from pathlib import Path
from statistics import mean

from map_rag_gym.utils.io import read_json


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _write_csv(path: str, rows: list[dict], fieldnames: list[str] | None = None):
    if not rows:
        print(f"  [SKIP] {path} (no data)")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [OK] {path} ({len(rows)} rows)")


# ── Macro Layer ──────────────────────────────────────────────────────────

def export_macro_budget_summary(eval_path: str, out_path: str):
    """Export per-budget routing summary metrics."""
    data = read_json(eval_path)
    budget_results = data.get("budget_results", {})
    rows = []
    for budget_mode in ["low", "medium", "high"]:
        result = budget_results.get(budget_mode, {})
        summary = result.get("summary", {})
        wf_counts = summary.get("workflow_counts", {})
        # Compute additional metrics
        per_question = result.get("per_question", [])
        utilities = [_safe_float(pq.get("utility_total")) for pq in per_question]
        ems = [_safe_float(pq.get("em")) for pq in per_question]
        f1s = [_safe_float(pq.get("f1_proxy")) for pq in per_question]
        rows.append({
            "budget_mode": budget_mode,
            "recommended_method": result.get("recommended_method", ""),
            "num_runs": summary.get("num_runs", 0),
            "avg_utility": round(summary.get("avg_utility", 0.0), 4),
            "avg_em": round(summary.get("avg_em", 0.0), 4),
            "avg_f1_proxy": round(summary.get("avg_f1_proxy", 0.0), 4),
            "avg_process_score": round(summary.get("avg_process_score", 0.0), 4),
            "avg_tokens": round(summary.get("avg_tokens", 0.0), 2),
            "avg_retrieval_calls": round(summary.get("avg_retrieval_calls", 0.0), 2),
            "avg_latency_ms": round(summary.get("avg_latency_ms", 0.0), 2),
            "workflow_distribution": str(wf_counts),
            "utility_std": round((sum((u - mean(utilities))**2 for u in utilities) / max(1, len(utilities))) ** 0.5, 4) if utilities else 0.0,
            "em_rate": round(sum(1 for e in ems if e > 0) / max(1, len(ems)), 4) if ems else 0.0,
            "zero_utility_rate": round(sum(1 for u in utilities if u <= 0) / max(1, len(utilities)), 4) if utilities else 0.0,
        })
    _write_csv(out_path, rows)


def export_macro_per_question(eval_path: str, out_path: str):
    """Export per-question routing decisions across all budgets."""
    data = read_json(eval_path)
    budget_results = data.get("budget_results", {})
    rows = []
    for budget_mode in ["low", "medium", "high"]:
        result = budget_results.get(budget_mode, {})
        for pq in result.get("per_question", []):
            rows.append({
                "budget_mode": budget_mode,
                "question_id": pq.get("question_id", ""),
                "question": pq.get("question", "")[:120],
                "gold_answer": pq.get("gold_answer", ""),
                "selected_method": pq.get("selected_method", ""),
                "workflow_id": pq.get("workflow_id", ""),
                "utility_total": round(_safe_float(pq.get("utility_total")), 4),
                "em": _safe_float(pq.get("em")),
                "f1_proxy": round(_safe_float(pq.get("f1_proxy")), 4),
            })
    _write_csv(out_path, rows)


def export_macro_bandit_holdout(bandit_meta_paths: dict[str, str], out_path: str):
    """Export bandit holdout diagnostics per question per budget."""
    rows = []
    for budget_mode, meta_path in sorted(bandit_meta_paths.items()):
        if not Path(meta_path).exists():
            continue
        meta = read_json(meta_path)
        holdout_eval = meta.get("holdout_policy_eval", {})
        # Add summary row
        for preview in holdout_eval.get("preview", []):
            rows.append({
                "budget_mode": budget_mode,
                "question": preview.get("question", "")[:120],
                "predicted_workflow": preview.get("predicted_workflow", ""),
                "predicted_confidence": round(_safe_float(preview.get("predicted_confidence")), 4),
                "oracle_workflow": preview.get("oracle_workflow", ""),
                "oracle_reward": round(_safe_float(preview.get("oracle_reward")), 4),
                "chosen_reward": round(_safe_float(preview.get("chosen_reward")), 4),
                "regret": round(_safe_float(preview.get("regret")), 4),
                "correct": int(preview.get("predicted_workflow") == preview.get("oracle_workflow")),
            })
        # Also add diagnostics
        for diag in meta.get("holdout_diagnostics_preview", []):
            if not any(r["question"][:80] == diag.get("question", "")[:80] for r in rows if r["budget_mode"] == budget_mode):
                rows.append({
                    "budget_mode": budget_mode,
                    "question": diag.get("question", "")[:120],
                    "predicted_workflow": "",
                    "predicted_confidence": 0.0,
                    "oracle_workflow": diag.get("best_workflow", ""),
                    "oracle_reward": round(_safe_float(diag.get("best_reward")), 4),
                    "chosen_reward": 0.0,
                    "regret": round(_safe_float(diag.get("reward_span")), 4),
                    "correct": 0,
                })
    _write_csv(out_path, rows)


# ── Micro Layer ──────────────────────────────────────────────────────────

def export_micro_critic_summary(critic_meta_paths: dict[str, str], out_path: str):
    """Export critic quality metrics per module."""
    rows = []
    for module, meta_path in sorted(critic_meta_paths.items()):
        if not Path(meta_path).exists():
            continue
        meta = read_json(meta_path)
        eval_data = meta.get("evaluation", {})
        overall = eval_data.get("overall", {})
        train_counts = meta.get("train_counts", {})
        eval_counts = meta.get("eval_counts", {})
        rows.append({
            "module": module,
            "train_examples": train_counts.get("num_examples", 0),
            "eval_examples": eval_counts.get("num_examples", 0),
            "train_budget_low": train_counts.get("budget_modes", {}).get("low", 0),
            "train_budget_medium": train_counts.get("budget_modes", {}).get("medium", 0),
            "train_budget_high": train_counts.get("budget_modes", {}).get("high", 0),
            "eval_mae": round(_safe_float(overall.get("mae")), 4),
            "eval_rmse": round(_safe_float(overall.get("rmse")), 4),
            "eval_pearson": round(_safe_float(overall.get("pearson")), 4),
            "eval_spearman": round(_safe_float(overall.get("spearman")), 4),
            "ready_as_offline_reward_model": str(overall.get("spearman", 0) >= 0.25),
        })
        # Also export per-module breakdown if available
        for mod_name, mod_eval in eval_data.get("per_module", {}).items():
            if mod_name != module:
                rows.append({
                    "module": f"{module}_{mod_name}",
                    "train_examples": 0,
                    "eval_examples": mod_eval.get("count", 0),
                    "train_budget_low": 0,
                    "train_budget_medium": 0,
                    "train_budget_high": 0,
                    "eval_mae": round(_safe_float(mod_eval.get("mae")), 4),
                    "eval_rmse": round(_safe_float(mod_eval.get("rmse")), 4),
                    "eval_pearson": round(_safe_float(mod_eval.get("pearson")), 4),
                    "eval_spearman": round(_safe_float(mod_eval.get("spearman")), 4),
                    "ready_as_offline_reward_model": str(mod_eval.get("spearman", 0) >= 0.25),
                })
    _write_csv(out_path, rows)


def export_micro_critic_predictions(critic_meta_paths: dict[str, str], out_path: str):
    """Export critic per-example predictions (preview samples)."""
    rows = []
    for module, meta_path in sorted(critic_meta_paths.items()):
        if not Path(meta_path).exists():
            continue
        meta = read_json(meta_path)
        for pred in meta.get("predictions", []):
            rows.append({
                "module": module,
                "example_id": pred.get("example_id", ""),
                "question_id": pred.get("question_id", ""),
                "question": str(pred.get("question", ""))[:120],
                "action_text": str(pred.get("action_text", ""))[:120],
                "target": round(_safe_float(pred.get("target")), 4),
                "prediction": round(_safe_float(pred.get("prediction")), 4),
                "abs_error": round(_safe_float(pred.get("abs_error")), 4),
            })
    _write_csv(out_path, rows)


# ── Micro Layer: Critic Deployment ───────────────────────────────────────

def export_micro_critic_deployment(critic_eval_path: str, out_path: str):
    """Export critic deployment comparison (with/without critic)."""
    if not Path(critic_eval_path).exists():
        print(f"  [SKIP] {out_path} (no critic eval)")
        return
    data = read_json(critic_eval_path)
    summary = data.get("summary", {})
    rows = []
    for method_name, method_stats in summary.items():
        if isinstance(method_stats, dict) and "avg_utility" in method_stats:
            rows.append({
                "method": method_name,
                "is_critic": "yes" if "critic" in method_name.lower() else "no",
                "num_runs": method_stats.get("num_runs", 0),
                "avg_utility": round(_safe_float(method_stats.get("avg_utility")), 4),
                "avg_em": round(_safe_float(method_stats.get("avg_em")), 4),
                "avg_f1_proxy": round(_safe_float(method_stats.get("avg_f1_proxy")), 4),
                "avg_process_score": round(_safe_float(method_stats.get("avg_process_score")), 4),
                "avg_tokens": round(_safe_float(method_stats.get("avg_tokens")), 2),
                "avg_retrieval_calls": round(_safe_float(method_stats.get("avg_retrieval_calls")), 2),
                "avg_latency_ms": round(_safe_float(method_stats.get("avg_latency_ms")), 2),
                "workflow_counts": str(method_stats.get("workflow_counts", {})),
            })
    if len(rows) >= 2:
        base = next((r for r in rows if r["is_critic"] == "no"), None)
        critic = next((r for r in rows if r["is_critic"] == "yes"), None)
        if base and critic:
            rows.append({
                "method": "DELTA (critic - base)",
                "is_critic": "delta",
                "num_runs": 0,
                "avg_utility": round(critic["avg_utility"] - base["avg_utility"], 4),
                "avg_em": round(critic["avg_em"] - base["avg_em"], 4),
                "avg_f1_proxy": round(critic["avg_f1_proxy"] - base["avg_f1_proxy"], 4),
                "avg_process_score": round(critic["avg_process_score"] - base["avg_process_score"], 4),
                "avg_tokens": round(critic["avg_tokens"] - base["avg_tokens"], 2),
                "avg_retrieval_calls": round(critic["avg_retrieval_calls"] - base["avg_retrieval_calls"], 2),
                "avg_latency_ms": round(critic["avg_latency_ms"] - base["avg_latency_ms"], 2),
                "workflow_counts": "",
            })
            rows.append({
                "method": "TOKEN MULTIPLIER",
                "is_critic": "ratio",
                "num_runs": 0,
                "avg_utility": 0.0,
                "avg_em": 0.0,
                "avg_f1_proxy": 0.0,
                "avg_process_score": 0.0,
                "avg_tokens": round(critic["avg_tokens"] / max(1, base["avg_tokens"]), 4),
                "avg_retrieval_calls": 0.0,
                "avg_latency_ms": round(critic["avg_latency_ms"] / max(1, base["avg_latency_ms"]), 4),
                "workflow_counts": "",
            })
    _write_csv(out_path, rows)


# ── System Layer ─────────────────────────────────────────────────────────

def export_system_overview(
    rl_package_path: str,
    promotion_path: str,
    regate_path: str,
    out_path: str,
):
    """Export system-level gate and readiness overview."""
    rows = []
    if Path(rl_package_path).exists():
        pkg = read_json(rl_package_path)
        stage = pkg.get("stage", {})
        rows.append({"category": "RL Gate", "metric": "offline_rl_ready", "value": str(stage.get("ready_for_offline_full_system_rl", False))})
        rows.append({"category": "RL Gate", "metric": "online_rl_ready", "value": str(stage.get("ready_for_online_full_system_rl", False))})
        rows.append({"category": "RL Gate", "metric": "deployment_mode", "value": str(stage.get("deployment_mode", ""))})
        rows.append({"category": "RL Gate", "metric": "recommended_next_stage", "value": str(stage.get("recommended_next_stage", ""))})
        for i, blocker in enumerate(stage.get("online_blockers", [])):
            rows.append({"category": "Online Blocker", "metric": f"blocker_{i+1}", "value": blocker})

    if Path(regate_path).exists():
        regate = read_json(regate_path)
        bc = regate.get("bandit_check", {})
        rows.append({"category": "Bandit Gate", "metric": "avg_regret", "value": str(bc.get("avg_regret", ""))})
        rows.append({"category": "Bandit Gate", "metric": "exact_best_rate", "value": str(bc.get("exact_best_rate", ""))})
        rows.append({"category": "Bandit Gate", "metric": "meets_online_threshold", "value": str(bc.get("meets_online_threshold", ""))})
        cc = regate.get("critic_check", {})
        token_mult = cc.get("token_multiplier") or cc.get("estimated_token_multiplier", "")
        rows.append({"category": "Critic Gate", "metric": "token_multiplier", "value": str(token_mult)})
        rows.append({"category": "Critic Gate", "metric": "utility_gap", "value": str(cc.get("utility_gap", ""))})
        rows.append({"category": "Critic Gate", "metric": "meets_online_threshold", "value": str(cc.get("meets_online_threshold", ""))})

    if Path(promotion_path).exists():
        promo = read_json(promotion_path)
        rows.append({"category": "Promotion", "metric": "promoted", "value": str(promo.get("promoted", False))})
        for budget_mode, comp in promo.get("budget_comparison", {}).items():
            rows.append({"category": "Promotion", "metric": f"{budget_mode}_utility_delta", "value": str(comp.get("delta", 0.0))})

    _write_csv(out_path, rows)


def export_system_promotion(
    promotion_path: str,
    promotion_check_path: str,
    out_path: str,
):
    """Export promotion comparison table."""
    rows = []
    if Path(promotion_check_path).exists():
        pc = read_json(promotion_check_path)
        for budget_mode, report in pc.get("budget_reports", {}).items():
            rows.append({
                "budget_mode": budget_mode,
                "passed": report.get("passed", False),
                "candidate_method": report.get("candidate_method", ""),
                "frozen_method": report.get("frozen_method", ""),
                "candidate_utility": round(_safe_float(report.get("candidate_utility")), 4),
                "frozen_utility": round(_safe_float(report.get("frozen_utility")), 4),
                "utility_delta": round(_safe_float(report.get("utility_delta")), 4),
                "candidate_tokens": round(_safe_float(report.get("candidate_tokens")), 2),
                "frozen_tokens": round(_safe_float(report.get("frozen_tokens")), 2),
                "candidate_workflows": str(report.get("candidate_workflow_counts", {})),
                "frozen_workflows": str(report.get("frozen_workflow_counts", {})),
            })
    _write_csv(out_path, rows)


def export_bandit_configs(report_path: str, out_path: str):
    """Export bandit model config comparison from improve_high_bandit."""
    if not Path(report_path).exists():
        print(f"  [SKIP] {out_path} (no report)")
        return
    data = read_json(report_path)
    rows = []
    best_name = data.get("best_config", {}).get("name", "")
    for result in data.get("all_results", []):
        config = result.get("config", {})
        eval_data = result.get("holdout_policy_eval", {})
        reward_metrics = result.get("reward_metrics", {})
        rows.append({
            "config_name": config.get("name", ""),
            "model_type": config.get("model_type", ""),
            "is_best": "yes" if config.get("name") == best_name else "no",
            "alpha": config.get("alpha", ""),
            "n_estimators": config.get("n_estimators", ""),
            "max_depth": config.get("max_depth", ""),
            "learning_rate": config.get("learning_rate", ""),
            "subsample": config.get("subsample", ""),
            "holdout_regret": round(_safe_float(eval_data.get("avg_regret")), 4),
            "holdout_exact_best_rate": round(_safe_float(eval_data.get("exact_best_rate")), 4),
            "holdout_policy_utility": round(_safe_float(eval_data.get("avg_policy_utility")), 4),
            "holdout_oracle_utility": round(_safe_float(eval_data.get("avg_oracle_utility")), 4),
            "train_mae": round(_safe_float(reward_metrics.get("train_mae")), 4),
            "train_rmse": round(_safe_float(reward_metrics.get("train_rmse")), 4),
            "holdout_mae": round(_safe_float(reward_metrics.get("holdout_mae")), 4),
            "holdout_rmse": round(_safe_float(reward_metrics.get("holdout_rmse")), 4),
        })
    _write_csv(out_path, rows)


def main():
    ap = argparse.ArgumentParser(description="Export evaluation metrics per layer as CSV")
    ap.add_argument("--eval", default="outputs/final_budget_policy_test_eval_rl_ready.json")
    ap.add_argument("--rl_package", default="outputs/full_system_rl_package.json")
    ap.add_argument("--promotion_report", default="outputs/promotion_report.json")
    ap.add_argument("--promotion_check", default="outputs/offline_full_system_rl_guarded/promotion_check.json")
    ap.add_argument("--regate_report", default="outputs/regate_report.json")
    ap.add_argument("--bandit_report", default="outputs/improve_high_bandit_report.json")
    ap.add_argument("--critic_eval", default="outputs/router_eval_large_budget_high_w2w3_det_critic2_val.json")
    ap.add_argument("--out_dir", default="outputs/metrics")
    args = ap.parse_args()

    out_dir = args.out_dir
    print("=== Exporting Macro Layer Metrics ===")
    export_macro_budget_summary(args.eval, f"{out_dir}/metrics_macro_budget_summary.csv")
    export_macro_per_question(args.eval, f"{out_dir}/metrics_macro_per_question.csv")

    bandit_meta_paths = {}
    for budget in ["low", "medium", "high"]:
        path = f"outputs/offline_full_system_rl_guarded/macro_bandit_{budget}.joblib.meta.json"
        if Path(path).exists():
            bandit_meta_paths[budget] = path
    export_macro_bandit_holdout(bandit_meta_paths, f"{out_dir}/metrics_macro_bandit_holdout.csv")

    print("\n=== Exporting Micro Layer Metrics ===")
    critic_meta_paths = {
        "QR": "outputs/process_critic_budget_qr_local.joblib.meta.json",
        "AG": "outputs/process_critic_budget_ag.joblib.meta.json",
    }
    export_micro_critic_summary(critic_meta_paths, f"{out_dir}/metrics_micro_critic_summary.csv")
    export_micro_critic_predictions(critic_meta_paths, f"{out_dir}/metrics_micro_critic_predictions.csv")
    export_micro_critic_deployment(args.critic_eval, f"{out_dir}/metrics_micro_critic_deployment.csv")

    print("\n=== Exporting System Layer Metrics ===")
    export_system_overview(args.rl_package, args.promotion_report, args.regate_report, f"{out_dir}/metrics_system_overview.csv")
    export_system_promotion(args.promotion_report, args.promotion_check, f"{out_dir}/metrics_system_promotion.csv")
    export_bandit_configs(args.bandit_report, f"{out_dir}/metrics_bandit_configs.csv")

    print(f"\nDone. All CSV files saved to {out_dir}/")


if __name__ == "__main__":
    main()
