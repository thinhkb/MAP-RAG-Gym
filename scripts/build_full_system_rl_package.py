from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from map_rag_gym.router.budget import normalize_budget_mode
from map_rag_gym.utils.experiment import build_experiment_manifest
from map_rag_gym.utils.io import read_json, write_json


ADAPTIVE_METHOD_PREFIXES = (
    "rule_based",
    "learned_router",
    "hybrid_router",
    "bandit_router",
    "gated_bandit_router",
    "meta_router",
)


def _parse_key_values(entries: list[str] | None, *, value_required: bool = True) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_entry in entries or []:
        if "=" not in raw_entry:
            raise ValueError(f"Invalid entry '{raw_entry}'. Expected KEY=VALUE.")
        key, value = raw_entry.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or (value_required and not value):
            raise ValueError(f"Invalid entry '{raw_entry}'. Expected KEY=VALUE.")
        parsed[key] = value
    return parsed


def _is_adaptive_method(method_name: str | None) -> bool:
    name = str(method_name or "").strip()
    if not name or name.startswith("fixed_"):
        return False
    return any(name == prefix or name.startswith(f"{prefix}_") for prefix in ADAPTIVE_METHOD_PREFIXES)


def _round(value: Any, digits: int = 4) -> float:
    try:
        return round(float(value), digits)
    except Exception:
        return 0.0


def _safe_read_json(path: str | None) -> dict:
    if not path:
        return {}
    return read_json(path)


def _summarize_policy_bundle(bundle: dict) -> dict:
    policies = bundle.get("budget_policies", {})
    rows = {}
    for budget_mode, policy in sorted(policies.items()):
        method = policy.get("recommended_method")
        rows[budget_mode] = {
            "recommended_method": method,
            "is_adaptive": _is_adaptive_method(method),
            "constraints": policy.get("constraints", {}),
            "router_settings": policy.get("router_settings", {}),
            "source_eval_file": policy.get("source_eval_file"),
        }
    return rows


def _summarize_final_eval(final_eval: dict, reference_eval: dict | None = None) -> dict:
    reference_eval = reference_eval or {}
    reference_results = reference_eval.get("budget_results", {})
    rows = {}
    for budget_mode, result in sorted(final_eval.get("budget_results", {}).items()):
        summary = result.get("summary", {})
        reference_summary = reference_results.get(budget_mode, {}).get("summary", {})
        utility = _round(summary.get("avg_utility"))
        reference_utility = _round(reference_summary.get("avg_utility")) if reference_summary else None
        rows[budget_mode] = {
            "recommended_method": result.get("recommended_method"),
            "num_runs": int(summary.get("num_runs", 0)),
            "avg_utility": utility,
            "reference_avg_utility": reference_utility,
            "utility_delta_vs_reference": _round(utility - reference_utility) if reference_utility is not None else None,
            "avg_em": _round(summary.get("avg_em")),
            "avg_f1_proxy": _round(summary.get("avg_f1_proxy")),
            "avg_tokens": _round(summary.get("avg_tokens"), 1),
            "avg_latency_ms": _round(summary.get("avg_latency_ms"), 1),
            "workflow_counts": summary.get("workflow_counts", {}),
        }
    return rows


def _budget_quality_is_monotonic(rows: dict[str, dict]) -> bool:
    required = ["low", "medium", "high"]
    if any(mode not in rows for mode in required):
        return False
    return (
        float(rows["low"]["avg_utility"])
        < float(rows["medium"]["avg_utility"])
        < float(rows["high"]["avg_utility"])
    )


def _summarize_rollout_file(path: str) -> dict:
    data = read_json(path)
    runs = list(data.get("runs", []))
    questions = {run.get("question") for run in runs}
    workflow_counts = Counter(str(run.get("workflow_id", "")).upper() for run in runs)
    budget_counts = Counter(str(run.get("metadata", {}).get("budget_mode") or "unspecified") for run in runs)
    manifest = data.get("manifest", {})
    return {
        "path": path,
        "num_runs": len(runs),
        "num_questions": len(questions),
        "workflow_counts": dict(workflow_counts),
        "budget_counts": dict(budget_counts),
        "manifest_budget_mode": manifest.get("settings", {}).get("budget_mode"),
        "n_candidates": manifest.get("settings", {}).get("n_candidates"),
        "workflow_avg_utility": data.get("workflow_avg_utility", {}),
        "has_counterfactual_workflows": len(workflow_counts) >= 2,
    }


def _critic_threshold(module_name: str, default_threshold: float, overrides: dict[str, str]) -> float:
    return float(overrides.get(module_name.upper(), default_threshold))


def _summarize_critic_meta(
    module_name: str,
    *,
    meta_path: str,
    model_path: str | None,
    min_eval_examples: int,
    default_min_spearman: float,
    min_spearman_by_module: dict[str, str],
) -> dict:
    meta = read_json(meta_path)
    evaluation = meta.get("evaluation", {}).get("overall", {})
    settings = meta.get("manifest", {}).get("settings", {})
    count = int(evaluation.get("count", 0))
    spearman = _round(evaluation.get("spearman"))
    pearson = _round(evaluation.get("pearson"))
    threshold = _critic_threshold(module_name, default_min_spearman, min_spearman_by_module)
    ready = count >= int(min_eval_examples) and spearman >= threshold
    return {
        "module": module_name.upper(),
        "model_path": model_path,
        "meta_path": meta_path,
        "target": settings.get("target"),
        "train_counts": meta.get("train_counts", {}),
        "eval_counts": meta.get("eval_counts", {}),
        "eval_metrics": {
            "count": count,
            "mae": _round(evaluation.get("mae")),
            "rmse": _round(evaluation.get("rmse")),
            "pearson": pearson,
            "spearman": spearman,
        },
        "readiness_gate": {
            "min_eval_examples": int(min_eval_examples),
            "min_spearman": threshold,
            "ready_as_offline_reward_model": ready,
        },
    }


def _summarize_direct_critic_eval(path: str, *, tolerance: float, max_token_multiplier: float) -> dict:
    data = read_json(path)
    summary = data.get("summary", {})
    comparisons = []
    ready = True
    for method, stats in sorted(summary.items()):
        if not method.endswith("_critic"):
            continue
        base_method = method.removesuffix("_critic")
        base_stats = summary.get(base_method)
        if not base_stats:
            continue
        utility_gap = _round(float(stats.get("avg_utility", 0.0)) - float(base_stats.get("avg_utility", 0.0)))
        base_tokens = float(base_stats.get("avg_tokens", 0.0) or 0.0)
        critic_tokens = float(stats.get("avg_tokens", 0.0) or 0.0)
        token_multiplier = round(critic_tokens / base_tokens, 4) if base_tokens > 0 else 0.0
        comparison_ready = utility_gap >= -float(tolerance) and token_multiplier <= float(max_token_multiplier)
        ready = ready and comparison_ready
        comparisons.append(
            {
                "base_method": base_method,
                "critic_method": method,
                "base_utility": _round(base_stats.get("avg_utility")),
                "critic_utility": _round(stats.get("avg_utility")),
                "utility_gap": utility_gap,
                "base_tokens": _round(base_tokens, 1),
                "critic_tokens": _round(critic_tokens, 1),
                "token_multiplier": token_multiplier,
                "ready_for_direct_deployment": comparison_ready,
            }
        )
    if not comparisons:
        ready = False
    return {
        "path": path,
        "comparisons": comparisons,
        "readiness_gate": {
            "utility_tolerance": float(tolerance),
            "max_token_multiplier": float(max_token_multiplier),
            "ready_for_direct_critic_deployment": ready,
        },
    }


def _summarize_final_project_report(report: dict) -> dict:
    readiness = report.get("rl_readiness", {})
    macro_stages = report.get("macro_bandit_stages", {})
    micro_stages = report.get("micro_stages", {})
    return {
        "ready_for_offline_rl": bool(readiness.get("ready_for_offline_rl")),
        "ready_for_online_rl": bool(readiness.get("ready_for_online_rl")),
        "recommended_next_stage": readiness.get("recommended_next_stage"),
        "reasons": readiness.get("reasons", []),
        "macro_stage_readiness": {
            budget_mode: stage.get("readiness", {})
            for budget_mode, stage in sorted(macro_stages.items())
        },
        "micro_stage_readiness": {
            budget_mode: stage.get("readiness", {})
            for budget_mode, stage in sorted(micro_stages.items())
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_bundle", required=True)
    ap.add_argument("--final_eval", required=True)
    ap.add_argument("--final_report", required=True)
    ap.add_argument("--reference_eval", default=None)
    ap.add_argument("--macro_rollout", nargs="+", default=[], help="Optional budget=rollout_path entries.")
    ap.add_argument("--critic_model", nargs="+", default=[], help="Optional MODULE=model_path entries.")
    ap.add_argument("--critic_meta", nargs="+", default=[], help="Required for reward-model gates: MODULE=meta_path.")
    ap.add_argument("--direct_critic_eval", nargs="+", default=[], help="Optional budget=eval_path entries.")
    ap.add_argument("--offline_reward_modules", nargs="+", default=["QR", "AG"])
    ap.add_argument("--min_critic_eval_examples", type=int, default=300)
    ap.add_argument("--min_critic_spearman", type=float, default=0.25)
    ap.add_argument("--min_critic_spearman_by_module", nargs="+", default=["QR=0.5", "AG=0.25"])
    ap.add_argument("--direct_critic_tolerance", type=float, default=0.002)
    ap.add_argument("--max_direct_critic_token_multiplier", type=float, default=1.5)
    ap.add_argument("--out", default="outputs/full_system_rl_package.json")
    args = ap.parse_args()

    bundle = read_json(args.policy_bundle)
    final_eval = read_json(args.final_eval)
    final_report = read_json(args.final_report)
    reference_eval = _safe_read_json(args.reference_eval)
    macro_rollouts = _parse_key_values(args.macro_rollout)
    critic_models = {key.upper(): value for key, value in _parse_key_values(args.critic_model).items()}
    critic_metas = {key.upper(): value for key, value in _parse_key_values(args.critic_meta).items()}
    direct_critic_evals = _parse_key_values(args.direct_critic_eval)
    min_spearman_by_module = {key.upper(): value for key, value in _parse_key_values(args.min_critic_spearman_by_module).items()}
    offline_reward_modules = [module.upper() for module in args.offline_reward_modules]

    policy_summary = _summarize_policy_bundle(bundle)
    final_eval_summary = _summarize_final_eval(final_eval, reference_eval)
    final_report_summary = _summarize_final_project_report(final_report)
    macro_coverage = {
        normalize_budget_mode(budget_mode): _summarize_rollout_file(path)
        for budget_mode, path in sorted(macro_rollouts.items())
    }
    critic_summary = {}
    missing_critic_metas = []
    for module in offline_reward_modules:
        meta_path = critic_metas.get(module)
        if not meta_path:
            missing_critic_metas.append(module)
            continue
        critic_summary[module] = _summarize_critic_meta(
            module,
            meta_path=meta_path,
            model_path=critic_models.get(module),
            min_eval_examples=args.min_critic_eval_examples,
            default_min_spearman=args.min_critic_spearman,
            min_spearman_by_module=min_spearman_by_module,
        )
    direct_critic_summary = {
        normalize_budget_mode(budget_mode): _summarize_direct_critic_eval(
            path,
            tolerance=args.direct_critic_tolerance,
            max_token_multiplier=args.max_direct_critic_token_multiplier,
        )
        for budget_mode, path in sorted(direct_critic_evals.items())
    }

    adaptive_macro_all_budgets = all(row["is_adaptive"] for row in policy_summary.values())
    medium_delta = final_eval_summary.get("medium", {}).get("utility_delta_vs_reference")
    high_delta = final_eval_summary.get("high", {}).get("utility_delta_vs_reference")
    reference_policy_beaten = (
        medium_delta is None
        or high_delta is None
        or (float(medium_delta) >= 0.0 and float(high_delta) >= 0.0)
    )
    budget_frontier_monotonic = _budget_quality_is_monotonic(final_eval_summary)
    required_macro_budget_modes = set(policy_summary)
    macro_counterfactual_coverage = bool(required_macro_budget_modes) and all(
        budget_mode in macro_coverage
        and macro_coverage[budget_mode].get("has_counterfactual_workflows")
        and macro_coverage[budget_mode].get("num_questions", 0) >= 50
        for budget_mode in required_macro_budget_modes
    )
    micro_reward_ready = (
        not missing_critic_metas
        and bool(critic_summary)
        and all(row["readiness_gate"]["ready_as_offline_reward_model"] for row in critic_summary.values())
    )
    direct_critic_ready = bool(direct_critic_summary) and all(
        row["readiness_gate"]["ready_for_direct_critic_deployment"]
        for row in direct_critic_summary.values()
    )
    macro_online_ready = all(
        stage.get("ready_for_macro_offline_bandit")
        for stage in final_report_summary.get("macro_stage_readiness", {}).values()
    )
    offline_full_system_ready = (
        final_report_summary["ready_for_offline_rl"]
        and adaptive_macro_all_budgets
        and reference_policy_beaten
        and budget_frontier_monotonic
        and macro_counterfactual_coverage
        and micro_reward_ready
    )
    online_full_system_ready = offline_full_system_ready and direct_critic_ready and macro_online_ready

    blockers = []
    if not final_report_summary["ready_for_offline_rl"]:
        blockers.append("Final project report has not opened offline RL.")
    if not adaptive_macro_all_budgets:
        blockers.append("Not every budget mode uses an adaptive macro policy.")
    if not reference_policy_beaten:
        blockers.append("Medium/high policies do not both beat the static reference.")
    if not budget_frontier_monotonic:
        blockers.append("Budget quality frontier is not monotonic: low < medium < high.")
    if not macro_counterfactual_coverage:
        blockers.append("Macro counterfactual rollout coverage is missing or too small for at least one budget mode.")
    if not micro_reward_ready:
        blockers.append("Micro critic reward models are missing or below quality thresholds.")
    online_blockers = []
    if not direct_critic_ready:
        online_blockers.append("Direct critic deployment is not stable enough under the current utility/cost gate.")
    if not macro_online_ready:
        online_blockers.append("Macro bandit holdout regret/best-rate is not strong enough for online updates.")

    package = {
        "manifest": build_experiment_manifest(
            script_name="scripts/build_full_system_rl_package.py",
            qa_path=args.final_eval,
            dataset_name="full_system_rl",
            dataset_split="gate",
            effective_questions=sum(
                int(row.get("num_runs", 0)) for row in final_eval_summary.values()
            ),
            router_model_path=args.policy_bundle,
            settings={
                "policy_bundle": args.policy_bundle,
                "final_eval": args.final_eval,
                "final_report": args.final_report,
                "reference_eval": args.reference_eval,
                "offline_reward_modules": offline_reward_modules,
                "direct_critic_tolerance": args.direct_critic_tolerance,
                "max_direct_critic_token_multiplier": args.max_direct_critic_token_multiplier,
            },
        ),
        "stage": {
            "ready_for_offline_full_system_rl": offline_full_system_ready,
            "ready_for_online_full_system_rl": online_full_system_ready,
            "recommended_next_stage": "offline_full_system_rl_training" if offline_full_system_ready else "fix_offline_rl_blockers",
            "deployment_mode": "offline_reward_model_only" if offline_full_system_ready and not online_full_system_ready else "online_rl_candidate",
            "offline_blockers": blockers,
            "online_blockers": online_blockers if offline_full_system_ready else [],
        },
        "macro_layer": {
            "policy_bundle": args.policy_bundle,
            "budget_policies": policy_summary,
            "counterfactual_rollout_coverage": macro_coverage,
            "rl_objective": "conservative contextual bandit policy improvement over workflow actions",
        },
        "micro_layer": {
            "reward_modules": offline_reward_modules,
            "critic_models": critic_summary,
            "direct_critic_evaluations": direct_critic_summary,
            "rl_objective": "critic-scored candidate action reranking/reward modeling inside selected workflows",
            "online_rerank_default": bool(online_full_system_ready),
        },
        "evaluation": {
            "final_budget_eval": final_eval_summary,
            "final_project_report": final_report_summary,
            "reference_eval": args.reference_eval,
        },
        "rl_contract": {
            "macro_state": ["question features", "budget_mode", "router confidence", "cheap retrieval probe"],
            "macro_action": "select one workflow under the active budget action set",
            "micro_state": ["workflow_id", "module", "question", "candidate action", "retrieved evidence", "cost features"],
            "micro_action": "rerank/select candidate query, answer, or evidence item",
            "reward": "budgeted end utility plus budget-aware process critic signal, with explicit token/retrieval/latency penalties",
            "safety": [
                "offline updates only until direct critic deployment is utility-safe",
                "keep budget-specific cost penalties active",
                "promote a policy only if held-out budget utility is non-regressive against the frozen reference",
            ],
        },
    }
    write_json(args.out, package)

    print("=== Full-system RL gate ===")
    print(f"offline_full_system_rl={offline_full_system_ready}")
    print(f"online_full_system_rl={online_full_system_ready}")
    print(f"deployment_mode={package['stage']['deployment_mode']}")
    if blockers:
        print("offline_blockers:", blockers)
    if online_blockers:
        print("online_blockers:", online_blockers)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
