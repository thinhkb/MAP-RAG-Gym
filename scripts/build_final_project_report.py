from __future__ import annotations

import argparse

from map_rag_gym.utils.io import read_json, write_json


ADAPTIVE_METHOD_PREFIXES = (
    "rule_based",
    "learned_router",
    "hybrid_router",
    "bandit_router",
    "gated_bandit_router",
    "meta_router",
)


def _is_adaptive_method(method_name: str | None) -> bool:
    name = str(method_name or "").strip()
    if not name:
        return False
    if name.startswith("fixed_"):
        return False
    return any(name == prefix or name.startswith(f"{prefix}_") for prefix in ADAPTIVE_METHOD_PREFIXES)


def _recommend_rl_stage(
    bundle: dict,
    test_eval: dict,
    *,
    reference_test_eval: dict | None = None,
    micro_stages: dict[str, dict] | None = None,
) -> dict:
    budget_policies = bundle.get("budget_policies", {})
    budget_results = test_eval.get("budget_results", {})
    reference_budget_results = (reference_test_eval or {}).get("budget_results", {})
    micro_stages = micro_stages or {}

    low_method = budget_policies.get("low", {}).get("recommended_method")
    medium_method = budget_policies.get("medium", {}).get("recommended_method")
    high_method = budget_policies.get("high", {}).get("recommended_method")

    low_summary = budget_results.get("low", {}).get("summary", {})
    medium_summary = budget_results.get("medium", {}).get("summary", {})
    high_summary = budget_results.get("high", {}).get("summary", {})
    reference_medium = reference_budget_results.get("medium", {}).get("summary", {})
    reference_high = reference_budget_results.get("high", {}).get("summary", {})

    macro_ready = (
        _is_adaptive_method(low_method)
        and _is_adaptive_method(medium_method)
        and _is_adaptive_method(high_method)
    )
    medium_delta = float(medium_summary.get("avg_utility", 0.0)) - float(reference_medium.get("avg_utility", 0.0))
    high_delta = float(high_summary.get("avg_utility", 0.0)) - float(reference_high.get("avg_utility", 0.0))
    reference_ready = not reference_budget_results or (medium_delta >= 0.0 and high_delta >= 0.0)
    micro_ready = any(stage.get("readiness", {}).get("ready_for_micro_validation_gate") for stage in micro_stages.values())
    ready = macro_ready and reference_ready and micro_ready
    reasons = []
    if not macro_ready:
        if not _is_adaptive_method(low_method):
            reasons.append("Low-budget selection is still not using an adaptive routing policy.")
        if not _is_adaptive_method(medium_method):
            reasons.append("Medium-budget selection is still dominated by a static workflow.")
        if not _is_adaptive_method(high_method):
            reasons.append("High-budget selection is still dominated by a static workflow.")
    if reference_budget_results and medium_delta < 0.0:
        reasons.append("Medium-budget adaptive policy is not yet beating the static reference on test.")
    if reference_budget_results and high_delta < 0.0:
        reasons.append("High-budget adaptive policy is not yet beating the static reference on test.")
    if not micro_ready:
        reasons.append("Micro critic has not yet shown a reliable validation-stage win over a strong fixed baseline.")
    if float(medium_summary.get("avg_utility", 0.0)) <= float(low_summary.get("avg_utility", 0.0)):
        reasons.append("Budget tiers are not yet creating clearly separated quality frontiers.")
    if float(high_summary.get("avg_utility", 0.0)) <= float(medium_summary.get("avg_utility", 0.0)):
        reasons.append("High-budget mode is not materially stronger than medium-budget mode yet.")
    if ready:
        reasons = [
            "Adaptive macro routing is active across low, medium, and high budgets.",
            "Adaptive medium/high policies are at or above the static reference on test.",
            "A budget-aware micro critic shows a validation-stage win over the fixed workflow baseline.",
        ]

    recommendation = {
        "ready_for_offline_rl": ready,
        "ready_for_online_rl": False,
        "recommended_next_stage": "offline_full_system_rl" if ready else "stabilize_micro_critic",
        "reasons": reasons,
    }
    if not ready:
        recommendation["recommended_scope"] = (
            "Continue with repeated validation for adaptive macro policies and critic gating. "
            "Do not open a global online RL loop yet."
        )
    else:
        recommendation["recommended_scope"] = (
            "Macro routing is adaptive across budget modes and the micro critic has a validated gain signal. "
            "An offline full-system RL stage is justified next; keep online RL disabled until critic deployment is stable on held-out test runs."
        )
        recommendation["micro_deployment_caution"] = (
            "Use the critic as a training/reward model first. Its direct online deployment is still noisier than its validation-stage signal."
        )
    return recommendation


def _summarize_macro_bandit(meta: dict | None, eval_payload: dict | None) -> dict | None:
    if not meta and not eval_payload:
        return None

    holdout = (meta or {}).get("holdout_policy_eval", {})
    eval_summary = (eval_payload or {}).get("summary", {})
    bandit_summary = eval_summary.get("bandit_router", {})
    bandit_critic_summary = eval_summary.get("bandit_router_critic", {})

    comparison_methods = {
        name: stats
        for name, stats in eval_summary.items()
        if name not in {"bandit_router", "bandit_router_critic"}
    }
    best_online_method = None
    best_online_utility = None
    if comparison_methods:
        best_online_method, best_online_stats = max(
            comparison_methods.items(),
            key=lambda item: float(item[1].get("avg_utility", 0.0)),
        )
        best_online_utility = float(best_online_stats.get("avg_utility", 0.0))

    bandit_utility = float(bandit_summary.get("avg_utility", 0.0))
    bandit_critic_utility = float(bandit_critic_summary.get("avg_utility", 0.0))
    holdout_regret = float(holdout.get("avg_regret", 1.0))
    holdout_best_rate = float(holdout.get("exact_best_rate", 0.0))

    return {
        "offline_holdout": {
            "avg_policy_utility": holdout.get("avg_policy_utility"),
            "avg_oracle_utility": holdout.get("avg_oracle_utility"),
            "avg_regret": holdout.get("avg_regret"),
            "exact_best_rate": holdout.get("exact_best_rate"),
            "num_questions": holdout.get("num_questions"),
        },
        "online_validation": {
            "bandit_router": bandit_summary,
            "bandit_router_critic": bandit_critic_summary if bandit_critic_summary else None,
            "best_non_bandit_method": best_online_method,
            "best_non_bandit_utility": best_online_utility,
            "bandit_gap_to_best": round(bandit_utility - best_online_utility, 4) if best_online_utility is not None else None,
            "bandit_critic_gap_to_best": round(bandit_critic_utility - best_online_utility, 4)
            if best_online_utility is not None and bandit_critic_summary
            else None,
        },
        "readiness": {
            "ready_for_macro_offline_bandit": holdout_regret <= 0.02 and holdout_best_rate >= 0.65,
            "ready_for_macro_policy_replacement": best_online_utility is not None and bandit_utility >= best_online_utility,
            "ready_for_macro_bandit_plus_critic": best_online_utility is not None and bandit_critic_utility >= best_online_utility,
            "recommended_next_stage": (
                "stabilize_online_bandit_eval"
                if holdout_regret <= 0.02 and holdout_best_rate >= 0.65
                else "improve_bandit_context"
            ),
        },
    }


def _parse_macro_bandit_stage_entries(entries: list[str] | None) -> dict[str, tuple[str, str]]:
    parsed: dict[str, tuple[str, str]] = {}
    for raw_entry in entries or []:
        if "=" not in raw_entry or "|" not in raw_entry:
            raise ValueError(
                f"Invalid macro bandit stage '{raw_entry}'. Expected budget=meta_path|eval_path."
            )
        budget_mode, payload = raw_entry.split("=", 1)
        meta_path, eval_path = payload.split("|", 1)
        budget_mode = budget_mode.strip().lower()
        meta_path = meta_path.strip()
        eval_path = eval_path.strip()
        if not budget_mode or not meta_path or not eval_path:
            raise ValueError(
                f"Invalid macro bandit stage '{raw_entry}'. Expected budget=meta_path|eval_path."
            )
        parsed[budget_mode] = (meta_path, eval_path)
    return parsed


def _parse_micro_eval_stage_entries(entries: list[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_entry in entries or []:
        if "=" not in raw_entry:
            raise ValueError(f"Invalid micro eval stage '{raw_entry}'. Expected budget=eval_path.")
        budget_mode, eval_path = raw_entry.split("=", 1)
        budget_mode = budget_mode.strip().lower()
        eval_path = eval_path.strip()
        if not budget_mode or not eval_path:
            raise ValueError(f"Invalid micro eval stage '{raw_entry}'. Expected budget=eval_path.")
        parsed[budget_mode] = eval_path
    return parsed


def _summarize_micro_stage(eval_payload: dict | None) -> dict | None:
    if not eval_payload:
        return None
    summary = eval_payload.get("summary", {})
    adaptive_noncritic = {
        name: stats
        for name, stats in summary.items()
        if _is_adaptive_method(name) and not name.endswith("_critic")
    }
    adaptive_critic = {
        name: stats
        for name, stats in summary.items()
        if _is_adaptive_method(name) and name.endswith("_critic")
    }
    fixed_baselines = {
        name: stats
        for name, stats in summary.items()
        if str(name).startswith("fixed_") and not str(name).endswith("_critic")
    }
    best_adaptive_method, best_adaptive_stats = max(
        adaptive_noncritic.items(),
        key=lambda item: float(item[1].get("avg_utility", 0.0)),
    ) if adaptive_noncritic else (None, {})
    best_adaptive_critic_method, best_adaptive_critic_stats = max(
        adaptive_critic.items(),
        key=lambda item: float(item[1].get("avg_utility", 0.0)),
    ) if adaptive_critic else (None, {})
    best_fixed_method, best_fixed_stats = max(
        fixed_baselines.items(),
        key=lambda item: float(item[1].get("avg_utility", 0.0)),
    ) if fixed_baselines else (None, {})

    best_adaptive_utility = float(best_adaptive_stats.get("avg_utility", 0.0))
    best_adaptive_critic_utility = float(best_adaptive_critic_stats.get("avg_utility", 0.0))
    best_fixed_utility = float(best_fixed_stats.get("avg_utility", 0.0))

    return {
        "best_adaptive_method": best_adaptive_method,
        "best_adaptive_utility": best_adaptive_utility if adaptive_noncritic else None,
        "best_adaptive_critic_method": best_adaptive_critic_method,
        "best_adaptive_critic_utility": best_adaptive_critic_utility if adaptive_critic else None,
        "best_fixed_method": best_fixed_method,
        "best_fixed_utility": best_fixed_utility if fixed_baselines else None,
        "adaptive_critic_gap_to_adaptive": round(best_adaptive_critic_utility - best_adaptive_utility, 4)
        if adaptive_noncritic and adaptive_critic
        else None,
        "adaptive_critic_gap_to_fixed": round(best_adaptive_critic_utility - best_fixed_utility, 4)
        if fixed_baselines and adaptive_critic
        else None,
        "readiness": {
            "ready_for_micro_validation_gate": (
                bool(adaptive_critic)
                and bool(fixed_baselines)
                and best_adaptive_critic_utility >= best_fixed_utility
                and best_adaptive_critic_utility >= best_adaptive_utility - 0.02
            ),
            "recommended_next_stage": (
                "use_critic_as_offline_reward_model"
                if adaptive_critic
                else "train_budget_aware_critic"
            ),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_bundle", required=True)
    ap.add_argument("--test_eval", required=True)
    ap.add_argument("--reference_test_eval", default=None)
    ap.add_argument("--macro_bandit_meta", default=None)
    ap.add_argument("--macro_bandit_eval", default=None)
    ap.add_argument("--macro_bandit_stage", nargs="+", default=[], help="Optional budget=meta_path|eval_path entries.")
    ap.add_argument("--micro_eval_stage", nargs="+", default=[], help="Optional budget=eval_path entries for critic validation stages.")
    ap.add_argument("--out", default="outputs/final_project_report.json")
    args = ap.parse_args()

    bundle = read_json(args.policy_bundle)
    test_eval = read_json(args.test_eval)
    reference_test_eval = read_json(args.reference_test_eval) if args.reference_test_eval else None
    macro_bandit_meta = read_json(args.macro_bandit_meta) if args.macro_bandit_meta else None
    macro_bandit_eval = read_json(args.macro_bandit_eval) if args.macro_bandit_eval else None
    macro_bandit_stages = {}
    for budget_mode, (meta_path, eval_path) in _parse_macro_bandit_stage_entries(args.macro_bandit_stage).items():
        macro_bandit_stages[budget_mode] = _summarize_macro_bandit(read_json(meta_path), read_json(eval_path))
    if not macro_bandit_stages and (macro_bandit_meta or macro_bandit_eval):
        macro_bandit_stages["default"] = _summarize_macro_bandit(macro_bandit_meta, macro_bandit_eval)
    micro_stages = {
        budget_mode: _summarize_micro_stage(read_json(eval_path))
        for budget_mode, eval_path in _parse_micro_eval_stage_entries(args.micro_eval_stage).items()
    }

    budget_results = test_eval.get("budget_results", {})
    final_table = []
    for budget_mode in sorted(budget_results):
        summary = budget_results[budget_mode].get("summary", {})
        final_table.append(
            {
                "budget_mode": budget_mode,
                "recommended_method": budget_results[budget_mode].get("recommended_method"),
                "avg_utility": summary.get("avg_utility"),
                "avg_em": summary.get("avg_em"),
                "avg_f1_proxy": summary.get("avg_f1_proxy"),
                "avg_tokens": summary.get("avg_tokens"),
                "avg_latency_ms": summary.get("avg_latency_ms"),
                "workflow_counts": summary.get("workflow_counts", {}),
            }
        )

    report = {
        "policy_bundle": args.policy_bundle,
        "test_eval": args.test_eval,
        "reference_test_eval": args.reference_test_eval,
        "final_policy_table": final_table,
        "rl_readiness": _recommend_rl_stage(
            bundle,
            test_eval,
            reference_test_eval=reference_test_eval,
            micro_stages=micro_stages,
        ),
        "macro_bandit_stages": macro_bandit_stages,
        "micro_stages": micro_stages,
    }
    write_json(args.out, report)

    print("Final policy table:")
    for row in final_table:
        print(
            f"{row['budget_mode']}: {row['recommended_method']} | utility={row['avg_utility']:.4f} | "
            f"tokens={row['avg_tokens']:.1f} | latency={row['avg_latency_ms']:.1f}"
        )
    readiness = report["rl_readiness"]
    print(
        f"RL readiness: offline={readiness['ready_for_offline_rl']} | "
        f"online={readiness['ready_for_online_rl']} | next={readiness['recommended_next_stage']}"
    )
    for budget_mode, macro_bandit in sorted(report.get("macro_bandit_stages", {}).items()):
        if not macro_bandit:
            continue
        stage = macro_bandit["readiness"]
        print(
            f"Macro bandit stage [{budget_mode}]: "
            f"offline_ready={stage['ready_for_macro_offline_bandit']} | "
            f"replace_ready={stage['ready_for_macro_policy_replacement']} | "
            f"bandit+critic_ready={stage['ready_for_macro_bandit_plus_critic']} | "
            f"next={stage['recommended_next_stage']}"
        )
    for budget_mode, micro_stage in sorted(report.get("micro_stages", {}).items()):
        if not micro_stage:
            continue
        stage = micro_stage["readiness"]
        print(
            f"Micro stage [{budget_mode}]: "
            f"critic={micro_stage.get('best_adaptive_critic_method')} | "
            f"gap_to_fixed={micro_stage.get('adaptive_critic_gap_to_fixed')} | "
            f"gap_to_adaptive={micro_stage.get('adaptive_critic_gap_to_adaptive')} | "
            f"ready={stage['ready_for_micro_validation_gate']} | "
            f"next={stage['recommended_next_stage']}"
        )
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
