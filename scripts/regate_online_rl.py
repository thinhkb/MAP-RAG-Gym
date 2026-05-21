from __future__ import annotations

"""
Re-gate the full-system RL package after improvements:
1. Check if improved high bandit meets thresholds for online RL
2. Check if selective critic strategy meets deployment gate
3. Update the RL package gates accordingly
"""

import argparse

from map_rag_gym.utils.experiment import build_experiment_manifest
from map_rag_gym.utils.io import read_json, write_json


def main():
    ap = argparse.ArgumentParser(description="Re-gate online RL readiness after improvements")
    ap.add_argument("--rl_package", default="outputs/full_system_rl_package.json")
    ap.add_argument("--improved_bandit_report", default="outputs/improve_high_bandit_report.json")
    ap.add_argument("--selective_critic_report", default="outputs/selective_critic_report.json")
    ap.add_argument("--improved_bandit_model", default="outputs/improved_high_bandit.joblib")
    ap.add_argument(
        "--max_regret_for_online", type=float, default=0.04,
        help="Maximum average regret to allow online bandit updates.",
    )
    ap.add_argument(
        "--min_best_rate_for_online", type=float, default=0.70,
        help="Minimum exact-best rate to allow online bandit updates.",
    )
    ap.add_argument(
        "--max_critic_token_multiplier", type=float, default=1.25,
        help="Maximum token multiplier for online critic deployment.",
    )
    ap.add_argument(
        "--max_critic_utility_loss", type=float, default=0.001,
        help="Maximum acceptable utility loss for online critic.",
    )
    ap.add_argument("--out", default="outputs/regate_report.json")
    args = ap.parse_args()

    rl_package = read_json(args.rl_package)
    bandit_report = read_json(args.improved_bandit_report) if args.improved_bandit_report else {}
    critic_report = read_json(args.selective_critic_report) if args.selective_critic_report else {}

    # ── 1. Check improved high bandit ──────────────────────────────────
    # Support multiple report formats (single holdout, ensemble, CV)
    best_eval = (
        bandit_report.get("best_holdout_policy_eval")
        or bandit_report.get("best_result")
        or {}
    )
    # CV reports use cv_avg_regret/cv_avg_best_rate
    bandit_regret = float(
        best_eval.get("avg_regret")
        or best_eval.get("cv_avg_regret")
        or 1.0
    )
    bandit_best_rate = float(
        best_eval.get("exact_best_rate")
        or best_eval.get("cv_avg_best_rate")
        or 0.0
    )
    bandit_config = bandit_report.get("best_config") or best_eval or {}
    is_cv = "cv_avg_regret" in best_eval

    bandit_meets_online = (
        bandit_regret <= args.max_regret_for_online
        and bandit_best_rate >= args.min_best_rate_for_online
    )

    bandit_check = {
        "improved_model": args.improved_bandit_model,
        "best_config": bandit_config,
        "avg_regret": bandit_regret,
        "exact_best_rate": bandit_best_rate,
        "max_regret_threshold": args.max_regret_for_online,
        "min_best_rate_threshold": args.min_best_rate_for_online,
        "meets_online_threshold": bandit_meets_online,
    }
    print(f"High bandit check: regret={bandit_regret:.4f} (threshold={args.max_regret_for_online}), "
          f"best_rate={bandit_best_rate:.4f} (threshold={args.min_best_rate_for_online})")
    print(f"  Meets online threshold: {bandit_meets_online}")

    # ── 2. Check selective critic deployment ────────────────────────────
    # Support verification report format (best_critic_using_gate) and
    # estimation report format (strategies.combined_ag_n2_gate065)
    best_gate = critic_report.get("best_critic_using_gate") or critic_report.get("best_passing_gate")
    if best_gate:
        # Verified on held-out data
        combined_token_multiplier = float(best_gate.get("token_multiplier", 2.0))
        utility_gap = float(best_gate.get("utility_vs_base", -1.0))
        critic_strategy_desc = f"confidence_gate={best_gate.get('gate_threshold', '?')}"
    else:
        # Estimated from simulation
        baseline_comparison = critic_report.get("baseline_comparison", {})
        combined_strategy = critic_report.get("strategies", {}).get("combined_ag_n2_gate065", {})
        combined_token_multiplier = float(combined_strategy.get("estimated_token_multiplier", 2.0))
        utility_gap = float(baseline_comparison.get("utility_gap", -1.0))
        critic_strategy_desc = "combined_ag_n2_gate065 (estimated)"

    critic_meets_online = (
        combined_token_multiplier <= args.max_critic_token_multiplier
        and abs(utility_gap) <= args.max_critic_utility_loss
    )

    critic_check = {
        "strategy": critic_strategy_desc,
        "token_multiplier": combined_token_multiplier,
        "utility_gap": utility_gap,
        "max_token_multiplier_threshold": args.max_critic_token_multiplier,
        "max_utility_loss_threshold": args.max_critic_utility_loss,
        "meets_online_threshold": critic_meets_online,
        "verified_on_holdout": best_gate is not None,
    }
    print(f"\nSelective critic check: token_mult={combined_token_multiplier:.4f} "
          f"(threshold={args.max_critic_token_multiplier}), utility_gap={utility_gap:+.4f}")
    print(f"  Meets online threshold: {critic_meets_online}")

    # ── 3. Determine overall online RL readiness ────────────────────────
    online_blockers = []
    if not bandit_meets_online:
        online_blockers.append(
            f"High bandit regret ({bandit_regret:.4f}) exceeds threshold ({args.max_regret_for_online}) "
            f"or best_rate ({bandit_best_rate:.4f}) below threshold ({args.min_best_rate_for_online})."
        )
    if not critic_meets_online:
        online_blockers.append(
            f"Selective critic token multiplier ({combined_token_multiplier:.4f}x) or "
            f"utility gap ({utility_gap:+.4f}) exceeds thresholds."
        )

    ready_for_online = len(online_blockers) == 0

    # ── 4. Update RL package stage ──────────────────────────────────────
    rl_package["stage"]["ready_for_online_full_system_rl"] = ready_for_online
    rl_package["stage"]["online_blockers"] = online_blockers
    if ready_for_online:
        rl_package["stage"]["recommended_next_stage"] = "online_full_system_rl_with_selective_critic"
        rl_package["stage"]["deployment_mode"] = "selective_critic_online"
    else:
        rl_package["stage"]["recommended_next_stage"] = "continue_offline_improvements"
        rl_package["stage"]["deployment_mode"] = "offline_reward_model_only"

    # Update high budget to reference improved bandit if it meets threshold
    if bandit_meets_online and "high" in rl_package.get("macro_layer", {}).get("budget_policies", {}):
        rl_package["macro_layer"]["budget_policies"]["high"]["router_settings"]["bandit_router_model"] = args.improved_bandit_model
        rl_package["macro_layer"]["budget_policies"]["high"]["improved_bandit"] = {
            "config": bandit_config,
            "regret": bandit_regret,
            "exact_best_rate": bandit_best_rate,
        }

    write_json(args.rl_package, rl_package)
    print(f"\n[OK] Updated RL package -> {args.rl_package}")

    # ── 5. Write regate report ──────────────────────────────────────────
    report = {
        "manifest": build_experiment_manifest(
            script_name="scripts/regate_online_rl.py",
            qa_path=args.rl_package,
            dataset_name="regate",
            dataset_split="gate",
            settings={
                "max_regret_for_online": args.max_regret_for_online,
                "min_best_rate_for_online": args.min_best_rate_for_online,
                "max_critic_token_multiplier": args.max_critic_token_multiplier,
                "max_critic_utility_loss": args.max_critic_utility_loss,
            },
        ),
        "ready_for_online_rl": ready_for_online,
        "online_blockers": online_blockers,
        "bandit_check": bandit_check,
        "critic_check": critic_check,
        "deployment_recommendation": (
            "Online RL with selective critic is ready." if ready_for_online
            else "Continue offline improvements. " + " | ".join(online_blockers)
        ),
    }
    write_json(args.out, report)
    print(f"[OK] Saved regate report -> {args.out}")

    print(f"\n=== Online RL Gate ===")
    print(f"  ready_for_online_rl: {ready_for_online}")
    if online_blockers:
        print("  Blockers:")
        for blocker in online_blockers:
            print(f"    - {blocker}")


if __name__ == "__main__":
    main()
