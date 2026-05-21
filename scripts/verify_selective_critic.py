from __future__ import annotations

"""
Verify selective critic deployment on held-out data.

Uses the per-question data from critic eval to simulate selective deployment:
- For each question, if confidence >= gate threshold, use base (no-critic) result
- If confidence < gate threshold, use critic result  
- Measure actual utility gap and effective token multiplier

This gives us a more realistic estimate than the theoretical simulation
in eval_selective_critic.py.
"""

import argparse
import csv
from statistics import mean

from map_rag_gym.utils.experiment import build_experiment_manifest
from map_rag_gym.utils.io import read_json, write_json


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def main():
    ap = argparse.ArgumentParser(description="Verify selective critic on held-out data")
    ap.add_argument(
        "--critic_eval",
        default="outputs/router_eval_large_budget_high_w2w3_det_critic2_val.json",
    )
    ap.add_argument("--budget_mode", default="high")
    ap.add_argument("--out", default="outputs/selective_critic_verification.json")
    ap.add_argument("--out_csv", default="outputs/metrics/metrics_selective_critic_verification.csv")
    args = ap.parse_args()

    data = read_json(args.critic_eval)
    per_question = data.get("per_question", [])
    summary = data.get("summary", {})

    # Find base and critic method names
    base_method = None
    critic_method = None
    for key, val in summary.items():
        if isinstance(val, dict) and "avg_utility" in val:
            if "critic" in key.lower():
                critic_method = key
            else:
                base_method = key

    if not base_method or not critic_method:
        print("Could not identify base/critic methods in summary")
        return

    base_tokens = _safe_float(summary[base_method].get("avg_tokens"))
    critic_tokens = _safe_float(summary[critic_method].get("avg_tokens"))

    print(f"Base method: {base_method}")
    print(f"Critic method: {critic_method}")
    print(f"Base tokens: {base_tokens:.1f}, Critic tokens: {critic_tokens:.1f}")
    print(f"Questions: {len(per_question)}\n")

    # Confidence gate thresholds to try
    gate_thresholds = [0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.0]

    results = []
    csv_rows = []

    for gate in gate_thresholds:
        utilities = []
        effective_tokens = []
        used_critic_count = 0
        skipped_critic_count = 0

        per_q_details = []
        for pq in per_question:
            question_results = pq.get("results", {})
            base_result = question_results.get(base_method, {})
            critic_result = question_results.get(critic_method, {})

            confidence = _safe_float(base_result.get("confidence", 0.5))
            base_utility = _safe_float(base_result.get("utility_total"))
            critic_utility = _safe_float(critic_result.get("utility_total"))

            if gate == 0.0:
                # Always use critic
                chosen_utility = critic_utility
                chosen_tokens = critic_tokens  # Rough per-question average
                used_critic_count += 1
            elif gate >= 1.0:
                # Never use critic (pure base)
                chosen_utility = base_utility
                chosen_tokens = base_tokens
                skipped_critic_count += 1
            elif confidence < gate:
                # Low confidence -> apply critic
                chosen_utility = critic_utility
                chosen_tokens = critic_tokens
                used_critic_count += 1
            else:
                # High confidence -> skip critic
                chosen_utility = base_utility
                chosen_tokens = base_tokens
                skipped_critic_count += 1

            utilities.append(chosen_utility)
            effective_tokens.append(chosen_tokens)

        total = used_critic_count + skipped_critic_count
        avg_utility = round(mean(utilities), 4) if utilities else 0.0
        avg_tokens = round(mean(effective_tokens), 2) if effective_tokens else 0.0
        token_multiplier = round(avg_tokens / max(1, base_tokens), 4)

        # Compare to both full-base and full-critic
        base_avg_utility = _safe_float(summary[base_method].get("avg_utility"))
        critic_avg_utility = _safe_float(summary[critic_method].get("avg_utility"))
        utility_vs_base = round(avg_utility - base_avg_utility, 4)
        utility_vs_critic = round(avg_utility - critic_avg_utility, 4)

        result = {
            "gate_threshold": gate,
            "critic_used_count": used_critic_count,
            "critic_skipped_count": skipped_critic_count,
            "critic_usage_rate": round(used_critic_count / max(1, total), 4),
            "avg_utility": avg_utility,
            "avg_tokens": avg_tokens,
            "token_multiplier": token_multiplier,
            "utility_vs_base": utility_vs_base,
            "utility_vs_critic": utility_vs_critic,
        }
        results.append(result)

        csv_rows.append({
            "gate_threshold": gate,
            "critic_used": used_critic_count,
            "critic_skipped": skipped_critic_count,
            "critic_usage_rate": round(used_critic_count / max(1, total), 4),
            "avg_utility": avg_utility,
            "avg_tokens": avg_tokens,
            "token_multiplier": token_multiplier,
            "utility_vs_base": utility_vs_base,
            "utility_vs_critic": utility_vs_critic,
            "meets_token_gate": "yes" if token_multiplier <= 1.25 else "no",
            "meets_utility_gate": "yes" if utility_vs_base >= -0.001 else "no",
            "meets_both_gates": "yes" if token_multiplier <= 1.25 and utility_vs_base >= -0.001 else "no",
        })

        label = f"gate={gate:.2f}"
        if gate == 0.0:
            label = "always_critic"
        elif gate >= 1.0:
            label = "never_critic"
        marker = ""
        if token_multiplier <= 1.25 and utility_vs_base >= -0.001:
            marker = " [PASS]"
        print(f"  {label:20s}: utility={avg_utility:.4f} tokens={avg_tokens:.1f} "
              f"mult={token_multiplier:.4f}x vs_base={utility_vs_base:+.4f} "
              f"critic_use={used_critic_count}/{total}{marker}")

    # Find best gate that passes both gates
    passing = [r for r in results if r["token_multiplier"] <= 1.25 and r["utility_vs_base"] >= -0.001]
    best_passing = None
    best_critic_gate = None
    if passing:
        # Among passing, pick the one with highest utility
        best_passing = max(passing, key=lambda r: r["avg_utility"])
        # Among passing that actually use the critic, pick the most critic-heavy one
        critic_using = [r for r in passing if r["critic_used_count"] > 0]
        if critic_using:
            best_critic_gate = max(critic_using, key=lambda r: r["critic_usage_rate"])
        print(f"\nBest passing gate (overall): {best_passing['gate_threshold']:.2f} "
              f"(utility={best_passing['avg_utility']:.4f}, mult={best_passing['token_multiplier']:.4f}x)")
        if best_critic_gate:
            print(f"Best critic-using gate: {best_critic_gate['gate_threshold']:.2f} "
                  f"(utility={best_critic_gate['avg_utility']:.4f}, mult={best_critic_gate['token_multiplier']:.4f}x, "
                  f"critic_use={best_critic_gate['critic_used_count']}/{best_critic_gate['critic_used_count'] + best_critic_gate['critic_skipped_count']})")
    else:
        print("\nNo gate threshold passes both deployment gates.")

    # Write report
    report = {
        "manifest": build_experiment_manifest(
            script_name="scripts/verify_selective_critic.py",
            qa_path=args.critic_eval,
            dataset_name="selective_critic_verification",
            dataset_split="validation",
            effective_questions=len(per_question),
            settings={
                "critic_eval": args.critic_eval,
                "base_method": base_method,
                "critic_method": critic_method,
                "budget_mode": args.budget_mode,
            },
        ),
        "baseline": {
            "method": base_method,
            "avg_utility": base_avg_utility,
            "avg_tokens": base_tokens,
        },
        "full_critic": {
            "method": critic_method,
            "avg_utility": critic_avg_utility,
            "avg_tokens": critic_tokens,
        },
        "gate_results": results,
        "best_passing_gate": best_passing,
        "best_critic_using_gate": best_critic_gate,
        "deployment_gates": {
            "max_token_multiplier": 1.25,
            "max_utility_loss": -0.001,
        },
    }
    write_json(args.out, report)
    print(f"\nSaved report -> {args.out}")

    # Write CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8-sig") as f:
        fields = list(csv_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved CSV -> {args.out_csv}")


if __name__ == "__main__":
    main()
