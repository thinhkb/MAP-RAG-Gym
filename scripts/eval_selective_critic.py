from __future__ import annotations

"""
Evaluate selective critic deployment strategies to reduce token cost
while maintaining utility.

Strategies:
1. AG-only critic: Only apply critic reranking at the AG (answer generation) step
2. Confidence-gated: Only apply critic when router confidence is below a threshold
3. Reduced candidates: Use critic_n_candidates=2 instead of 3
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from map_rag_gym.evaluation.heuristics import compute_budgeted_utility
from map_rag_gym.utils.experiment import build_experiment_manifest, set_global_seed
from map_rag_gym.utils.io import read_json, write_json


def _extract_result_utility(result: dict) -> float:
    """Extract the utility_total from a per-question result."""
    return float(result.get("final_scores", {}).get("utility_total", 0.0))


def _extract_tokens(result: dict) -> float:
    """Extract total tokens from a per-question result."""
    return float(result.get("total_cost", {}).get("tokens", 0.0))


def _has_critic_rerank(result: dict) -> bool:
    """Check if any step in this result was critic-reranked."""
    for step in result.get("steps", []):
        if "critic_reranked" in str(step.get("notes", "")):
            return True
    return False


def _compute_ag_only_savings(per_question: list[dict]) -> dict:
    """
    Simulate AG-only critic: estimate savings from skipping QR critic.
    For each question that had QR critic reranked, estimate the token savings
    as the proportion of QR steps vs total steps.
    """
    qr_critic_count = 0
    ag_critic_count = 0
    for result in per_question:
        for step in result.get("steps", []):
            notes = str(step.get("notes", ""))
            if "critic_reranked" in notes:
                if step.get("module") == "QR":
                    qr_critic_count += 1
                elif step.get("module") == "AG":
                    ag_critic_count += 1
    return {
        "qr_critic_steps": qr_critic_count,
        "ag_critic_steps": ag_critic_count,
        "estimated_qr_token_savings_fraction": round(
            qr_critic_count / max(1, qr_critic_count + ag_critic_count), 4
        ),
    }


def _simulate_confidence_gate(
    per_question: list[dict],
    confidence_threshold: float,
    budget_mode: str,
) -> dict:
    """
    Simulate confidence-gated critic: only apply critic when router confidence
    is below the threshold. Questions with high confidence skip the critic entirely.
    """
    total_utility_no_critic = 0.0
    total_utility_with_critic = 0.0
    total_tokens_no_critic = 0.0
    total_tokens_with_critic = 0.0
    gated_count = 0
    pass_count = 0

    for result in per_question:
        confidence = float(result.get("metadata", {}).get("router_confidence", 0.5))
        utility = _extract_result_utility(result)
        tokens = _extract_tokens(result)

        if confidence >= confidence_threshold:
            # High confidence: skip critic (use base utility)
            gated_count += 1
            total_utility_no_critic += utility
            total_tokens_no_critic += tokens
        else:
            # Low confidence: apply critic
            pass_count += 1
            total_utility_with_critic += utility
            total_tokens_with_critic += tokens

    total = gated_count + pass_count
    return {
        "confidence_threshold": confidence_threshold,
        "gated_count": gated_count,
        "pass_count": pass_count,
        "total_questions": total,
        "gated_fraction": round(gated_count / max(1, total), 4),
        "estimated_token_savings_fraction": round(gated_count / max(1, total) * 0.46, 4),
    }


def _simulate_n_candidates(
    per_question: list[dict],
    original_n: int,
    reduced_n: int,
) -> dict:
    """
    Estimate savings from reducing n_candidates. Token cost for critic scales
    roughly linearly with n_candidates.
    """
    scaling_factor = reduced_n / original_n
    total_critic_tokens = 0.0
    total_base_tokens = 0.0

    for result in per_question:
        tokens = _extract_tokens(result)
        has_critic = _has_critic_rerank(result)
        if has_critic:
            # Estimate: critic overhead is ~46% of total (from the 1.46x multiplier)
            critic_overhead = tokens * (1.0 - 1.0 / 1.46)
            base_tokens = tokens - critic_overhead
            total_critic_tokens += critic_overhead
            total_base_tokens += base_tokens
        else:
            total_base_tokens += tokens

    reduced_critic_tokens = total_critic_tokens * scaling_factor
    original_total = total_base_tokens + total_critic_tokens
    reduced_total = total_base_tokens + reduced_critic_tokens

    return {
        "original_n": original_n,
        "reduced_n": reduced_n,
        "original_avg_tokens": round(original_total / max(1, len(per_question)), 2),
        "reduced_avg_tokens": round(reduced_total / max(1, len(per_question)), 2),
        "token_multiplier_original": round(original_total / max(1, total_base_tokens), 4),
        "token_multiplier_reduced": round(reduced_total / max(1, total_base_tokens), 4),
        "token_savings_fraction": round(1.0 - reduced_total / max(1, original_total), 4),
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate selective critic deployment strategies")
    ap.add_argument(
        "--critic_eval",
        default="outputs/router_eval_large_budget_high_w2w3_det_critic2_val.json",
        help="Path to evaluation with critic enabled",
    )
    ap.add_argument(
        "--base_eval",
        default="outputs/router_eval_large_budget_high_w2w3_det_val.json",
        help="Path to evaluation without critic",
    )
    ap.add_argument("--budget_mode", default="high")
    ap.add_argument("--out", default="outputs/selective_critic_report.json")
    args = ap.parse_args()

    critic_eval = read_json(args.critic_eval)
    base_eval = read_json(args.base_eval)

    # The summary has method-level aggregates at the top
    critic_summary = critic_eval.get("summary", {})
    base_summary = base_eval.get("summary", {})

    # Find base and critic methods from summary keys
    base_method = None
    critic_method = None
    for key in critic_summary:
        if isinstance(critic_summary[key], dict) and "avg_utility" in critic_summary[key]:
            if "critic" in key.lower():
                critic_method = key
            else:
                base_method = key

    if not base_method:
        for key in base_summary:
            if isinstance(base_summary[key], dict) and "avg_utility" in base_summary[key]:
                base_method = key
                break

    print(f"Base method: {base_method}")
    print(f"Critic method: {critic_method}")
    print(f"Budget mode: {args.budget_mode}")

    # Get stats
    base_stats = critic_summary.get(base_method, {}) if base_method else {}
    critic_stats = critic_summary.get(critic_method, {}) if critic_method else {}

    base_utility = float(base_stats.get("avg_utility", 0))
    critic_utility = float(critic_stats.get("avg_utility", 0))
    base_tokens = float(base_stats.get("avg_tokens", 0))
    critic_tokens = float(critic_stats.get("avg_tokens", 0))
    utility_gap = round(critic_utility - base_utility, 4)
    token_multiplier = round(critic_tokens / max(1, base_tokens), 4)

    print(f"\nBaseline: utility={base_utility:.4f}, tokens={base_tokens:.1f}")
    print(f"Critic:   utility={critic_utility:.4f}, tokens={critic_tokens:.1f}")
    print(f"Gap: utility={utility_gap:+.4f}, token_multiplier={token_multiplier:.4f}x")

    # Get per-question data for detailed analysis
    per_question = critic_eval.get("per_question", [])

    report = {
        "manifest": build_experiment_manifest(
            script_name="scripts/eval_selective_critic.py",
            qa_path=args.critic_eval,
            dataset_name="selective_critic",
            dataset_split="validation",
            settings={
                "critic_eval": args.critic_eval,
                "base_eval": args.base_eval,
                "budget_mode": args.budget_mode,
            },
        ),
        "baseline_comparison": {
            "base_method": base_method,
            "critic_method": critic_method,
            "base_utility": base_utility,
            "critic_utility": critic_utility,
            "utility_gap": utility_gap,
            "base_tokens": base_tokens,
            "critic_tokens": critic_tokens,
            "token_multiplier": token_multiplier,
        },
        "strategies": {},
    }

    # Strategy 1: AG-only critic
    if per_question:
        ag_savings = _compute_ag_only_savings(per_question)
        # Estimate new token multiplier: remove QR critic overhead
        qr_fraction = ag_savings["estimated_qr_token_savings_fraction"]
        critic_overhead = critic_tokens - base_tokens
        reduced_overhead = critic_overhead * (1.0 - qr_fraction)
        ag_only_tokens = base_tokens + reduced_overhead
        ag_only_multiplier = round(ag_only_tokens / max(1, base_tokens), 4)
        report["strategies"]["ag_only_critic"] = {
            "description": "Only apply critic at AG step, skip QR critic",
            "estimated_tokens": round(ag_only_tokens, 2),
            "estimated_token_multiplier": ag_only_multiplier,
            "estimated_savings": ag_savings,
            "recommendation": (
                "Expected token multiplier: ~{:.2f}x. ".format(ag_only_multiplier) +
                "QR critic is useful offline (spearman=0.67) but costly online."
            ),
        }

    # Strategy 2: Confidence-gated critic
    for threshold in [0.55, 0.60, 0.65, 0.70, 0.75]:
        gate_result = _simulate_confidence_gate(
            per_question,
            confidence_threshold=threshold,
            budget_mode=args.budget_mode,
        )
        # Estimate: gated questions use base_tokens, passed questions use critic_tokens
        gated_frac = gate_result["gated_fraction"]
        estimated_avg_tokens = base_tokens * gated_frac + critic_tokens * (1 - gated_frac)
        estimated_multiplier = round(estimated_avg_tokens / max(1, base_tokens), 4)
        gate_result["estimated_avg_tokens"] = round(estimated_avg_tokens, 2)
        gate_result["estimated_token_multiplier"] = estimated_multiplier
        report["strategies"][f"confidence_gate_{threshold}"] = {
            "description": f"Only apply critic when router confidence < {threshold}",
            "result": gate_result,
        }

    # Strategy 3: Reduced n_candidates
    for reduced_n in [2, 1]:
        n_result = _simulate_n_candidates(per_question, original_n=3, reduced_n=reduced_n)
        report["strategies"][f"n_candidates_{reduced_n}"] = {
            "description": f"Reduce critic_n_candidates from 3 to {reduced_n}",
            "result": n_result,
        }

    # Combined strategy: AG-only + n_candidates=2 + confidence_gate=0.65
    if per_question:
        # Combined estimate: 
        # - Skip QR critic (save ~50% of critic overhead)
        # - n_candidates=2 (save ~33% of remaining critic overhead)
        # - Confidence gate at 0.65 (skip critic for ~40% of questions)
        combined_overhead = critic_overhead * 0.5 * (2/3) * 0.6  # Rough estimate
        combined_tokens = base_tokens + combined_overhead
        combined_multiplier = round(combined_tokens / max(1, base_tokens), 4)
        report["strategies"]["combined_ag_n2_gate065"] = {
            "description": "Combined: AG-only + n_candidates=2 + confidence_gate=0.65",
            "estimated_avg_tokens": round(combined_tokens, 2),
            "estimated_token_multiplier": combined_multiplier,
            "estimated_savings_fraction": round(1.0 - combined_tokens / max(1, critic_tokens), 4),
        }

    # Overall recommendation
    report["recommendation"] = {
        "immediate": "Continue using critic as offline reward model only (current deployment mode).",
        "next_step": (
            "When high-bandit improves (regret < 0.04), retry online critic with combined strategy: "
            "AG-only + n_candidates=2 + confidence_gate=0.65. "
            "Expected token multiplier: ~{:.2f}x instead of {:.2f}x.".format(
                report["strategies"].get("combined_ag_n2_gate065", {}).get("estimated_token_multiplier", 1.0),
                token_multiplier,
            )
        ),
        "deployment_gate": {
            "max_token_multiplier": 1.25,
            "min_utility_delta": -0.001,
            "description": "Allow online critic only if tokens < 1.25x and utility loss < 0.001",
        },
    }

    write_json(args.out, report)
    print(f"\nSaved report -> {args.out}")

    # Print summary
    print("\n=== Selective Critic Strategies ===")
    for name, strategy in report["strategies"].items():
        desc = strategy.get("description", "")
        mult = strategy.get("estimated_token_multiplier") or strategy.get("result", {}).get("estimated_token_multiplier")
        print(f"  {name}: {desc}")
        if mult:
            print(f"    estimated_token_multiplier: {mult:.4f}x")
        print()


if __name__ == "__main__":
    main()
