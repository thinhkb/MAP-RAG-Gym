from __future__ import annotations

import argparse
from statistics import mean
from typing import Any

from map_rag_gym.router.budget import normalize_budget_mode
from map_rag_gym.utils.experiment import build_experiment_manifest
from map_rag_gym.utils.io import read_json, write_json


def _parse_key_values(entries: list[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_entry in entries or []:
        if "=" not in raw_entry:
            raise ValueError(f"Invalid entry '{raw_entry}'. Expected KEY=VALUE.")
        key, value = raw_entry.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Invalid entry '{raw_entry}'. Expected KEY=VALUE.")
        parsed[key] = value
    return parsed


def _parse_budget_float_overrides(entries: list[str] | None, default: float) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, raw_value in _parse_key_values(entries).items():
        values[normalize_budget_mode(key)] = float(raw_value)
    values.setdefault("_default", float(default))
    return values


def _get_threshold(values: dict[str, float], budget_mode: str) -> float:
    return float(values.get(budget_mode, values["_default"]))


def _avg_per_question_utility(result: dict[str, Any]) -> float | None:
    per_question = result.get("per_question", [])
    values = [float(item.get("utility_total", 0.0)) for item in per_question]
    return round(mean(values), 4) if values else None


def _summary_utility(result: dict[str, Any]) -> float:
    summary = result.get("summary", {})
    if "avg_utility" in summary:
        return float(summary["avg_utility"])
    per_question_avg = _avg_per_question_utility(result)
    if per_question_avg is None:
        raise ValueError("Budget result has neither summary.avg_utility nor per_question utility values.")
    return float(per_question_avg)


def _budget_report(
    *,
    budget_mode: str,
    candidate_result: dict[str, Any],
    frozen_result: dict[str, Any],
    min_delta: float,
    tolerance: float,
) -> dict[str, Any]:
    candidate_summary = candidate_result.get("summary", {})
    frozen_summary = frozen_result.get("summary", {})
    candidate_utility = _summary_utility(candidate_result)
    frozen_utility = _summary_utility(frozen_result)
    delta = round(candidate_utility - frozen_utility, 4)
    passed = (delta + tolerance) >= min_delta
    return {
        "budget_mode": budget_mode,
        "passed": passed,
        "candidate_method": candidate_result.get("recommended_method"),
        "frozen_method": frozen_result.get("recommended_method"),
        "candidate_utility": round(candidate_utility, 4),
        "frozen_utility": round(frozen_utility, 4),
        "utility_delta": delta,
        "min_delta": float(min_delta),
        "tolerance": float(tolerance),
        "candidate_tokens": candidate_summary.get("avg_tokens"),
        "frozen_tokens": frozen_summary.get("avg_tokens"),
        "candidate_workflow_counts": candidate_summary.get("workflow_counts", {}),
        "frozen_workflow_counts": frozen_summary.get("workflow_counts", {}),
    }


def _frontier_check(budget_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ordered = [budget for budget in ("low", "medium", "high") if budget in budget_reports]
    utilities = [float(budget_reports[budget]["candidate_utility"]) for budget in ordered]
    passed = all(left <= right for left, right in zip(utilities, utilities[1:]))
    return {
        "required": len(ordered) >= 2,
        "passed": passed,
        "budget_order": ordered,
        "candidate_utilities": {budget: budget_reports[budget]["candidate_utility"] for budget in ordered},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate_eval", required=True)
    ap.add_argument("--frozen_eval", default="outputs/final_budget_policy_test_eval_rl_ready.json")
    ap.add_argument("--budget_modes", nargs="+", default=None)
    ap.add_argument("--min_delta", type=float, default=0.0)
    ap.add_argument("--min_delta_by_budget", nargs="+", default=[])
    ap.add_argument("--tolerance", type=float, default=0.0)
    ap.add_argument("--tolerance_by_budget", nargs="+", default=[])
    ap.add_argument("--require_frontier", action="store_true")
    ap.add_argument("--fail_on_block", action="store_true")
    ap.add_argument("--out", default="outputs/offline_full_system_rl/promotion_check.json")
    args = ap.parse_args()

    candidate_eval = read_json(args.candidate_eval)
    frozen_eval = read_json(args.frozen_eval)
    candidate_results = candidate_eval.get("budget_results", {})
    frozen_results = frozen_eval.get("budget_results", {})
    common_budgets = sorted(set(candidate_results) & set(frozen_results))
    requested_budgets = [normalize_budget_mode(mode) for mode in (args.budget_modes or common_budgets)]
    if not requested_budgets:
        raise ValueError("No common budget results were found to compare.")

    min_delta_by_budget = _parse_budget_float_overrides(args.min_delta_by_budget, args.min_delta)
    tolerance_by_budget = _parse_budget_float_overrides(args.tolerance_by_budget, args.tolerance)

    budget_reports = {}
    for budget_mode in requested_budgets:
        if budget_mode not in candidate_results:
            raise ValueError(f"Candidate eval has no budget result for '{budget_mode}'.")
        if budget_mode not in frozen_results:
            raise ValueError(f"Frozen eval has no budget result for '{budget_mode}'.")
        budget_reports[budget_mode] = _budget_report(
            budget_mode=budget_mode,
            candidate_result=candidate_results[budget_mode],
            frozen_result=frozen_results[budget_mode],
            min_delta=_get_threshold(min_delta_by_budget, budget_mode),
            tolerance=_get_threshold(tolerance_by_budget, budget_mode),
        )

    frontier = _frontier_check(budget_reports)
    frontier_passed = frontier["passed"] if args.require_frontier else True
    blocked_budgets = [budget for budget, row in budget_reports.items() if not row["passed"]]
    ready_to_promote = not blocked_budgets and frontier_passed
    payload = {
        "manifest": build_experiment_manifest(
            script_name="scripts/check_offline_rl_promotion.py",
            qa_path=args.candidate_eval,
            router_model_path=args.frozen_eval,
            dataset_name="offline_full_system_rl",
            dataset_split="promotion_gate",
            effective_questions=sum(
                int(candidate_results[budget].get("summary", {}).get("num_runs", 0))
                for budget in requested_budgets
            ),
            settings={
                "candidate_eval": args.candidate_eval,
                "frozen_eval": args.frozen_eval,
                "budget_modes": requested_budgets,
                "min_delta": args.min_delta,
                "min_delta_by_budget": min_delta_by_budget,
                "tolerance": args.tolerance,
                "tolerance_by_budget": tolerance_by_budget,
                "require_frontier": args.require_frontier,
            },
        ),
        "ready_to_promote": ready_to_promote,
        "blocked_budgets": blocked_budgets,
        "budget_reports": budget_reports,
        "frontier_check": frontier,
        "promotion_rule": "Promote only when every requested budget is non-regressive vs the frozen bundle.",
    }
    write_json(args.out, payload)

    print("=== Offline RL promotion gate ===")
    for budget_mode in requested_budgets:
        row = budget_reports[budget_mode]
        status = "PASS" if row["passed"] else "BLOCK"
        print(
            f"{budget_mode}: {status} | candidate={row['candidate_utility']:.4f} | "
            f"frozen={row['frozen_utility']:.4f} | delta={row['utility_delta']:+.4f}"
        )
    if args.require_frontier:
        print(f"frontier: {'PASS' if frontier['passed'] else 'BLOCK'} | {frontier['candidate_utilities']}")
    print(f"ready_to_promote={ready_to_promote}")
    print(f"Saved {args.out}")

    if args.fail_on_block and not ready_to_promote:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
