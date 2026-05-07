from __future__ import annotations

import argparse
from itertools import product

from map_rag_gym.utils.io import read_json, write_json


ALLOWED_WORKFLOW_SETS = {
    "any": None,
    "non_w1": {"W2", "W3", "W4", "W5", "W6"},
    "reasoned": {"W2", "W4", "W5", "W6"},
    "w2_w6": {"W2", "W6"},
    "w2_only": {"W2"},
    "w6_only": {"W6"},
}


def _confidence(result: dict) -> float:
    return float(result.get("confidence", 0.0))


def _match_condition(
    candidate_workflow: str,
    advisor_result: dict | None,
    advisor_conf_min: float,
) -> bool:
    if advisor_result is None:
        return True
    if str(advisor_result.get("workflow_id")) != candidate_workflow:
        return False
    if "confidence" in advisor_result:
        return _confidence(advisor_result) >= advisor_conf_min
    return True


def _evaluate_policy(entries: list[dict], config: dict) -> dict:
    chosen_utilities = []
    candidate_count = 0
    baseline_count = 0
    workflow_counts: dict[str, int] = {}

    for entry in entries:
        results = entry.get("results", {})
        baseline = results.get(config["baseline_method"])
        candidate = results.get(config["candidate_method"])
        if not baseline or not candidate:
            continue

        choose_candidate = True
        candidate_workflow = str(candidate.get("workflow_id", ""))
        if candidate_workflow == str(baseline.get("workflow_id", "")):
            choose_candidate = False
        if _confidence(candidate) < config["candidate_conf_min"]:
            choose_candidate = False
        allowed_workflows = config["allowed_candidate_workflows"]
        if allowed_workflows is not None and candidate_workflow not in allowed_workflows:
            choose_candidate = False

        advisor_method = config.get("advisor_method")
        advisor_result = results.get(advisor_method) if advisor_method else None
        rule_result = results.get("rule_based") if config.get("require_rule_match") else None
        advisor_ok = _match_condition(candidate_workflow, advisor_result, config["advisor_conf_min"])
        rule_ok = _match_condition(candidate_workflow, rule_result, 0.0)
        mode = config["match_mode"]
        if mode == "advisor":
            choose_candidate = choose_candidate and advisor_ok
        elif mode == "rule":
            choose_candidate = choose_candidate and rule_ok
        elif mode == "advisor_and_rule":
            choose_candidate = choose_candidate and advisor_ok and rule_ok
        elif mode == "advisor_or_rule":
            choose_candidate = choose_candidate and (advisor_ok or rule_ok)

        chosen = candidate if choose_candidate else baseline
        chosen_utilities.append(float(chosen.get("utility_total", 0.0)))
        chosen_workflow = str(chosen.get("workflow_id", ""))
        workflow_counts[chosen_workflow] = workflow_counts.get(chosen_workflow, 0) + 1
        if choose_candidate:
            candidate_count += 1
        else:
            baseline_count += 1

    total = len(chosen_utilities)
    return {
        **config,
        "num_questions": total,
        "avg_utility": round(sum(chosen_utilities) / total, 4) if total else 0.0,
        "candidate_choice_rate": round(candidate_count / total, 4) if total else 0.0,
        "baseline_choice_rate": round(baseline_count / total, 4) if total else 0.0,
        "workflow_counts": workflow_counts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--budget_mode", default=None)
    ap.add_argument("--bandit_router_model", default=None)
    ap.add_argument("--baseline_method", default="fixed_W3")
    ap.add_argument("--candidate_methods", nargs="+", default=["bandit_router", "learned_router", "hybrid_router"])
    ap.add_argument("--candidate_conf_thresholds", nargs="+", type=float, default=[0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
    ap.add_argument("--advisor_conf_thresholds", nargs="+", type=float, default=[0.55, 0.6, 0.65, 0.7])
    ap.add_argument("--allowed_workflow_sets", nargs="+", default=["any", "non_w1", "reasoned", "w2_w6", "w2_only", "w6_only"])
    ap.add_argument("--match_modes", nargs="+", default=["none", "advisor", "rule", "advisor_and_rule", "advisor_or_rule"])
    ap.add_argument("--out", default="outputs/switch_policy_tuning.json")
    args = ap.parse_args()

    data = read_json(args.input)
    entries = list(data.get("per_question", []))
    available_methods = set()
    for entry in entries[:5]:
        available_methods.update(entry.get("results", {}).keys())

    candidate_methods = [method for method in args.candidate_methods if method in available_methods]
    advisor_candidates = [method for method in ["learned_router", "bandit_router", "hybrid_router"] if method in available_methods]
    configs = []

    for candidate_method in candidate_methods:
        advisor_options = [None] + [method for method in advisor_candidates if method != candidate_method]
        for candidate_conf_min, advisor_conf_min, allowed_key, match_mode, advisor_method in product(
            args.candidate_conf_thresholds,
            args.advisor_conf_thresholds,
            args.allowed_workflow_sets,
            args.match_modes,
            advisor_options,
        ):
            if match_mode == "none" and advisor_method is not None:
                continue
            if match_mode == "advisor" and advisor_method is None:
                continue
            if match_mode == "advisor_and_rule" and advisor_method is None:
                continue
            if match_mode == "advisor_or_rule" and advisor_method is None:
                continue
            configs.append(
                {
                    "baseline_method": args.baseline_method,
                    "candidate_method": candidate_method,
                    "candidate_conf_min": float(candidate_conf_min),
                    "advisor_method": advisor_method,
                    "advisor_conf_min": float(advisor_conf_min),
                    "match_mode": match_mode,
                    "require_rule_match": match_mode in {"rule", "advisor_and_rule", "advisor_or_rule"},
                    "allowed_workflow_set": allowed_key,
                    "allowed_candidate_workflows": sorted(ALLOWED_WORKFLOW_SETS[allowed_key]) if ALLOWED_WORKFLOW_SETS[allowed_key] is not None else None,
                }
            )

    scored = [_evaluate_policy(entries, config) for config in configs]
    ranked = sorted(
        scored,
        key=lambda row: (
            row["avg_utility"],
            -row["candidate_choice_rate"],
        ),
        reverse=True,
    )
    best = ranked[0] if ranked else {}
    payload = {
        "input": args.input,
        "budget_mode": args.budget_mode or data.get("settings", {}).get("budget_mode"),
        "baseline_method": args.baseline_method,
        "candidate_methods": candidate_methods,
        "top_candidates": ranked[:50],
        "recommended_policy": best,
    }
    if best and best.get("candidate_method") == "bandit_router" and args.baseline_method == "fixed_W3":
        payload["policy_recommendation"] = {
            "budget_mode": payload["budget_mode"],
            "recommended_method": "gated_bandit_router",
            "constraints": {"max_tokens": None, "max_latency_ms": None, "max_retrieval_calls": None},
            "source_eval_file": args.input,
            "router_settings": {
                "bandit_gate_baseline_workflow": "W3",
                "bandit_gate_min_advantage": 0.0,
                "bandit_gate_min_confidence": float(best.get("candidate_conf_min", 0.0)),
                "bandit_gate_allowed_workflows": list(best.get("allowed_candidate_workflows") or []),
                "bandit_router_model": args.bandit_router_model,
            },
        }
    write_json(args.out, payload)
    print("Recommended policy:", best)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
