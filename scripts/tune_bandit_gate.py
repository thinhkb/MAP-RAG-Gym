from __future__ import annotations

import argparse
import random
from collections import defaultdict
from statistics import mean

from map_rag_gym.evaluation.heuristics import compute_budgeted_utility
from map_rag_gym.router.bandit import BanditRouter
from map_rag_gym.router.budget import ALLOWED_WORKFLOWS_BY_BUDGET, normalize_budget_mode
from map_rag_gym.retrieval.bm25 import LocalBM25Retriever
from map_rag_gym.utils.io import read_json, write_json


def _group_runs_by_question(runs: list[dict]) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for run in runs:
        grouped[run["question"]][run["workflow_id"]] = run
    return grouped


def _split_questions(grouped_runs: dict[str, dict[str, dict]], holdout_ratio: float, seed: int) -> set[str]:
    questions = list(grouped_runs)
    random.Random(seed).shuffle(questions)
    holdout_count = max(1, int(len(questions) * holdout_ratio))
    return set(questions[:holdout_count])


def _candidate_rewards(
    runs_by_wf: dict[str, dict],
    *,
    budget_mode: str,
    allowed_workflows: set[str],
) -> dict[str, float]:
    rewards = {}
    for workflow_id, run in runs_by_wf.items():
        if workflow_id not in allowed_workflows:
            continue
        rewards[workflow_id] = compute_budgeted_utility(
            final_scores=run.get("final_scores", {}),
            total_cost=run.get("total_cost", {}),
            process_score=float(run.get("final_scores", {}).get("process_score", 0.0)),
            budget_mode=budget_mode,
        )
    return rewards


def _evaluate_threshold(
    router: BanditRouter,
    grouped_runs: dict[str, dict[str, dict]],
    *,
    budget_mode: str,
    allowed_workflows: set[str],
    baseline_workflow: str,
    threshold: float,
) -> dict:
    regrets = []
    chosen_rewards = []
    oracle_rewards = []
    exact_best = 0
    baseline_fallbacks = 0
    baseline_chosen = 0

    for question, runs_by_wf in grouped_runs.items():
        rewards = _candidate_rewards(runs_by_wf, budget_mode=budget_mode, allowed_workflows=allowed_workflows)
        if len(rewards) < 2:
            continue
        predicted_workflow, _, _, gate_meta = router.predict_with_gate(
            question,
            budget_mode=budget_mode,
            candidate_workflows=sorted(rewards),
            baseline_workflow=baseline_workflow,
            minimum_advantage=threshold,
        )
        oracle_workflow, oracle_reward = max(rewards.items(), key=lambda item: item[1])
        chosen_reward = float(rewards[predicted_workflow])
        chosen_rewards.append(chosen_reward)
        oracle_rewards.append(float(oracle_reward))
        regrets.append(float(oracle_reward - chosen_reward))
        exact_best += int(predicted_workflow == oracle_workflow)
        baseline_chosen += int(predicted_workflow == baseline_workflow)
        baseline_fallbacks += int(bool(gate_meta.get("gate_applied")))

    total = len(oracle_rewards)
    return {
        "threshold": round(float(threshold), 4),
        "num_questions": total,
        "avg_policy_utility": round(mean(chosen_rewards), 4) if chosen_rewards else 0.0,
        "avg_oracle_utility": round(mean(oracle_rewards), 4) if oracle_rewards else 0.0,
        "avg_regret": round(mean(regrets), 4) if regrets else 0.0,
        "exact_best_rate": round(exact_best / total, 4) if total else 0.0,
        "baseline_choice_rate": round(baseline_chosen / total, 4) if total else 0.0,
        "gate_applied_rate": round(baseline_fallbacks / total, 4) if total else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--bandit_router_model", required=True)
    ap.add_argument("--budget_mode", required=True)
    ap.add_argument("--baseline_workflow", default="W3")
    ap.add_argument("--probe_corpus", default=None)
    ap.add_argument("--allowed_workflows", nargs="+", default=None)
    ap.add_argument("--holdout_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1])
    ap.add_argument("--out", default="outputs/bandit_gate_tuning.json")
    args = ap.parse_args()

    budget_mode = normalize_budget_mode(args.budget_mode)
    baseline_workflow = str(args.baseline_workflow).upper()
    allowed_workflows = {
        str(workflow).upper()
        for workflow in (args.allowed_workflows or sorted(ALLOWED_WORKFLOWS_BY_BUDGET[budget_mode]))
    }

    data = read_json(args.input)
    probe_corpus = args.probe_corpus or data.get("manifest", {}).get("paths", {}).get("corpus")
    grouped_runs = _group_runs_by_question(data.get("runs", []))
    holdout_questions = _split_questions(grouped_runs, args.holdout_ratio, args.seed)
    holdout_grouped = {question: grouped_runs[question] for question in holdout_questions}

    router = BanditRouter(random_state=args.seed)
    router.load(args.bandit_router_model)
    if probe_corpus:
        router.attach_probe_retriever(LocalBM25Retriever(probe_corpus))

    threshold_results = [
        _evaluate_threshold(
            router,
            holdout_grouped,
            budget_mode=budget_mode,
            allowed_workflows=allowed_workflows,
            baseline_workflow=baseline_workflow,
            threshold=threshold,
        )
        for threshold in args.thresholds
    ]
    ranked = sorted(
        threshold_results,
        key=lambda row: (
            row["avg_policy_utility"],
            -row["avg_regret"],
            -row["baseline_choice_rate"],
        ),
        reverse=True,
    )
    best = ranked[0] if ranked else {}

    payload = {
        "input": args.input,
        "bandit_router_model": args.bandit_router_model,
        "budget_mode": budget_mode,
        "baseline_workflow": baseline_workflow,
        "allowed_workflows": sorted(allowed_workflows),
        "probe_corpus": probe_corpus,
        "holdout_ratio": args.holdout_ratio,
        "seed": args.seed,
        "threshold_results": threshold_results,
        "recommended_threshold": best.get("threshold"),
        "recommended_metrics": best,
    }
    write_json(args.out, payload)
    print("Recommended threshold:", best.get("threshold"))
    print("Recommended metrics:", best)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
