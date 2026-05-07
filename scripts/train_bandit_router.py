from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from statistics import mean

from sklearn.metrics import mean_absolute_error, mean_squared_error

from map_rag_gym.evaluation.heuristics import UTILITY_PROFILES, compute_budgeted_utility
from map_rag_gym.router.bandit import BanditRouter
from map_rag_gym.router.budget import ALLOWED_WORKFLOWS_BY_BUDGET, normalize_budget_mode
from map_rag_gym.router.learned import LearnedRouter
from map_rag_gym.retrieval.bm25 import LocalBM25Retriever
from map_rag_gym.utils.experiment import build_experiment_manifest, set_global_seed
from map_rag_gym.utils.io import read_json, write_json


def _group_runs_by_question(runs: list[dict]) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for run in runs:
        grouped[run["question"]][run["workflow_id"]] = run
    return grouped


def _split_questions(grouped_runs: dict[str, dict[str, dict]], holdout_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    questions = list(grouped_runs)
    random.Random(seed).shuffle(questions)
    holdout_count = max(1, int(len(questions) * holdout_ratio))
    holdout_questions = set(questions[:holdout_count])
    train_questions = set(questions[holdout_count:])
    return train_questions, holdout_questions


def _build_rows(
    grouped_runs: dict[str, dict[str, dict]],
    *,
    budget_mode: str,
    allowed_workflows: set[str],
) -> tuple[list[dict], list[float], list[float], list[dict]]:
    rows: list[dict] = []
    rewards: list[float] = []
    sample_weights: list[float] = []
    diagnostics: list[dict] = []

    for question, runs_by_wf in grouped_runs.items():
        candidates = []
        for workflow_id, run in runs_by_wf.items():
            if workflow_id not in allowed_workflows:
                continue
            reward = compute_budgeted_utility(
                final_scores=run.get("final_scores", {}),
                total_cost=run.get("total_cost", {}),
                process_score=float(run.get("final_scores", {}).get("process_score", 0.0)),
                budget_mode=budget_mode,
            )
            candidates.append((workflow_id, reward, run))
        if len(candidates) < 2:
            continue

        reward_span = max(reward for _, reward, _ in candidates) - min(reward for _, reward, _ in candidates)
        question_weight = max(0.1, round(reward_span, 4))
        ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
        diagnostics.append(
            {
                "question": question,
                "best_workflow": ranked[0][0],
                "best_reward": round(ranked[0][1], 4),
                "runner_up_workflow": ranked[1][0],
                "runner_up_reward": round(ranked[1][1], 4),
                "reward_span": round(reward_span, 4),
                "reward_by_workflow": {workflow_id: round(reward, 4) for workflow_id, reward, _ in ranked},
            }
        )

        for workflow_id, reward, _ in candidates:
            rows.append(
                {
                    "question": question,
                    "workflow_id": workflow_id,
                    "budget_mode": budget_mode,
                }
            )
            rewards.append(float(reward))
            sample_weights.append(float(question_weight))

    return rows, rewards, sample_weights, diagnostics


def _evaluate_policy(
    router: BanditRouter,
    grouped_runs: dict[str, dict[str, dict]],
    *,
    budget_mode: str,
    allowed_workflows: set[str],
) -> dict:
    regrets = []
    chosen_rewards = []
    oracle_rewards = []
    exact_best = 0
    preview = []

    for question, runs_by_wf in grouped_runs.items():
        candidate_rewards = {}
        for workflow_id, run in runs_by_wf.items():
            if workflow_id not in allowed_workflows:
                continue
            candidate_rewards[workflow_id] = compute_budgeted_utility(
                final_scores=run.get("final_scores", {}),
                total_cost=run.get("total_cost", {}),
                process_score=float(run.get("final_scores", {}).get("process_score", 0.0)),
                budget_mode=budget_mode,
            )
        if len(candidate_rewards) < 2:
            continue

        pred_workflow, confidence, score_map = router.predict_with_scores(
            question,
            budget_mode=budget_mode,
            candidate_workflows=sorted(candidate_rewards),
        )
        oracle_workflow, oracle_reward = max(candidate_rewards.items(), key=lambda item: item[1])
        chosen_reward = float(candidate_rewards[pred_workflow])
        regrets.append(float(oracle_reward - chosen_reward))
        chosen_rewards.append(chosen_reward)
        oracle_rewards.append(float(oracle_reward))
        exact_best += int(pred_workflow == oracle_workflow)
        if len(preview) < 200:
            preview.append(
                {
                    "question": question,
                    "predicted_workflow": pred_workflow,
                    "predicted_confidence": round(confidence, 4),
                    "predicted_scores": {workflow_id: round(score, 4) for workflow_id, score in score_map.items()},
                    "oracle_workflow": oracle_workflow,
                    "oracle_reward": round(float(oracle_reward), 4),
                    "chosen_reward": round(chosen_reward, 4),
                    "regret": round(float(oracle_reward - chosen_reward), 4),
                }
            )

    total = len(oracle_rewards)
    return {
        "num_questions": total,
        "avg_policy_utility": round(mean(chosen_rewards), 4) if chosen_rewards else 0.0,
        "avg_oracle_utility": round(mean(oracle_rewards), 4) if oracle_rewards else 0.0,
        "avg_regret": round(mean(regrets), 4) if regrets else 0.0,
        "exact_best_rate": round(exact_best / total, 4) if total else 0.0,
        "preview": preview,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/batch_rollouts.json")
    ap.add_argument("--output", default="outputs/router_bandit.joblib")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--budget_mode", default="low")
    ap.add_argument("--allowed_workflows", nargs="+", default=None)
    ap.add_argument("--holdout_ratio", type=float, default=0.15)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--preference_margin", type=float, default=0.0)
    ap.add_argument("--preferred_workflows", nargs="+", default=[])
    ap.add_argument("--base_router_model", default=None, help="Optional learned router model used as context features for the bandit.")
    ap.add_argument("--probe_corpus", default=None, help="Optional corpus path for cheap BM25 probe features.")
    args = ap.parse_args()

    set_global_seed(args.seed)
    budget_mode = normalize_budget_mode(args.budget_mode)
    allowed_workflows = {
        str(workflow).upper()
        for workflow in (args.allowed_workflows or sorted(ALLOWED_WORKFLOWS_BY_BUDGET[budget_mode]))
    }
    preferred_workflows = [str(workflow).upper() for workflow in args.preferred_workflows]

    data = read_json(args.input)
    source_manifest = data.get("manifest", {})
    grouped_runs = _group_runs_by_question(data.get("runs", []))
    train_questions, holdout_questions = _split_questions(grouped_runs, args.holdout_ratio, args.seed)
    train_grouped = {question: grouped_runs[question] for question in train_questions}
    holdout_grouped = {question: grouped_runs[question] for question in holdout_questions}

    train_rows, train_rewards, sample_weights, diagnostics = _build_rows(
        train_grouped,
        budget_mode=budget_mode,
        allowed_workflows=allowed_workflows,
    )
    holdout_rows, holdout_rewards, _, holdout_diagnostics = _build_rows(
        holdout_grouped,
        budget_mode=budget_mode,
        allowed_workflows=allowed_workflows,
    )

    if not train_rows:
        raise ValueError("No training rows were generated for the selected budget mode and workflows.")

    router = BanditRouter(
        random_state=args.seed,
        alpha=args.alpha,
        default_budget_mode=budget_mode,
        allowed_workflows=sorted(allowed_workflows),
        preference_margin=args.preference_margin,
        preferred_workflows=preferred_workflows,
    )
    if args.base_router_model:
        base_router = LearnedRouter(random_state=args.seed)
        base_router.load(args.base_router_model)
        router.attach_learned_router(base_router)
    probe_corpus = args.probe_corpus or source_manifest.get("paths", {}).get("corpus")
    if probe_corpus:
        router.attach_probe_retriever(LocalBM25Retriever(probe_corpus))
    router.fit(train_rows, train_rewards, sample_weight=sample_weights)
    router.save(args.output)

    train_preds = router.predict_row_rewards(train_rows)
    holdout_preds = router.predict_row_rewards(holdout_rows) if holdout_rows else []
    reward_metrics = {
        "train_mae": round(mean_absolute_error(train_rewards, train_preds), 4),
        "train_rmse": round(mean_squared_error(train_rewards, train_preds) ** 0.5, 4),
        "holdout_mae": round(mean_absolute_error(holdout_rewards, holdout_preds), 4) if holdout_rows else 0.0,
        "holdout_rmse": round(mean_squared_error(holdout_rewards, holdout_preds) ** 0.5, 4) if holdout_rows else 0.0,
    }
    holdout_policy = _evaluate_policy(
        router,
        holdout_grouped,
        budget_mode=budget_mode,
        allowed_workflows=allowed_workflows,
    )

    meta = {
        "manifest": build_experiment_manifest(
            script_name="scripts/train_bandit_router.py",
            qa_path=args.input,
            dataset_name=source_manifest.get("dataset", {}).get("name", "bandit_router"),
            dataset_split=source_manifest.get("dataset", {}).get("split", "custom"),
            limit=source_manifest.get("dataset", {}).get("limit"),
            effective_questions=len(train_questions),
            seed=args.seed,
            prompt_version=source_manifest.get("reproducibility", {}).get("prompt_version", "v1"),
            router_model_path=args.output,
            settings={
                "input_rollout_file": args.input,
                "budget_mode": budget_mode,
                "allowed_workflows": sorted(allowed_workflows),
                "holdout_ratio": args.holdout_ratio,
                "alpha": args.alpha,
                "preference_margin": args.preference_margin,
                "preferred_workflows": preferred_workflows,
                "base_router_model": args.base_router_model,
                "probe_corpus": probe_corpus,
                "utility_profile": UTILITY_PROFILES[budget_mode],
            },
        ),
        "source_rollout_manifest": source_manifest,
        "num_questions_total": len(grouped_runs),
        "num_questions_train": len(train_questions),
        "num_questions_holdout": len(holdout_questions),
        "num_train_rows": len(train_rows),
        "num_holdout_rows": len(holdout_rows),
        "allowed_workflows": sorted(allowed_workflows),
        "budget_mode": budget_mode,
        "base_router_model": args.base_router_model,
        "probe_corpus": probe_corpus,
        "row_counts_by_workflow": dict(Counter(row["workflow_id"] for row in train_rows)),
        "reward_metrics": reward_metrics,
        "holdout_policy_eval": holdout_policy,
        "train_diagnostics_preview": diagnostics[:200],
        "holdout_diagnostics_preview": holdout_diagnostics[:200],
    }
    write_json(f"{args.output}.meta.json", meta)

    print("Bandit training rows by workflow:", dict(Counter(row["workflow_id"] for row in train_rows)))
    print("Reward metrics:", reward_metrics)
    print("Holdout policy eval:", {
        key: value
        for key, value in holdout_policy.items()
        if key != "preview"
    })
    print(f"Saved {args.output}")
    print(f"Saved {args.output}.meta.json")


if __name__ == "__main__":
    main()
