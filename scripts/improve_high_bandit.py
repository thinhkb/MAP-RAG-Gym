from __future__ import annotations

"""
Improve the high-budget bandit by trying multiple model configurations
(Ridge with various alpha values, GradientBoosting) and selecting the
best one on the holdout set based on regret and exact_best_rate.

This script wraps the existing train_bandit_router.py and extends the
BanditRouter with GradientBoosting as an alternative to Ridge.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from statistics import mean

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from map_rag_gym.evaluation.heuristics import UTILITY_PROFILES, compute_budgeted_utility
from map_rag_gym.router.bandit import BanditRouter
from map_rag_gym.router.budget import ALLOWED_WORKFLOWS_BY_BUDGET, normalize_budget_mode
from map_rag_gym.router.learned import LearnedRouter
from map_rag_gym.retrieval.bm25 import LocalBM25Retriever
from map_rag_gym.utils.experiment import build_experiment_manifest, set_global_seed
from map_rag_gym.utils.io import read_json, write_json


class GBTBanditRouter(BanditRouter):
    """BanditRouter variant that uses GradientBoosting instead of Ridge."""

    def __init__(
        self,
        random_state: int = 13,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        default_budget_mode: str = "low",
        allowed_workflows: list[str] | None = None,
        preference_margin: float = 0.0,
        preferred_workflows: list[str] | None = None,
    ) -> None:
        super().__init__(
            random_state=random_state,
            alpha=1.0,
            default_budget_mode=default_budget_mode,
            allowed_workflows=allowed_workflows,
            preference_margin=preference_margin,
            preferred_workflows=preferred_workflows,
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample

    def _build_pipeline(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer

        return Pipeline([
            ("features", ColumnTransformer([
                ("text", TfidfVectorizer(ngram_range=(1, 2), max_features=4000), "question"),
                ("cats", OneHotEncoder(handle_unknown="ignore"), ["wh_word", "budget_mode", "rule_workflow", "learned_top_workflow"]),
                (
                    "nums",
                    "passthrough",
                    [
                        "token_len",
                        "comparative_flag",
                        "conjunction_flag",
                        "ambiguity_flag",
                        "temporal_flag",
                        "negation_flag",
                        "superlative_flag",
                        "multi_entity_flag",
                        "entity_density",
                        "estimated_hops",
                        "rule_confidence",
                        "learned_confidence",
                        "learned_margin",
                        "learned_prob_W1",
                        "learned_prob_W2",
                        "learned_prob_W3",
                        "learned_prob_W4",
                        "learned_prob_W5",
                        "learned_prob_W6",
                        "probe_top1_score",
                        "probe_top2_score",
                        "probe_top3_mean_score",
                        "probe_score_gap12",
                        "probe_doc_overlap_mean",
                        "probe_title_overlap_max",
                        "probe_num_docs",
                    ],
                ),
            ])),
            ("reg", GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                random_state=self.random_state,
            )),
        ])


def _group_runs_by_question(runs: list[dict]) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for run in runs:
        grouped[run["question"]][run["workflow_id"]] = run
    return grouped


def _split_questions(grouped_runs, holdout_ratio, seed):
    questions = list(grouped_runs)
    random.Random(seed).shuffle(questions)
    holdout_count = max(1, int(len(questions) * holdout_ratio))
    holdout_questions = set(questions[:holdout_count])
    train_questions = set(questions[holdout_count:])
    return train_questions, holdout_questions


def _build_rows(grouped_runs, *, budget_mode, allowed_workflows):
    rows, rewards, sample_weights = [], [], []
    for question, runs_by_wf in grouped_runs.items():
        candidates = []
        for wf_id, run in runs_by_wf.items():
            if wf_id not in allowed_workflows:
                continue
            reward = compute_budgeted_utility(
                final_scores=run.get("final_scores", {}),
                total_cost=run.get("total_cost", {}),
                process_score=float(run.get("final_scores", {}).get("process_score", 0.0)),
                budget_mode=budget_mode,
            )
            candidates.append((wf_id, reward))
        if len(candidates) < 2:
            continue
        reward_span = max(r for _, r in candidates) - min(r for _, r in candidates)
        question_weight = max(0.1, round(reward_span, 4))
        for wf_id, reward in candidates:
            rows.append({"question": question, "workflow_id": wf_id, "budget_mode": budget_mode})
            rewards.append(float(reward))
            sample_weights.append(float(question_weight))
    return rows, rewards, sample_weights


def _evaluate_policy(router, grouped_runs, *, budget_mode, allowed_workflows):
    regrets, chosen_rewards, oracle_rewards = [], [], []
    exact_best = 0
    for question, runs_by_wf in grouped_runs.items():
        candidate_rewards = {}
        for wf_id, run in runs_by_wf.items():
            if wf_id not in allowed_workflows:
                continue
            candidate_rewards[wf_id] = compute_budgeted_utility(
                final_scores=run.get("final_scores", {}),
                total_cost=run.get("total_cost", {}),
                process_score=float(run.get("final_scores", {}).get("process_score", 0.0)),
                budget_mode=budget_mode,
            )
        if len(candidate_rewards) < 2:
            continue
        pred_wf, _, _ = router.predict_with_scores(
            question, budget_mode=budget_mode,
            candidate_workflows=sorted(candidate_rewards),
        )
        oracle_wf, oracle_reward = max(candidate_rewards.items(), key=lambda x: x[1])
        chosen_reward = float(candidate_rewards[pred_wf])
        regrets.append(float(oracle_reward - chosen_reward))
        chosen_rewards.append(chosen_reward)
        oracle_rewards.append(float(oracle_reward))
        exact_best += int(pred_wf == oracle_wf)

    total = len(oracle_rewards)
    return {
        "num_questions": total,
        "avg_policy_utility": round(mean(chosen_rewards), 4) if chosen_rewards else 0.0,
        "avg_oracle_utility": round(mean(oracle_rewards), 4) if oracle_rewards else 0.0,
        "avg_regret": round(mean(regrets), 4) if regrets else 0.0,
        "exact_best_rate": round(exact_best / total, 4) if total else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description="Improve high-budget bandit with multiple model configs")
    ap.add_argument("--input", default="outputs/hotpotqa_large_train_rollouts_high_w2w3_det.json")
    ap.add_argument("--output", default="outputs/improved_high_bandit.joblib")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--budget_mode", default="high")
    ap.add_argument("--allowed_workflows", nargs="+", default=["W2", "W3"])
    ap.add_argument("--holdout_ratio", type=float, default=0.15)
    ap.add_argument("--base_router_model", default="outputs/router_hotpot_budget_calibrated.joblib")
    ap.add_argument("--probe_corpus", default="data/hotpotqa_large/corpus.json")
    ap.add_argument("--out_report", default="outputs/improve_high_bandit_report.json")
    args = ap.parse_args()

    set_global_seed(args.seed)
    budget_mode = normalize_budget_mode(args.budget_mode)
    allowed_workflows = {str(w).upper() for w in args.allowed_workflows}

    data = read_json(args.input)
    grouped_runs = _group_runs_by_question(data.get("runs", []))
    train_questions, holdout_questions = _split_questions(grouped_runs, args.holdout_ratio, args.seed)
    train_grouped = {q: grouped_runs[q] for q in train_questions}
    holdout_grouped = {q: grouped_runs[q] for q in holdout_questions}

    train_rows, train_rewards, sample_weights = _build_rows(train_grouped, budget_mode=budget_mode, allowed_workflows=allowed_workflows)
    holdout_rows, holdout_rewards, _ = _build_rows(holdout_grouped, budget_mode=budget_mode, allowed_workflows=allowed_workflows)

    # Load shared context features
    base_router = None
    if args.base_router_model:
        base_router = LearnedRouter(random_state=args.seed)
        base_router.load(args.base_router_model)

    probe_retriever = None
    if args.probe_corpus:
        probe_retriever = LocalBM25Retriever(args.probe_corpus)

    configs = [
        # Ridge configurations with various alpha values
        {"name": "ridge_a0.1", "model_type": "ridge", "alpha": 0.1},
        {"name": "ridge_a0.5", "model_type": "ridge", "alpha": 0.5},
        {"name": "ridge_a1.0", "model_type": "ridge", "alpha": 1.0},
        {"name": "ridge_a5.0", "model_type": "ridge", "alpha": 5.0},
        {"name": "ridge_a10.0", "model_type": "ridge", "alpha": 10.0},
        # GradientBoosting configurations
        {"name": "gbt_n100_d3_lr005", "model_type": "gbt", "n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8},
        {"name": "gbt_n200_d4_lr005", "model_type": "gbt", "n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8},
        {"name": "gbt_n200_d3_lr01",  "model_type": "gbt", "n_estimators": 200, "max_depth": 3, "learning_rate": 0.1,  "subsample": 0.8},
        {"name": "gbt_n300_d4_lr003", "model_type": "gbt", "n_estimators": 300, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.8},
        {"name": "gbt_n150_d5_lr005", "model_type": "gbt", "n_estimators": 150, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.7},
    ]

    results = []
    best_result = None
    best_score = -1.0  # Higher is better (exact_best_rate)
    best_router = None

    print(f"Training {len(configs)} configurations for {budget_mode} budget with workflows {sorted(allowed_workflows)}...")
    print(f"Train: {len(train_questions)} questions, Holdout: {len(holdout_questions)} questions")
    print()

    for config in configs:
        config_name = config["name"]
        if config["model_type"] == "ridge":
            router = BanditRouter(
                random_state=args.seed,
                alpha=config["alpha"],
                default_budget_mode=budget_mode,
                allowed_workflows=sorted(allowed_workflows),
            )
        else:  # gbt
            router = GBTBanditRouter(
                random_state=args.seed,
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                learning_rate=config["learning_rate"],
                subsample=config["subsample"],
                default_budget_mode=budget_mode,
                allowed_workflows=sorted(allowed_workflows),
            )

        if base_router:
            router.attach_learned_router(base_router)
        if probe_retriever:
            router.attach_probe_retriever(probe_retriever)

        router.fit(train_rows, train_rewards, sample_weight=sample_weights)

        # Evaluate on holdout
        train_preds = router.predict_row_rewards(train_rows)
        holdout_preds = router.predict_row_rewards(holdout_rows) if holdout_rows else []
        train_mae = round(mean_absolute_error(train_rewards, train_preds), 4)
        train_rmse = round(mean_squared_error(train_rewards, train_preds) ** 0.5, 4)
        holdout_mae = round(mean_absolute_error(holdout_rewards, holdout_preds), 4) if holdout_rows else 0.0
        holdout_rmse = round(mean_squared_error(holdout_rewards, holdout_preds) ** 0.5, 4) if holdout_rows else 0.0

        policy_eval = _evaluate_policy(router, holdout_grouped, budget_mode=budget_mode, allowed_workflows=allowed_workflows)

        result = {
            "config": config,
            "reward_metrics": {
                "train_mae": train_mae, "train_rmse": train_rmse,
                "holdout_mae": holdout_mae, "holdout_rmse": holdout_rmse,
            },
            "holdout_policy_eval": policy_eval,
        }
        results.append(result)

        # Score = weighted combination prioritizing exact_best_rate and low regret
        score = policy_eval["exact_best_rate"] - 0.5 * policy_eval["avg_regret"]

        status = ""
        if score > best_score:
            best_score = score
            best_result = result
            best_router = router
            status = " <-- BEST"

        print(
            f"  {config_name}: regret={policy_eval['avg_regret']:.4f} | "
            f"best_rate={policy_eval['exact_best_rate']:.4f} | "
            f"holdout_mae={holdout_mae:.4f} | score={score:.4f}{status}"
        )

    # Save the best model
    if best_router and best_result:
        best_router.save(args.output)
        print(f"\nBest config: {best_result['config']['name']}")
        print(f"  regret: {best_result['holdout_policy_eval']['avg_regret']:.4f}")
        print(f"  exact_best_rate: {best_result['holdout_policy_eval']['exact_best_rate']:.4f}")
        print(f"Saved -> {args.output}")

        # Save meta
        meta = {
            "manifest": build_experiment_manifest(
                script_name="scripts/improve_high_bandit.py",
                qa_path=args.input,
                dataset_name="high_bandit_improvement",
                dataset_split="train",
                effective_questions=len(train_questions),
                seed=args.seed,
                router_model_path=args.output,
                settings={
                    "budget_mode": budget_mode,
                    "allowed_workflows": sorted(allowed_workflows),
                    "holdout_ratio": args.holdout_ratio,
                    "num_configs_tested": len(configs),
                },
            ),
            "best_config": best_result["config"],
            "best_holdout_policy_eval": best_result["holdout_policy_eval"],
            "best_reward_metrics": best_result["reward_metrics"],
            "all_results": results,
            "holdout_policy_eval": best_result["holdout_policy_eval"],
        }
        write_json(f"{args.output}.meta.json", meta)
        write_json(args.out_report, meta)
        print(f"Saved report -> {args.out_report}")


if __name__ == "__main__":
    main()
