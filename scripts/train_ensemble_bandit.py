from __future__ import annotations

"""
Ensemble bandit: combine Ridge and GBT predictions to reduce regret.

Strategy:
- Train both Ridge and GBT on the same data
- Combine predictions with a learned blending weight
- Evaluate on holdout to check if ensemble reduces regret below Ridge/GBT alone
"""

import argparse
import random
from collections import Counter, defaultdict
from statistics import mean

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from map_rag_gym.evaluation.heuristics import compute_budgeted_utility
from map_rag_gym.router.bandit import BanditRouter
from map_rag_gym.router.budget import normalize_budget_mode
from map_rag_gym.router.learned import LearnedRouter
from map_rag_gym.retrieval.bm25 import LocalBM25Retriever
from map_rag_gym.utils.experiment import build_experiment_manifest, set_global_seed
from map_rag_gym.utils.io import read_json, write_json

# Import GBT variant from previous script
from improve_high_bandit import GBTBanditRouter


class EnsembleBanditRouter:
    """Combines Ridge and GBT bandit routers with a blend weight."""

    def __init__(self, ridge: BanditRouter, gbt: GBTBanditRouter, alpha: float = 0.5):
        self.ridge = ridge
        self.gbt = gbt
        self.alpha = alpha  # weight for GBT; (1-alpha) for Ridge

    def predict_with_scores(self, question, budget_mode=None, candidate_workflows=None):
        _, _, ridge_scores = self.ridge.predict_with_scores(
            question, budget_mode=budget_mode, candidate_workflows=candidate_workflows,
        )
        _, _, gbt_scores = self.gbt.predict_with_scores(
            question, budget_mode=budget_mode, candidate_workflows=candidate_workflows,
        )
        # Blend scores
        all_wf = set(ridge_scores) | set(gbt_scores)
        blended = {}
        for wf in all_wf:
            r_score = ridge_scores.get(wf, 0.0)
            g_score = gbt_scores.get(wf, 0.0)
            blended[wf] = (1 - self.alpha) * r_score + self.alpha * g_score

        ranked = sorted(blended.items(), key=lambda x: x[1], reverse=True)
        best_wf = ranked[0][0]
        best_score = ranked[0][1]
        import math
        second = ranked[1][1] if len(ranked) > 1 else best_score
        confidence = 1.0 / (1.0 + math.exp(-4.0 * max(0.0, best_score - second)))
        return best_wf, float(confidence), blended

    def save(self, path):
        import joblib
        joblib.dump({
            "type": "ensemble",
            "alpha": self.alpha,
        }, path)
        self.ridge.save(path.replace(".joblib", "_ridge.joblib"))
        self.gbt.save(path.replace(".joblib", "_gbt.joblib"))


def _group_runs_by_question(runs):
    grouped = defaultdict(dict)
    for run in runs:
        grouped[run["question"]][run["workflow_id"]] = run
    return grouped


def _split_questions(grouped_runs, holdout_ratio, seed):
    questions = list(grouped_runs)
    random.Random(seed).shuffle(questions)
    holdout_count = max(1, int(len(questions) * holdout_ratio))
    return set(questions[holdout_count:]), set(questions[:holdout_count])


def _build_rows(grouped_runs, *, budget_mode, allowed_workflows):
    rows, rewards, weights = [], [], []
    for question, runs_by_wf in grouped_runs.items():
        cands = [(wf, compute_budgeted_utility(
            final_scores=run.get("final_scores", {}),
            total_cost=run.get("total_cost", {}),
            process_score=float(run.get("final_scores", {}).get("process_score", 0.0)),
            budget_mode=budget_mode,
        )) for wf, run in runs_by_wf.items() if wf in allowed_workflows]
        if len(cands) < 2:
            continue
        span = max(r for _, r in cands) - min(r for _, r in cands)
        w = max(0.1, round(span, 4))
        for wf, reward in cands:
            rows.append({"question": question, "workflow_id": wf, "budget_mode": budget_mode})
            rewards.append(float(reward))
            weights.append(float(w))
    return rows, rewards, weights


def _evaluate_policy(router, grouped_runs, *, budget_mode, allowed_workflows):
    regrets, chosen_rewards, oracle_rewards = [], [], []
    exact_best = 0
    for question, runs_by_wf in grouped_runs.items():
        cand_rewards = {}
        for wf, run in runs_by_wf.items():
            if wf not in allowed_workflows:
                continue
            cand_rewards[wf] = compute_budgeted_utility(
                final_scores=run.get("final_scores", {}),
                total_cost=run.get("total_cost", {}),
                process_score=float(run.get("final_scores", {}).get("process_score", 0.0)),
                budget_mode=budget_mode,
            )
        if len(cand_rewards) < 2:
            continue
        pred_wf, _, _ = router.predict_with_scores(
            question, budget_mode=budget_mode, candidate_workflows=sorted(cand_rewards),
        )
        oracle_wf, oracle_reward = max(cand_rewards.items(), key=lambda x: x[1])
        chosen = float(cand_rewards[pred_wf])
        regrets.append(float(oracle_reward - chosen))
        chosen_rewards.append(chosen)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/hotpotqa_large_train_rollouts_high_w2w3_det.json")
    ap.add_argument("--output", default="outputs/ensemble_high_bandit.joblib")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--budget_mode", default="high")
    ap.add_argument("--allowed_workflows", nargs="+", default=["W2", "W3"])
    ap.add_argument("--holdout_ratio", type=float, default=0.15)
    ap.add_argument("--base_router_model", default="outputs/router_hotpot_budget_calibrated.joblib")
    ap.add_argument("--probe_corpus", default="data/hotpotqa_large/corpus.json")
    ap.add_argument("--out_report", default="outputs/ensemble_high_bandit_report.json")
    args = ap.parse_args()

    set_global_seed(args.seed)
    budget_mode = normalize_budget_mode(args.budget_mode)
    allowed_workflows = {str(w).upper() for w in args.allowed_workflows}

    data = read_json(args.input)
    grouped = _group_runs_by_question(data.get("runs", []))
    train_q, holdout_q = _split_questions(grouped, args.holdout_ratio, args.seed)
    train_g = {q: grouped[q] for q in train_q}
    holdout_g = {q: grouped[q] for q in holdout_q}

    train_rows, train_rewards, weights = _build_rows(train_g, budget_mode=budget_mode, allowed_workflows=allowed_workflows)

    base_router = None
    if args.base_router_model:
        base_router = LearnedRouter(random_state=args.seed)
        base_router.load(args.base_router_model)

    probe = None
    if args.probe_corpus:
        probe = LocalBM25Retriever(args.probe_corpus)

    # Train Ridge models with various alphas
    ridge_configs = [0.1, 0.5, 1.0, 5.0]
    # Train GBT models
    gbt_configs = [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8},
        {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8},
    ]
    # Ensemble blend weights
    blend_weights = [0.3, 0.4, 0.5, 0.6, 0.7]

    all_results = []
    best_result = None
    best_score = -1.0
    best_router = None

    print(f"Training ensemble configs for {budget_mode} budget...")
    print(f"Train: {len(train_q)} questions, Holdout: {len(holdout_q)} questions\n")

    # First, train all individual models
    ridge_routers = {}
    for alpha in ridge_configs:
        r = BanditRouter(random_state=args.seed, alpha=alpha, default_budget_mode=budget_mode, allowed_workflows=sorted(allowed_workflows))
        if base_router: r.attach_learned_router(base_router)
        if probe: r.attach_probe_retriever(probe)
        r.fit(train_rows, train_rewards, sample_weight=weights)
        ridge_routers[alpha] = r

        eval_result = _evaluate_policy(r, holdout_g, budget_mode=budget_mode, allowed_workflows=allowed_workflows)
        result = {"name": f"ridge_a{alpha}", "type": "ridge", "alpha": alpha, **eval_result}
        all_results.append(result)
        score = eval_result["exact_best_rate"] - 0.5 * eval_result["avg_regret"]
        marker = ""
        if score > best_score:
            best_score, best_result, best_router = score, result, r
            marker = " <-- BEST"
        print(f"  ridge_a{alpha}: regret={eval_result['avg_regret']:.4f} best_rate={eval_result['exact_best_rate']:.4f}{marker}")

    gbt_routers = {}
    for i, cfg in enumerate(gbt_configs):
        g = GBTBanditRouter(random_state=args.seed, default_budget_mode=budget_mode, allowed_workflows=sorted(allowed_workflows), **cfg)
        if base_router: g.attach_learned_router(base_router)
        if probe: g.attach_probe_retriever(probe)
        g.fit(train_rows, train_rewards, sample_weight=weights)
        gbt_routers[i] = g

        eval_result = _evaluate_policy(g, holdout_g, budget_mode=budget_mode, allowed_workflows=allowed_workflows)
        name = f"gbt_n{cfg['n_estimators']}_d{cfg['max_depth']}_lr{str(cfg['learning_rate']).replace('.','')}"
        result = {"name": name, "type": "gbt", **cfg, **eval_result}
        all_results.append(result)
        score = eval_result["exact_best_rate"] - 0.5 * eval_result["avg_regret"]
        marker = ""
        if score > best_score:
            best_score, best_result, best_router = score, result, g
            marker = " <-- BEST"
        print(f"  {name}: regret={eval_result['avg_regret']:.4f} best_rate={eval_result['exact_best_rate']:.4f}{marker}")

    # Now try ensembles
    print("\n  Ensembles:")
    for ridge_alpha, ridge_r in ridge_routers.items():
        for gbt_idx, gbt_r in gbt_routers.items():
            for blend in blend_weights:
                ens = EnsembleBanditRouter(ridge_r, gbt_r, alpha=blend)
                eval_result = _evaluate_policy(ens, holdout_g, budget_mode=budget_mode, allowed_workflows=allowed_workflows)
                gbt_cfg = gbt_configs[gbt_idx]
                name = f"ens_ra{ridge_alpha}_gbt{gbt_idx}_b{blend}"
                result = {"name": name, "type": "ensemble", "ridge_alpha": ridge_alpha, "gbt_idx": gbt_idx, "blend": blend, **eval_result}
                all_results.append(result)
                score = eval_result["exact_best_rate"] - 0.5 * eval_result["avg_regret"]
                marker = ""
                if score > best_score:
                    best_score, best_result, best_router = score, result, ens
                    marker = " <-- BEST"
                if marker:
                    print(f"  {name}: regret={eval_result['avg_regret']:.4f} best_rate={eval_result['exact_best_rate']:.4f}{marker}")

    print(f"\n=== Best Result ===")
    print(f"  Config: {best_result['name']}")
    print(f"  Regret: {best_result['avg_regret']:.4f}")
    print(f"  Best rate: {best_result['exact_best_rate']:.4f}")

    # Sort all by score
    all_results_sorted = sorted(all_results, key=lambda r: r["exact_best_rate"] - 0.5 * r["avg_regret"], reverse=True)

    report = {
        "manifest": build_experiment_manifest(
            script_name="scripts/train_ensemble_bandit.py",
            qa_path=args.input,
            dataset_name="ensemble_bandit",
            dataset_split="train",
            effective_questions=len(train_q),
            seed=args.seed,
            router_model_path=args.output,
            settings={
                "budget_mode": budget_mode,
                "allowed_workflows": sorted(allowed_workflows),
                "holdout_ratio": args.holdout_ratio,
                "num_configs": len(all_results),
            },
        ),
        "best_result": best_result,
        "top_10": all_results_sorted[:10],
        "all_results_count": len(all_results),
    }
    write_json(args.out_report, report)

    # Save best model (if it's a basic BanditRouter or GBTBanditRouter)
    if hasattr(best_router, "save"):
        best_router.save(args.output)
        print(f"Saved -> {args.output}")

    # Export as CSV too
    import csv
    import os
    csv_path = "outputs/metrics/ensemble_high_bandit_configs.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        fields = ["name", "type", "avg_regret", "exact_best_rate", "avg_policy_utility", "avg_oracle_utility"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results_sorted)
    print(f"Saved CSV -> {csv_path}")
    print(f"Saved report -> {args.out_report}")


if __name__ == "__main__":
    main()
