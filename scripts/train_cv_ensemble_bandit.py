from __future__ import annotations

"""
K-fold cross-validated ensemble bandit to:
1. Get a more stable regret estimate (reduce variance from single holdout split)
2. Train on more data per fold (only 15% held out at a time)
3. Find the optimal ensemble config with less overfitting risk

Also tries:
- Wider range of blend weights
- Additional GBT configs with more careful regularization
- Per-question regret analysis to find error patterns
"""

import argparse
import csv
import math
import os
import random
from collections import defaultdict
from statistics import mean, stdev

import numpy as np

from map_rag_gym.evaluation.heuristics import compute_budgeted_utility
from map_rag_gym.router.bandit import BanditRouter
from map_rag_gym.router.budget import normalize_budget_mode
from map_rag_gym.router.learned import LearnedRouter
from map_rag_gym.retrieval.bm25 import LocalBM25Retriever
from map_rag_gym.utils.experiment import build_experiment_manifest, set_global_seed
from map_rag_gym.utils.io import read_json, write_json

from improve_high_bandit import GBTBanditRouter


class EnsembleBanditRouter:
    """Combines Ridge and GBT bandit routers with a blend weight."""

    def __init__(self, ridge: BanditRouter, gbt: GBTBanditRouter, alpha: float = 0.5):
        self.ridge = ridge
        self.gbt = gbt
        self.alpha = alpha

    def predict_with_scores(self, question, budget_mode=None, candidate_workflows=None):
        _, _, ridge_scores = self.ridge.predict_with_scores(
            question, budget_mode=budget_mode, candidate_workflows=candidate_workflows,
        )
        _, _, gbt_scores = self.gbt.predict_with_scores(
            question, budget_mode=budget_mode, candidate_workflows=candidate_workflows,
        )
        all_wf = set(ridge_scores) | set(gbt_scores)
        blended = {}
        for wf in all_wf:
            blended[wf] = (1 - self.alpha) * ridge_scores.get(wf, 0.0) + self.alpha * gbt_scores.get(wf, 0.0)
        ranked = sorted(blended.items(), key=lambda x: x[1], reverse=True)
        best_wf = ranked[0][0]
        second = ranked[1][1] if len(ranked) > 1 else ranked[0][1]
        confidence = 1.0 / (1.0 + math.exp(-4.0 * max(0.0, ranked[0][1] - second)))
        return best_wf, float(confidence), blended

    def save(self, path):
        import joblib
        joblib.dump({"type": "ensemble", "alpha": self.alpha}, path)
        self.ridge.save(path.replace(".joblib", "_ridge.joblib"))
        self.gbt.save(path.replace(".joblib", "_gbt.joblib"))


def _group_runs_by_question(runs):
    grouped = defaultdict(dict)
    for run in runs:
        grouped[run["question"]][run["workflow_id"]] = run
    return grouped


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
    per_q = []
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
        pred_wf, conf, _ = router.predict_with_scores(
            question, budget_mode=budget_mode, candidate_workflows=sorted(cand_rewards),
        )
        oracle_wf, oracle_reward = max(cand_rewards.items(), key=lambda x: x[1])
        chosen = float(cand_rewards[pred_wf])
        regret = float(oracle_reward - chosen)
        regrets.append(regret)
        chosen_rewards.append(chosen)
        oracle_rewards.append(float(oracle_reward))
        exact_best += int(pred_wf == oracle_wf)
        per_q.append({
            "question": question[:100],
            "pred_wf": pred_wf,
            "oracle_wf": oracle_wf,
            "confidence": round(conf, 4),
            "regret": round(regret, 4),
            "reward_span": round(max(cand_rewards.values()) - min(cand_rewards.values()), 4),
        })

    total = len(oracle_rewards)
    return {
        "num_questions": total,
        "avg_policy_utility": round(mean(chosen_rewards), 4) if chosen_rewards else 0.0,
        "avg_oracle_utility": round(mean(oracle_rewards), 4) if oracle_rewards else 0.0,
        "avg_regret": round(mean(regrets), 4) if regrets else 0.0,
        "exact_best_rate": round(exact_best / total, 4) if total else 0.0,
    }, per_q


def _kfold_split(questions, k, seed):
    """Split questions into k folds."""
    q_list = list(questions)
    random.Random(seed).shuffle(q_list)
    folds = [[] for _ in range(k)]
    for i, q in enumerate(q_list):
        folds[i % k].append(q)
    return folds


def _train_and_eval_config(config, folds, grouped, *, budget_mode, allowed_workflows,
                           base_router, probe, seed):
    """Train a config on k-1 folds and evaluate on the held-out fold. Return per-fold results."""
    fold_results = []
    for fold_idx in range(len(folds)):
        holdout_q = set(folds[fold_idx])
        train_q = set()
        for j in range(len(folds)):
            if j != fold_idx:
                train_q.update(folds[j])

        train_g = {q: grouped[q] for q in train_q if q in grouped}
        holdout_g = {q: grouped[q] for q in holdout_q if q in grouped}

        train_rows, train_rewards, weights = _build_rows(
            train_g, budget_mode=budget_mode, allowed_workflows=allowed_workflows,
        )

        if config["type"] == "ridge":
            router = BanditRouter(
                random_state=seed, alpha=config["alpha"],
                default_budget_mode=budget_mode,
                allowed_workflows=sorted(allowed_workflows),
            )
            if base_router: router.attach_learned_router(base_router)
            if probe: router.attach_probe_retriever(probe)
            router.fit(train_rows, train_rewards, sample_weight=weights)

        elif config["type"] == "gbt":
            router = GBTBanditRouter(
                random_state=seed, default_budget_mode=budget_mode,
                allowed_workflows=sorted(allowed_workflows),
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                learning_rate=config["learning_rate"],
                subsample=config.get("subsample", 0.8),
            )
            if base_router: router.attach_learned_router(base_router)
            if probe: router.attach_probe_retriever(probe)
            router.fit(train_rows, train_rewards, sample_weight=weights)

        elif config["type"] == "ensemble":
            ridge = BanditRouter(
                random_state=seed, alpha=config["ridge_alpha"],
                default_budget_mode=budget_mode,
                allowed_workflows=sorted(allowed_workflows),
            )
            if base_router: ridge.attach_learned_router(base_router)
            if probe: ridge.attach_probe_retriever(probe)
            ridge.fit(train_rows, train_rewards, sample_weight=weights)

            gbt = GBTBanditRouter(
                random_state=seed, default_budget_mode=budget_mode,
                allowed_workflows=sorted(allowed_workflows),
                n_estimators=config["gbt_n_estimators"],
                max_depth=config["gbt_max_depth"],
                learning_rate=config["gbt_learning_rate"],
                subsample=config.get("gbt_subsample", 0.8),
            )
            if base_router: gbt.attach_learned_router(base_router)
            if probe: gbt.attach_probe_retriever(probe)
            gbt.fit(train_rows, train_rewards, sample_weight=weights)

            router = EnsembleBanditRouter(ridge, gbt, alpha=config["blend"])

        eval_result, _ = _evaluate_policy(
            router, holdout_g, budget_mode=budget_mode, allowed_workflows=allowed_workflows,
        )
        fold_results.append(eval_result)

    return fold_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/hotpotqa_large_train_rollouts_high_w2w3_det.json")
    ap.add_argument("--output", default="outputs/cv_ensemble_high_bandit.joblib")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--budget_mode", default="high")
    ap.add_argument("--allowed_workflows", nargs="+", default=["W2", "W3"])
    ap.add_argument("--k_folds", type=int, default=5)
    ap.add_argument("--base_router_model", default="outputs/router_hotpot_budget_calibrated.joblib")
    ap.add_argument("--probe_corpus", default="data/hotpotqa_large/corpus.json")
    ap.add_argument("--out_report", default="outputs/cv_ensemble_report.json")
    args = ap.parse_args()

    set_global_seed(args.seed)
    budget_mode = normalize_budget_mode(args.budget_mode)
    allowed_workflows = {str(w).upper() for w in args.allowed_workflows}

    data = read_json(args.input)
    grouped = _group_runs_by_question(data.get("runs", []))

    # Only keep questions with all workflows
    filtered = {q: wfs for q, wfs in grouped.items()
                if len(set(wfs.keys()) & allowed_workflows) == len(allowed_workflows)}
    print(f"Questions with all workflows: {len(filtered)}/{len(grouped)}")

    folds = _kfold_split(list(filtered.keys()), args.k_folds, args.seed)
    print(f"K-fold: {args.k_folds} folds, sizes: {[len(f) for f in folds]}")

    base_router = None
    if args.base_router_model:
        base_router = LearnedRouter(random_state=args.seed)
        base_router.load(args.base_router_model)

    probe = None
    if args.probe_corpus:
        probe = LocalBM25Retriever(args.probe_corpus)

    # Configs to evaluate
    configs = []
    # Ridge configs
    for alpha in [0.1, 0.5, 1.0, 5.0, 10.0]:
        configs.append({"name": f"ridge_a{alpha}", "type": "ridge", "alpha": alpha})

    # GBT configs - wider search
    for n in [50, 100, 150, 200, 300]:
        for d in [2, 3, 4]:
            for lr in [0.01, 0.03, 0.05, 0.1]:
                configs.append({
                    "name": f"gbt_n{n}_d{d}_lr{str(lr).replace('.', '')}",
                    "type": "gbt",
                    "n_estimators": n, "max_depth": d,
                    "learning_rate": lr, "subsample": 0.8,
                })

    # Top ensemble configs from previous run
    for ra in [1.0, 5.0, 10.0]:
        for gn, gd, glr in [(100, 3, 0.05), (200, 3, 0.1), (150, 4, 0.05)]:
            for blend in [0.4, 0.5, 0.6, 0.7]:
                configs.append({
                    "name": f"ens_ra{ra}_n{gn}_d{gd}_lr{str(glr).replace('.','')}_b{blend}",
                    "type": "ensemble",
                    "ridge_alpha": ra,
                    "gbt_n_estimators": gn, "gbt_max_depth": gd,
                    "gbt_learning_rate": glr, "gbt_subsample": 0.8,
                    "blend": blend,
                })

    print(f"\nTotal configs to evaluate: {len(configs)}")
    print(f"Total train-eval runs: {len(configs) * args.k_folds}\n")

    all_cv_results = []
    best_result = None
    best_score = -1.0

    for i, config in enumerate(configs):
        fold_results = _train_and_eval_config(
            config, folds, filtered,
            budget_mode=budget_mode, allowed_workflows=allowed_workflows,
            base_router=base_router, probe=probe, seed=args.seed,
        )
        regrets = [fr["avg_regret"] for fr in fold_results]
        best_rates = [fr["exact_best_rate"] for fr in fold_results]
        cv_regret = round(mean(regrets), 4)
        cv_best_rate = round(mean(best_rates), 4)
        cv_regret_std = round(stdev(regrets), 4) if len(regrets) > 1 else 0.0
        cv_best_rate_std = round(stdev(best_rates), 4) if len(best_rates) > 1 else 0.0

        result = {
            **config,
            "cv_avg_regret": cv_regret,
            "cv_std_regret": cv_regret_std,
            "cv_avg_best_rate": cv_best_rate,
            "cv_std_best_rate": cv_best_rate_std,
            "per_fold_regret": regrets,
            "per_fold_best_rate": best_rates,
        }
        all_cv_results.append(result)

        # Score: balance regret and best_rate, penalize high variance
        score = cv_best_rate - 0.5 * cv_regret - 0.3 * cv_regret_std
        if score > best_score:
            best_score = score
            best_result = result
            print(f"  [{i+1}/{len(configs)}] {config['name']}: cv_regret={cv_regret:.4f}+/-{cv_regret_std:.4f} "
                  f"cv_best_rate={cv_best_rate:.4f}+/-{cv_best_rate_std:.4f} <-- BEST")
        elif (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(configs)}] progress... best so far: {best_result['name']} "
                  f"regret={best_result['cv_avg_regret']:.4f}")

    # Sort by score
    all_cv_results.sort(key=lambda r: r["cv_avg_best_rate"] - 0.5 * r["cv_avg_regret"] - 0.3 * r["cv_std_regret"], reverse=True)

    print(f"\n=== Top 10 CV Results ===")
    for r in all_cv_results[:10]:
        print(f"  {r['name']:45s}: regret={r['cv_avg_regret']:.4f}+/-{r['cv_std_regret']:.4f} "
              f"best_rate={r['cv_avg_best_rate']:.4f}+/-{r['cv_std_best_rate']:.4f}")

    print(f"\n=== Best Config ===")
    print(f"  {best_result['name']}")
    print(f"  CV regret: {best_result['cv_avg_regret']:.4f} +/- {best_result['cv_std_regret']:.4f}")
    print(f"  CV best_rate: {best_result['cv_avg_best_rate']:.4f} +/- {best_result['cv_std_best_rate']:.4f}")
    print(f"  Per-fold regret: {best_result['per_fold_regret']}")

    # Now train the best config on ALL data
    print(f"\n=== Training best config on all data ===")
    all_rows, all_rewards, all_weights = _build_rows(
        filtered, budget_mode=budget_mode, allowed_workflows=allowed_workflows,
    )

    if best_result["type"] == "ridge":
        final_router = BanditRouter(
            random_state=args.seed, alpha=best_result["alpha"],
            default_budget_mode=budget_mode,
            allowed_workflows=sorted(allowed_workflows),
        )
        if base_router: final_router.attach_learned_router(base_router)
        if probe: final_router.attach_probe_retriever(probe)
        final_router.fit(all_rows, all_rewards, sample_weight=all_weights)

    elif best_result["type"] == "gbt":
        final_router = GBTBanditRouter(
            random_state=args.seed, default_budget_mode=budget_mode,
            allowed_workflows=sorted(allowed_workflows),
            n_estimators=best_result["n_estimators"],
            max_depth=best_result["max_depth"],
            learning_rate=best_result["learning_rate"],
            subsample=best_result.get("subsample", 0.8),
        )
        if base_router: final_router.attach_learned_router(base_router)
        if probe: final_router.attach_probe_retriever(probe)
        final_router.fit(all_rows, all_rewards, sample_weight=all_weights)

    elif best_result["type"] == "ensemble":
        ridge = BanditRouter(
            random_state=args.seed, alpha=best_result["ridge_alpha"],
            default_budget_mode=budget_mode,
            allowed_workflows=sorted(allowed_workflows),
        )
        if base_router: ridge.attach_learned_router(base_router)
        if probe: ridge.attach_probe_retriever(probe)
        ridge.fit(all_rows, all_rewards, sample_weight=all_weights)

        gbt = GBTBanditRouter(
            random_state=args.seed, default_budget_mode=budget_mode,
            allowed_workflows=sorted(allowed_workflows),
            n_estimators=best_result["gbt_n_estimators"],
            max_depth=best_result["gbt_max_depth"],
            learning_rate=best_result["gbt_learning_rate"],
            subsample=best_result.get("gbt_subsample", 0.8),
        )
        if base_router: gbt.attach_learned_router(base_router)
        if probe: gbt.attach_probe_retriever(probe)
        gbt.fit(all_rows, all_rewards, sample_weight=all_weights)

        final_router = EnsembleBanditRouter(ridge, gbt, alpha=best_result["blend"])

    final_router.save(args.output)
    print(f"Saved final model -> {args.output}")

    # Report
    report = {
        "manifest": build_experiment_manifest(
            script_name="scripts/train_cv_ensemble_bandit.py",
            qa_path=args.input,
            dataset_name="cv_ensemble_bandit",
            dataset_split="train",
            effective_questions=len(filtered),
            seed=args.seed,
            router_model_path=args.output,
            settings={
                "budget_mode": budget_mode,
                "allowed_workflows": sorted(allowed_workflows),
                "k_folds": args.k_folds,
                "num_configs": len(configs),
                "total_runs": len(configs) * args.k_folds,
            },
        ),
        "best_result": best_result,
        "top_20": all_cv_results[:20],
        "all_results_count": len(all_cv_results),
        "cv_summary": {
            "best_cv_regret": best_result["cv_avg_regret"],
            "best_cv_regret_std": best_result["cv_std_regret"],
            "best_cv_best_rate": best_result["cv_avg_best_rate"],
            "meets_online_gate": best_result["cv_avg_regret"] <= 0.04,
        },
    }
    write_json(args.out_report, report)
    print(f"Saved report -> {args.out_report}")

    # CSV export
    csv_path = "outputs/metrics/cv_ensemble_bandit_configs.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        fields = ["name", "type", "cv_avg_regret", "cv_std_regret", "cv_avg_best_rate", "cv_std_best_rate"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_cv_results)
    print(f"Saved CSV -> {csv_path}")


if __name__ == "__main__":
    main()
