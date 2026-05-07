from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from statistics import mean

from map_rag_gym.router.learned import LearnedRouter
from map_rag_gym.router.meta import MetaRouterGate
from map_rag_gym.router.rule_based import RuleBasedRouter
from map_rag_gym.utils.experiment import build_experiment_manifest, set_global_seed
from map_rag_gym.utils.io import read_json, write_json


def _group_runs_by_question(runs: list[dict]) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for run in runs:
        grouped[run["question"]][run["workflow_id"]] = run
    return grouped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--router_model", required=True)
    ap.add_argument("--output", default="outputs/meta_router_gate.joblib")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--tie_margin", type=float, default=0.0)
    args = ap.parse_args()

    set_global_seed(args.seed)
    rollout = read_json(args.rollouts)
    source_manifest = rollout.get("manifest", {})
    grouped = _group_runs_by_question(rollout.get("runs", []))

    learned = LearnedRouter(random_state=args.seed)
    learned.load(args.router_model)
    rule_router = RuleBasedRouter()

    rows: list[dict] = []
    labels: list[str] = []
    sample_weight: list[float] = []
    diagnostics = []

    for question, runs_by_wf in grouped.items():
        learned_wf, learned_conf = learned.predict(question)
        rule_decision = rule_router.decide(question)
        if learned_wf not in runs_by_wf or rule_decision.workflow_id not in runs_by_wf:
            continue
        learned_utility = float(runs_by_wf[learned_wf]["final_scores"]["utility_total"])
        rule_utility = float(runs_by_wf[rule_decision.workflow_id]["final_scores"]["utility_total"])
        diff = learned_utility - rule_utility
        label = "learned" if diff > args.tie_margin else "rule"

        rows.append({
            "question": question,
            "learned_workflow": learned_wf,
            "learned_confidence": learned_conf,
            "rule_workflow": rule_decision.workflow_id,
            "rule_confidence": rule_decision.confidence,
        })
        labels.append(label)
        sample_weight.append(max(0.1, abs(diff)))
        diagnostics.append({
            "question": question,
            "learned_workflow": learned_wf,
            "rule_workflow": rule_decision.workflow_id,
            "learned_confidence": round(learned_conf, 4),
            "rule_confidence": round(rule_decision.confidence, 4),
            "learned_utility": round(learned_utility, 4),
            "rule_utility": round(rule_utility, 4),
            "utility_diff": round(diff, 4),
            "label": label,
        })

    if len(set(labels)) < 2:
        raise ValueError("Meta-router training needs both 'learned' and 'rule' labels.")

    gate = MetaRouterGate(random_state=args.seed)
    gate.fit(rows, labels, sample_weight=sample_weight)
    gate.save(args.output)

    train_preds = [gate.predict(row)[0] for row in rows]
    accuracy = sum(int(pred == gold) for pred, gold in zip(train_preds, labels)) / len(labels)
    avg_margin = mean(abs(item["utility_diff"]) for item in diagnostics) if diagnostics else 0.0

    meta = {
        "manifest": build_experiment_manifest(
            script_name="scripts/train_meta_router.py",
            qa_path=args.rollouts,
            dataset_name=source_manifest.get("dataset", {}).get("name", "meta_router"),
            dataset_split=source_manifest.get("dataset", {}).get("split", "custom"),
            limit=source_manifest.get("dataset", {}).get("limit"),
            effective_questions=len(rows),
            seed=args.seed,
            prompt_version=source_manifest.get("reproducibility", {}).get("prompt_version", "v1"),
            router_model_path=args.output,
            settings={
                "source_rollout_file": args.rollouts,
                "source_router_model": args.router_model,
                "tie_margin": args.tie_margin,
            },
        ),
        "source_rollout_manifest": source_manifest,
        "label_counts": dict(Counter(labels)),
        "train_accuracy": round(accuracy, 4),
        "avg_abs_utility_diff": round(avg_margin, 4),
        "diagnostics_preview": diagnostics[:200],
    }
    write_json(f"{args.output}.meta.json", meta)

    print("Meta-router label counts:", dict(Counter(labels)))
    print("Meta-router train accuracy:", round(accuracy, 4))
    print("Average absolute utility diff:", round(avg_margin, 4))
    print(f"Saved {args.output}")
    print(f"Saved {args.output}.meta.json")


if __name__ == "__main__":
    main()
