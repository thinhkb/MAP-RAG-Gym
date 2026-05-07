from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from map_rag_gym.evaluation.heuristics import UTILITY_PROFILES, compute_budgeted_utility
from map_rag_gym.router.budget import ALLOWED_WORKFLOWS_BY_BUDGET, normalize_budget_mode
from map_rag_gym.router.learned import LearnedRouter
from map_rag_gym.utils.experiment import build_experiment_manifest, set_global_seed
from map_rag_gym.utils.io import read_json, write_json


def _group_runs_by_question(runs: list[dict]) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for run in runs:
        grouped[run["question"]][run["workflow_id"]] = run
    return grouped


def _parse_budget_margin_overrides(items: list[str]) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for item in items:
        mode, raw_value = item.split("=", 1)
        overrides[normalize_budget_mode(mode)] = float(raw_value)
    return overrides


def _parse_budget_workflow_preferences(items: list[str]) -> dict[str, list[str]]:
    preferences: dict[str, list[str]] = {}
    for item in items:
        mode, raw_workflows = item.split("=", 1)
        workflows = [workflow.strip().upper() for workflow in raw_workflows.split(",") if workflow.strip()]
        preferences[normalize_budget_mode(mode)] = workflows
    return preferences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/batch_rollouts.json")
    ap.add_argument("--output", default="outputs/router.joblib")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument(
        "--allowed_workflows",
        nargs="+",
        default=["W1", "W2", "W3", "W6"],
        help="Only keep labels in this set for training. Default keeps the strongest workflows.",
    )
    ap.add_argument("--min_samples_per_label", type=int, default=2)
    ap.add_argument(
        "--budget_modes",
        nargs="+",
        default=["medium"],
        help="Budget-conditioned labels to train on, e.g. low medium high.",
    )
    ap.add_argument(
        "--budget_margin_overrides",
        nargs="+",
        default=[],
        help="Optional utility margins like low=0.05. Preferred workflows inside the margin will be selected.",
    )
    ap.add_argument(
        "--budget_preferred_workflows",
        nargs="+",
        default=[],
        help="Optional workflow preferences like low=W3 medium=W3,W2.",
    )
    args = ap.parse_args()

    set_global_seed(args.seed)
    data = read_json(args.input)
    source_manifest = data.get("manifest", {})
    budget_modes = [normalize_budget_mode(mode) for mode in args.budget_modes]
    budget_margin_overrides = _parse_budget_margin_overrides(args.budget_margin_overrides)
    budget_preferred_workflows = _parse_budget_workflow_preferences(args.budget_preferred_workflows)
    allowed = set(args.allowed_workflows)
    labels = []
    label_diagnostics = []
    grouped_runs = _group_runs_by_question(data.get("runs", []))
    calibration_counts = Counter()

    for question, runs_by_wf in grouped_runs.items():
        sample_run = next(iter(runs_by_wf.values()), None)
        if not sample_run:
            continue
        question_id = sample_run.get("metadata", {}).get("question_id")
        gold_answer = sample_run.get("gold_answer", "")
        for budget_mode in budget_modes:
            allowed_budget = allowed & ALLOWED_WORKFLOWS_BY_BUDGET[budget_mode]
            candidates = []
            for workflow_id, run in runs_by_wf.items():
                if workflow_id not in allowed_budget:
                    continue
                utility = compute_budgeted_utility(
                    final_scores=run.get("final_scores", {}),
                    total_cost=run.get("total_cost", {}),
                    process_score=float(run.get("final_scores", {}).get("process_score", 0.0)),
                    budget_mode=budget_mode,
                )
                candidates.append((workflow_id, utility, run))
            if not candidates:
                continue
            ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
            best_workflow, best_utility, _ = ranked[0]
            runner_up = ranked[1] if len(ranked) > 1 else None
            selected_by_calibration = False
            preferred_workflows = [
                workflow_id
                for workflow_id in budget_preferred_workflows.get(budget_mode, [])
                if workflow_id in allowed_budget
            ]
            margin = budget_margin_overrides.get(budget_mode, 0.0)
            if preferred_workflows and margin > 0:
                preferred_candidates = [
                    item
                    for item in ranked
                    if item[0] in preferred_workflows and (best_utility - item[1]) <= margin
                ]
                if preferred_candidates:
                    calibrated_best = preferred_candidates[0]
                    if calibrated_best[0] != best_workflow:
                        selected_by_calibration = True
                        calibration_counts[budget_mode] += 1
                    best_workflow, best_utility, _ = calibrated_best
            labels.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "answer": gold_answer,
                    "budget_mode": budget_mode,
                    "best_workflow": best_workflow,
                    "best_utility": round(best_utility, 4),
                    "runner_up_workflow": runner_up[0] if runner_up else None,
                    "runner_up_utility": round(runner_up[1], 4) if runner_up else None,
                    "utility_margin": round(best_utility - runner_up[1], 4) if runner_up else None,
                    "selected_by_calibration": selected_by_calibration,
                }
            )
            label_diagnostics.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "budget_mode": budget_mode,
                    "selected_by_calibration": selected_by_calibration,
                    "preferred_workflows": preferred_workflows,
                    "margin_override": margin,
                    "ranked_workflows": [
                        {"workflow_id": workflow_id, "utility": round(utility, 4)}
                        for workflow_id, utility, _ in ranked
                    ],
                }
            )

    labels_after_allowed = len(labels)
    counts = Counter(r["best_workflow"] for r in labels)
    filtered = [r for r in labels if counts[r["best_workflow"]] >= args.min_samples_per_label]
    dropped = len(labels) - len(filtered)
    labels = filtered

    if len(labels) < max(4, len(set(r["best_workflow"] for r in labels))):
        raise ValueError(
            f"Not enough labels after filtering: {len(labels)}. "
            f"Try adding more rollout data or relaxing --allowed_workflows / --min_samples_per_label."
        )

    questions = [r["question"] for r in labels]
    budget_values = [r["budget_mode"] for r in labels]
    targets = [r["best_workflow"] for r in labels]
    router = LearnedRouter(random_state=args.seed)
    router.fit(questions, targets, budget_modes=budget_values)
    router.save(args.output)
    write_json(
        f"{args.output}.meta.json",
        {
            "manifest": build_experiment_manifest(
                script_name="scripts/train_phase4_router.py",
                qa_path=args.input,
                dataset_name=source_manifest.get("dataset", {}).get("name", "router_training"),
                dataset_split=source_manifest.get("dataset", {}).get("split", "custom"),
                limit=len(data.get("best_labels", [])),
                effective_questions=len(labels),
                seed=args.seed,
                prompt_version=source_manifest.get("reproducibility", {}).get("prompt_version", "v1"),
                router_model_path=args.output,
                settings={
                    "input_rollout_file": args.input,
                    "output_model": args.output,
                    "allowed_workflows": args.allowed_workflows,
                    "min_samples_per_label": args.min_samples_per_label,
                    "budget_modes": budget_modes,
                    "utility_profiles": {mode: UTILITY_PROFILES[mode] for mode in budget_modes},
                    "budget_margin_overrides": budget_margin_overrides,
                    "budget_preferred_workflows": budget_preferred_workflows,
                },
            ),
            "source_rollout_manifest": source_manifest,
            "label_counts_before_min_filter": dict(counts),
            "label_counts_used": dict(Counter(targets)),
            "label_counts_by_budget_used": {
                mode: dict(Counter(r["best_workflow"] for r in labels if r["budget_mode"] == mode))
                for mode in budget_modes
            },
            "num_labels_total": len(label_diagnostics),
            "num_labels_after_allowed_filter": labels_after_allowed,
            "num_labels_used": len(labels),
            "dropped_below_min_samples": dropped,
            "calibration_counts": dict(calibration_counts),
            "diagnostics_preview": label_diagnostics[:200],
        },
    )

    print("Training label counts:", dict(Counter(targets)))
    print("Training label counts by budget:", {
        mode: dict(Counter(r["best_workflow"] for r in labels if r["budget_mode"] == mode))
        for mode in budget_modes
    })
    if calibration_counts:
        print("Calibration counts:", dict(calibration_counts))
    if dropped:
        print(f"Dropped {dropped} labels because their class count was below {args.min_samples_per_label}.")
    for q, budget_mode in list(zip(questions, budget_values))[:10]:
        pred, prob = router.predict(q, budget_mode=budget_mode)
        print(f"[{budget_mode}] {q} -> {pred} {round(prob, 4)}")
    print(f"Saved {args.output}")
    print(f"Saved {args.output}.meta.json")


if __name__ == "__main__":
    main()
