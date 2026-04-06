from __future__ import annotations

import argparse
from collections import Counter

from map_rag_gym.router.learned import LearnedRouter
from map_rag_gym.utils.experiment import build_experiment_manifest, set_global_seed
from map_rag_gym.utils.io import read_json, write_json


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
    args = ap.parse_args()

    set_global_seed(args.seed)
    data = read_json(args.input)
    source_manifest = data.get("manifest", {})
    labels = data.get("best_labels", [])
    allowed = set(args.allowed_workflows)
    labels = [r for r in labels if r.get("best_workflow") in allowed]
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
    targets = [r["best_workflow"] for r in labels]
    router = LearnedRouter(random_state=args.seed)
    router.fit(questions, targets)
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
                },
            ),
            "source_rollout_manifest": source_manifest,
            "label_counts_before_min_filter": dict(counts),
            "label_counts_used": dict(Counter(targets)),
            "num_labels_total": len(data.get("best_labels", [])),
            "num_labels_after_allowed_filter": labels_after_allowed,
            "num_labels_used": len(labels),
            "dropped_below_min_samples": dropped,
        },
    )

    print("Training label counts:", dict(Counter(targets)))
    if dropped:
        print(f"Dropped {dropped} labels because their class count was below {args.min_samples_per_label}.")
    for q in questions[:10]:
        pred, prob = router.predict(q)
        print(q, "->", pred, round(prob, 4))
    print(f"Saved {args.output}")
    print(f"Saved {args.output}.meta.json")


if __name__ == "__main__":
    main()
