from __future__ import annotations

import argparse
from collections import Counter

from map_rag_gym.router.learned import LearnedRouter
from map_rag_gym.utils.io import read_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/batch_rollouts.json")
    ap.add_argument("--output", default="outputs/router.joblib")
    ap.add_argument(
        "--allowed_workflows",
        nargs="+",
        default=["W1", "W2", "W3", "W6"],
        help="Only keep labels in this set for training. Default keeps the strongest workflows.",
    )
    ap.add_argument("--min_samples_per_label", type=int, default=2)
    args = ap.parse_args()

    data = read_json(args.input)
    labels = data.get("best_labels", [])
    allowed = set(args.allowed_workflows)
    labels = [r for r in labels if r.get("best_workflow") in allowed]

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
    router = LearnedRouter()
    router.fit(questions, targets)
    router.save(args.output)

    print("Training label counts:", dict(Counter(targets)))
    if dropped:
        print(f"Dropped {dropped} labels because their class count was below {args.min_samples_per_label}.")
    for q in questions[:10]:
        pred, prob = router.predict(q)
        print(q, "->", pred, round(prob, 4))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
