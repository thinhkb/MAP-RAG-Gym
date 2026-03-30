from __future__ import annotations

import argparse

from map_rag_gym.router.learned import LearnedRouter
from map_rag_gym.utils.io import read_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/batch_rollouts.json")
    ap.add_argument("--output", default="outputs/router.joblib")
    args = ap.parse_args()

    data = read_json(args.input)
    labels = data.get("best_labels", [])
    questions = [r["question"] for r in labels]
    targets = [r["best_workflow"] for r in labels]
    router = LearnedRouter()
    router.fit(questions, targets)
    router.save(args.output)
    for q in questions[:10]:
        pred, prob = router.predict(q)
        print(q, "->", pred, round(prob, 4))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
