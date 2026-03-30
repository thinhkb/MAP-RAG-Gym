from __future__ import annotations

import argparse
from collections import defaultdict

from map_rag_gym.core.pipeline import MAPRAGGym
from map_rag_gym.core.workflows import WORKFLOWS
from map_rag_gym.utils.io import read_json, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--qa", required=True)
    ap.add_argument("--llm_provider", default="dummy")
    ap.add_argument("--llm_model", default=None)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--out", default="outputs/batch_rollouts.json")
    args = ap.parse_args()

    qa = read_json(args.qa)[: args.limit]
    pipe = MAPRAGGym(args.corpus, llm_provider=args.llm_provider, llm_model=args.llm_model)
    all_runs = []
    best_labels = []
    workflow_stats = defaultdict(list)

    for item in qa:
        candidates = []
        for wf in WORKFLOWS:
            run = pipe.run(item["question"], item["answer"], wf, planner_reason="batch-rollout")
            payload = run.to_dict()
            all_runs.append(payload)
            candidates.append(payload)
            workflow_stats[wf].append(payload["final_scores"]["utility_total"])
        best = max(candidates, key=lambda r: r["final_scores"].get("utility_total", 0.0))
        best_labels.append({
            "question": item["question"],
            "answer": item["answer"],
            "best_workflow": best["workflow_id"],
            "best_utility": best["final_scores"]["utility_total"],
        })

    summary = {
        "runs": all_runs,
        "best_labels": best_labels,
        "workflow_avg_utility": {wf: round(sum(vals) / len(vals), 4) for wf, vals in workflow_stats.items()},
    }
    write_json(args.out, summary)
    print(f"Saved {len(all_runs)} runs to {args.out}")


if __name__ == "__main__":
    main()
