from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from map_rag_gym.core.pipeline import MAPRAGGym
from map_rag_gym.core.workflows import WORKFLOWS
from map_rag_gym.utils.io import read_json, write_json


def build_summary(all_runs, best_labels, workflow_stats):
    return {
        "runs": all_runs,
        "best_labels": best_labels,
        "workflow_avg_utility": {
            wf: round(sum(vals) / len(vals), 4) for wf, vals in workflow_stats.items() if vals
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--qa", required=True)
    ap.add_argument("--llm_provider", default="dummy")
    ap.add_argument("--llm_model", default=None)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--out", default="outputs/batch_rollouts.json")
    ap.add_argument("--resume", action="store_true", help="Resume from an existing output file if present.")
    args = ap.parse_args()

    qa = read_json(args.qa)[: args.limit]
    pipe = MAPRAGGym(args.corpus, llm_provider=args.llm_provider, llm_model=args.llm_model)
    out_path = Path(args.out)
    all_runs = []
    best_labels = []
    workflow_stats = defaultdict(list)
    completed_questions = set()

    if args.resume and out_path.exists():
        existing = read_json(args.out)
        all_runs = list(existing.get("runs", []))
        best_labels = list(existing.get("best_labels", []))
        completed_questions = {item["question"] for item in best_labels}
        for run in all_runs:
            workflow_stats[run["workflow_id"]].append(run["final_scores"]["utility_total"])
        print(f"Resuming from {args.out}: {len(completed_questions)} questions already completed.")

    for item in qa:
        if item["question"] in completed_questions:
            continue
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
        completed_questions.add(item["question"])

        write_json(args.out, build_summary(all_runs, best_labels, workflow_stats))
        print(f"Checkpointed {len(best_labels)}/{len(qa)} questions to {args.out}")

    summary = build_summary(all_runs, best_labels, workflow_stats)
    write_json(args.out, summary)
    print(f"Saved {len(all_runs)} runs to {args.out}")


if __name__ == "__main__":
    main()
