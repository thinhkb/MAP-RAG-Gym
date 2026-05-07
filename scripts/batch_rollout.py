from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

from map_rag_gym.core.pipeline import MAPRAGGym
from map_rag_gym.core.workflows import WORKFLOWS
from map_rag_gym.evaluation.heuristics import get_utility_profile
from map_rag_gym.retrieval.policy import parse_workflow_retriever_overrides
from map_rag_gym.utils.dataset import normalize_qa_records
from map_rag_gym.utils.experiment import build_experiment_manifest, compare_manifest_fields, set_global_seed
from map_rag_gym.utils.io import read_json, write_json


def build_summary(manifest, all_runs, best_labels, workflow_stats):
    margins = [row["utility_margin"] for row in best_labels if row.get("utility_margin") is not None]
    return {
        "manifest": manifest,
        "runs": all_runs,
        "best_labels": best_labels,
        "workflow_avg_utility": {
            wf: round(sum(vals) / len(vals), 4) for wf, vals in workflow_stats.items() if vals
        },
        "label_margin_stats": {
            "num_questions": len(best_labels),
            "avg_margin": round(mean(margins), 4) if margins else 0.0,
            "small_margin_le_0.05": sum(1 for margin in margins if margin <= 0.05),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--qa", required=True)
    ap.add_argument("--llm_provider", default="dummy")
    ap.add_argument("--llm_model", default=None)
    ap.add_argument("--retriever_type", default="bm25", choices=["bm25", "tfidf", "hybrid"])
    ap.add_argument("--retriever_bm25_weight", type=float, default=0.5)
    ap.add_argument("--workflow_retriever_overrides", nargs="+", default=[], help="Optional overrides like W6=hybrid W3=bm25.")
    ap.add_argument("--workflow_ids", nargs="+", default=None, help="Optional workflow subset, e.g. W2 W3.")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--n_candidates", type=int, default=3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--budget_mode", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--dataset_name", default=None)
    ap.add_argument("--dataset_split", default=None)
    ap.add_argument("--prompt_version", default="v1")
    ap.add_argument("--out", default="outputs/batch_rollouts.json")
    ap.add_argument("--resume", action="store_true", help="Resume from an existing output file if present.")
    args = ap.parse_args()

    set_global_seed(args.seed)
    workflow_retriever_overrides = parse_workflow_retriever_overrides(args.workflow_retriever_overrides)
    workflow_ids = [str(workflow_id).upper() for workflow_id in (args.workflow_ids or list(WORKFLOWS))]
    unknown_workflows = [workflow_id for workflow_id in workflow_ids if workflow_id not in WORKFLOWS]
    if unknown_workflows:
        raise ValueError(f"Unknown workflows: {unknown_workflows}. Valid workflows: {sorted(WORKFLOWS)}")
    qa = normalize_qa_records(read_json(args.qa))[: args.limit]
    pipe = MAPRAGGym(
        args.corpus,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        retriever_type=args.retriever_type,
        retriever_bm25_weight=args.retriever_bm25_weight,
        workflow_retriever_overrides=workflow_retriever_overrides,
    )
    out_path = Path(args.out)
    all_runs = []
    best_labels = []
    workflow_stats = defaultdict(list)
    completed_questions = set()
    manifest = build_experiment_manifest(
        script_name="scripts/batch_rollout.py",
        qa_path=args.qa,
        corpus_path=args.corpus,
        llm_provider=pipe.llm_provider,
        llm_model=pipe.llm_model,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        limit=args.limit,
        effective_questions=len(qa),
        seed=args.seed,
        prompt_version=args.prompt_version,
        utility_config=get_utility_profile(args.budget_mode),
        settings={
            "workflow_ids": workflow_ids,
            "n_candidates": args.n_candidates,
            "budget_mode": args.budget_mode,
            "retriever_type": args.retriever_type,
            "retriever_bm25_weight": args.retriever_bm25_weight,
            "workflow_retriever_overrides": workflow_retriever_overrides,
            "resume": args.resume,
        },
    )

    if args.resume and out_path.exists():
        existing = read_json(args.out)
        mismatches = compare_manifest_fields(
            existing.get("manifest", {}),
            manifest,
            [
                ("paths", "qa"),
                ("paths", "corpus"),
                ("dataset", "name"),
                ("dataset", "split"),
                ("llm", "provider"),
                ("llm", "model"),
                ("reproducibility", "seed"),
                ("reproducibility", "prompt_version"),
                ("settings", "workflow_ids"),
                ("settings", "n_candidates"),
                ("settings", "budget_mode"),
                ("settings", "retriever_type"),
                ("settings", "retriever_bm25_weight"),
                ("settings", "workflow_retriever_overrides"),
            ],
        )
        if mismatches:
            raise ValueError(f"Cannot resume {args.out}: manifest mismatch on {', '.join(mismatches)}.")
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
        for wf in workflow_ids:
            run = pipe.run(
                item["question"],
                item["answer"],
                wf,
                planner_reason="batch-rollout",
                n_candidates=args.n_candidates,
                budget_mode=args.budget_mode,
            )
            payload = run.to_dict()
            payload.setdefault("metadata", {})["question_id"] = item["id"]
            payload["metadata"]["dataset_split"] = manifest["dataset"]["split"]
            all_runs.append(payload)
            candidates.append(payload)
            workflow_stats[wf].append(payload["final_scores"]["utility_total"])
        ranked = sorted(candidates, key=lambda r: r["final_scores"].get("utility_total", 0.0), reverse=True)
        best = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None
        best_labels.append({
            "question_id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "best_workflow": best["workflow_id"],
            "best_utility": best["final_scores"]["utility_total"],
            "runner_up_workflow": runner_up["workflow_id"] if runner_up else None,
            "runner_up_utility": runner_up["final_scores"]["utility_total"] if runner_up else None,
            "utility_margin": round(best["final_scores"]["utility_total"] - runner_up["final_scores"]["utility_total"], 4) if runner_up else None,
        })
        completed_questions.add(item["question"])

        write_json(args.out, build_summary(manifest, all_runs, best_labels, workflow_stats))
        print(f"Checkpointed {len(best_labels)}/{len(qa)} questions to {args.out}")

    summary = build_summary(manifest, all_runs, best_labels, workflow_stats)
    write_json(args.out, summary)
    print(f"Saved {len(all_runs)} runs to {args.out}")


if __name__ == "__main__":
    main()
