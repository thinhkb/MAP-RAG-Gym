from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from statistics import mean

from map_rag_gym.core.pipeline import MAPRAGGym
from map_rag_gym.evaluation.heuristics import UTILITY_CONFIG
from map_rag_gym.router.learned import LearnedRouter
from map_rag_gym.router.rule_based import RuleBasedRouter
from map_rag_gym.utils.dataset import normalize_qa_records
from map_rag_gym.utils.experiment import build_experiment_manifest, set_global_seed
from map_rag_gym.utils.io import read_json, write_json


def _avg(values: list[float]) -> float:
    return round(mean(values), 4) if values else 0.0


def _summarize(runs: list[dict]) -> dict:
    if not runs:
        return {
            "num_runs": 0,
            "avg_utility": 0.0,
            "avg_em": 0.0,
            "avg_f1_proxy": 0.0,
            "avg_process_score": 0.0,
            "avg_tokens": 0.0,
            "avg_retrieval_calls": 0.0,
            "avg_latency_ms": 0.0,
            "workflow_counts": {},
        }
    return {
        "num_runs": len(runs),
        "avg_utility": _avg([r["final_scores"].get("utility_total", 0.0) for r in runs]),
        "avg_em": _avg([r["final_scores"].get("em", 0.0) for r in runs]),
        "avg_f1_proxy": _avg([r["final_scores"].get("f1_proxy", 0.0) for r in runs]),
        "avg_process_score": _avg([r["final_scores"].get("process_score", 0.0) for r in runs]),
        "avg_tokens": _avg([r["total_cost"].get("tokens", 0.0) for r in runs]),
        "avg_retrieval_calls": _avg([r["total_cost"].get("retrieval_calls", 0.0) for r in runs]),
        "avg_latency_ms": _avg([r["total_cost"].get("latency_ms", 0.0) for r in runs]),
        "workflow_counts": dict(Counter(r["workflow_id"] for r in runs)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--qa", required=True)
    ap.add_argument("--router_model", required=True)
    ap.add_argument("--llm_provider", default="dummy")
    ap.add_argument("--llm_model", default=None)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--n_candidates", type=int, default=1)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--dataset_name", default=None)
    ap.add_argument("--dataset_split", default=None)
    ap.add_argument("--prompt_version", default="v1")
    ap.add_argument("--fixed_workflows", nargs="+", default=["W1", "W2", "W3", "W6"])
    ap.add_argument("--out", default="outputs/router_eval.json")
    args = ap.parse_args()

    set_global_seed(args.seed)
    qa = normalize_qa_records(read_json(args.qa))[: args.limit]
    pipe = MAPRAGGym(args.corpus, llm_provider=args.llm_provider, llm_model=args.llm_model)
    learned = LearnedRouter(random_state=args.seed)
    learned.load(args.router_model)
    rule_based = RuleBasedRouter()
    manifest = build_experiment_manifest(
        script_name="scripts/eval_phase4_router.py",
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
        router_model_path=args.router_model,
        utility_config=UTILITY_CONFIG,
        settings={
            "fixed_workflows": args.fixed_workflows,
            "n_candidates": args.n_candidates,
        },
    )

    buckets: dict[str, list[dict]] = defaultdict(list)
    per_question: list[dict] = []

    for item in qa:
        q = item["question"]
        a = item["answer"]
        entry: dict = {"question_id": item["id"], "question": q, "gold_answer": a, "results": {}}

        for wf in args.fixed_workflows:
            run = pipe.run(q, a, wf, planner_reason=f"fixed:{wf}", n_candidates=args.n_candidates)
            payload = run.to_dict()
            payload.setdefault("metadata", {})["question_id"] = item["id"]
            payload["metadata"]["dataset_split"] = manifest["dataset"]["split"]
            buckets[f"fixed_{wf}"].append(payload)
            entry["results"][f"fixed_{wf}"] = {
                "workflow_id": wf,
                "utility_total": payload["final_scores"]["utility_total"],
                "em": payload["final_scores"]["em"],
                "f1_proxy": payload["final_scores"]["f1_proxy"],
            }

        rb = rule_based.decide(q)
        rb_run = pipe.run(q, a, rb.workflow_id, planner_reason=f"rule:{rb.reason}", n_candidates=args.n_candidates)
        rb_payload = rb_run.to_dict()
        rb_payload.setdefault("metadata", {})["question_id"] = item["id"]
        rb_payload["metadata"]["dataset_split"] = manifest["dataset"]["split"]
        buckets["rule_based"].append(rb_payload)
        entry["results"]["rule_based"] = {
            "workflow_id": rb.workflow_id,
            "utility_total": rb_payload["final_scores"]["utility_total"],
            "em": rb_payload["final_scores"]["em"],
            "f1_proxy": rb_payload["final_scores"]["f1_proxy"],
        }

        pred, prob = learned.predict(q)
        lr_run = pipe.run(q, a, pred, planner_reason=f"learned:{prob:.4f}", n_candidates=args.n_candidates)
        lr_payload = lr_run.to_dict()
        lr_payload.setdefault("metadata", {})["router_confidence"] = prob
        lr_payload["metadata"]["question_id"] = item["id"]
        lr_payload["metadata"]["dataset_split"] = manifest["dataset"]["split"]
        buckets["learned_router"].append(lr_payload)
        entry["results"]["learned_router"] = {
            "workflow_id": pred,
            "confidence": round(prob, 4),
            "utility_total": lr_payload["final_scores"]["utility_total"],
            "em": lr_payload["final_scores"]["em"],
            "f1_proxy": lr_payload["final_scores"]["f1_proxy"],
        }

        per_question.append(entry)

    summary = {name: _summarize(runs) for name, runs in buckets.items()}

    print("=== Router evaluation summary ===")
    ranked = sorted(summary.items(), key=lambda kv: kv[1]["avg_utility"], reverse=True)
    for name, stats in ranked:
        print(
            f"{name:14s} | utility={stats['avg_utility']:.4f} | em={stats['avg_em']:.4f} | "
            f"f1={stats['avg_f1_proxy']:.4f} | tokens={stats['avg_tokens']:.1f} | workflows={stats['workflow_counts']}"
        )

    payload = {
        "manifest": manifest,
        "settings": {
            "limit": args.limit,
            "llm_provider": pipe.llm_provider,
            "llm_model": pipe.llm_model,
            "n_candidates": args.n_candidates,
            "seed": args.seed,
            "fixed_workflows": args.fixed_workflows,
            "router_model": args.router_model,
        },
        "summary": summary,
        "per_question": per_question,
    }
    write_json(args.out, payload)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
