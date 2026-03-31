from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from statistics import mean

from map_rag_gym.core.pipeline import MAPRAGGym
from map_rag_gym.router.learned import LearnedRouter
from map_rag_gym.router.rule_based import RuleBasedRouter
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
    ap.add_argument("--fixed_workflows", nargs="+", default=["W1", "W2", "W3", "W6"])
    ap.add_argument("--out", default="outputs/router_eval.json")
    args = ap.parse_args()

    qa = read_json(args.qa)[: args.limit]
    pipe = MAPRAGGym(args.corpus, llm_provider=args.llm_provider, llm_model=args.llm_model)
    learned = LearnedRouter()
    learned.load(args.router_model)
    rule_based = RuleBasedRouter()

    buckets: dict[str, list[dict]] = defaultdict(list)
    per_question: list[dict] = []

    for item in qa:
        q = item["question"]
        a = item["answer"]
        entry: dict = {"question": q, "gold_answer": a, "results": {}}

        for wf in args.fixed_workflows:
            run = pipe.run(q, a, wf, planner_reason=f"fixed:{wf}", n_candidates=args.n_candidates)
            payload = run.to_dict()
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
        "settings": {
            "limit": args.limit,
            "llm_provider": args.llm_provider,
            "llm_model": args.llm_model,
            "n_candidates": args.n_candidates,
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
