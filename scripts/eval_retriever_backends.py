from __future__ import annotations

import argparse
from collections import defaultdict
from statistics import mean

from map_rag_gym.core.pipeline import MAPRAGGym
from map_rag_gym.core.workflows import WORKFLOWS
from map_rag_gym.utils.dataset import normalize_qa_records
from map_rag_gym.utils.io import read_json, write_json


def _avg(values: list[float]) -> float:
    return round(mean(values), 4) if values else 0.0


def _summarize(runs: list[dict]) -> dict:
    return {
        "num_runs": len(runs),
        "avg_utility": _avg([run["final_scores"].get("utility_total", 0.0) for run in runs]),
        "avg_em": _avg([run["final_scores"].get("em", 0.0) for run in runs]),
        "avg_f1_proxy": _avg([run["final_scores"].get("f1_proxy", 0.0) for run in runs]),
        "avg_tokens": _avg([run["total_cost"].get("tokens", 0.0) for run in runs]),
        "avg_latency_ms": _avg([run["total_cost"].get("latency_ms", 0.0) for run in runs]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--qa", required=True)
    ap.add_argument("--llm_provider", default="dummy")
    ap.add_argument("--llm_model", default=None)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--n_candidates", type=int, default=3)
    ap.add_argument("--retrievers", nargs="+", default=["bm25", "tfidf", "hybrid"])
    ap.add_argument("--hybrid_weight", type=float, default=0.5)
    ap.add_argument("--workflows", nargs="+", default=["W2", "W3", "W6"])
    ap.add_argument("--out", default="outputs/retriever_eval.json")
    args = ap.parse_args()

    qa = normalize_qa_records(read_json(args.qa))[: args.limit]
    buckets: dict[str, list[dict]] = defaultdict(list)

    for retriever_type in args.retrievers:
        pipe = MAPRAGGym(
            args.corpus,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            retriever_type=retriever_type,
            retriever_bm25_weight=args.hybrid_weight,
        )
        for item in qa:
            for workflow_id in args.workflows:
                if workflow_id not in WORKFLOWS:
                    continue
                run = pipe.run(item["question"], item["answer"], workflow_id, planner_reason=f"retriever:{retriever_type}", n_candidates=args.n_candidates)
                payload = run.to_dict()
                payload.setdefault("metadata", {})["question_id"] = item["id"]
                buckets[f"{retriever_type}:{workflow_id}"].append(payload)

    summary = {key: _summarize(runs) for key, runs in buckets.items()}
    payload = {
        "settings": {
            "llm_provider": args.llm_provider,
            "llm_model": args.llm_model,
            "limit": args.limit,
            "n_candidates": args.n_candidates,
            "retrievers": args.retrievers,
            "hybrid_weight": args.hybrid_weight,
            "workflows": args.workflows,
        },
        "summary": summary,
    }
    write_json(args.out, payload)
    print(f"Saved {args.out}")
    for key, stats in sorted(summary.items(), key=lambda item: item[1]["avg_utility"], reverse=True):
        print(key, stats)


if __name__ == "__main__":
    main()
