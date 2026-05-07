from __future__ import annotations

import argparse

from map_rag_gym.utils.io import read_json, write_json


def _dominates(left: dict, right: dict) -> bool:
    utility_better = float(left.get("avg_utility", 0.0)) >= float(right.get("avg_utility", 0.0))
    tokens_better = float(left.get("avg_tokens", 0.0)) <= float(right.get("avg_tokens", 0.0))
    latency_better = float(left.get("avg_latency_ms", 0.0)) <= float(right.get("avg_latency_ms", 0.0))
    retrieval_better = float(left.get("avg_retrieval_calls", 0.0)) <= float(right.get("avg_retrieval_calls", 0.0))
    strictly_better = (
        float(left.get("avg_utility", 0.0)) > float(right.get("avg_utility", 0.0))
        or float(left.get("avg_tokens", 0.0)) < float(right.get("avg_tokens", 0.0))
        or float(left.get("avg_latency_ms", 0.0)) < float(right.get("avg_latency_ms", 0.0))
        or float(left.get("avg_retrieval_calls", 0.0)) < float(right.get("avg_retrieval_calls", 0.0))
    )
    return utility_better and tokens_better and latency_better and retrieval_better and strictly_better


def _rank_key(item: tuple[str, dict]) -> tuple[float, float, float, float]:
    _, stats = item
    return (
        float(stats.get("avg_utility", 0.0)),
        -float(stats.get("avg_tokens", 0.0)),
        -float(stats.get("avg_latency_ms", 0.0)),
        -float(stats.get("avg_retrieval_calls", 0.0)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Evaluation JSON from scripts/eval_phase4_router.py")
    ap.add_argument("--methods", nargs="+", default=None, help="Optional subset of methods to analyze.")
    ap.add_argument("--max_tokens", type=float, default=None)
    ap.add_argument("--max_latency_ms", type=float, default=None)
    ap.add_argument("--max_retrieval_calls", type=float, default=None)
    ap.add_argument("--out", default="outputs/budget_policy_selection.json")
    args = ap.parse_args()

    data = read_json(args.input)
    summary = data.get("summary", {})
    methods = set(args.methods or [])
    rows = {
        method: stats
        for method, stats in summary.items()
        if not methods or method in methods
    }
    if not rows:
        raise ValueError("No methods available after filtering.")

    frontier = []
    for method, stats in rows.items():
        dominated_by = [
            other_method
            for other_method, other_stats in rows.items()
            if other_method != method and _dominates(other_stats, stats)
        ]
        if not dominated_by:
            frontier.append((method, stats))

    feasible = []
    for method, stats in rows.items():
        if args.max_tokens is not None and float(stats.get("avg_tokens", 0.0)) > args.max_tokens:
            continue
        if args.max_latency_ms is not None and float(stats.get("avg_latency_ms", 0.0)) > args.max_latency_ms:
            continue
        if args.max_retrieval_calls is not None and float(stats.get("avg_retrieval_calls", 0.0)) > args.max_retrieval_calls:
            continue
        feasible.append((method, stats))

    ranked_frontier = sorted(frontier, key=_rank_key, reverse=True)
    ranked_feasible = sorted(feasible, key=_rank_key, reverse=True)
    recommended = ranked_feasible[0][0] if ranked_feasible else (ranked_frontier[0][0] if ranked_frontier else None)

    output = {
        "source_eval_file": args.input,
        "budget_mode": data.get("settings", {}).get("budget_mode"),
        "constraints": {
            "max_tokens": args.max_tokens,
            "max_latency_ms": args.max_latency_ms,
            "max_retrieval_calls": args.max_retrieval_calls,
        },
        "methods": sorted(rows),
        "pareto_frontier": [{"method": method, **stats} for method, stats in ranked_frontier],
        "feasible_methods": [{"method": method, **stats} for method, stats in ranked_feasible],
        "recommended_method": recommended,
    }
    write_json(args.out, output)

    print(f"Pareto frontier methods: {[method for method, _ in ranked_frontier]}")
    if ranked_feasible:
        print(f"Recommended feasible method: {recommended}")
    else:
        print(f"No feasible method under constraints; frontier best is {recommended}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
