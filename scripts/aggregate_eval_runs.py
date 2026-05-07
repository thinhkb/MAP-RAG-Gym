from __future__ import annotations

import argparse
from collections import defaultdict
from statistics import mean, stdev

from map_rag_gym.utils.io import read_json, write_json


def _safe_mean(values: list[float]) -> float:
    return round(mean(values), 4) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return round(stdev(values), 4) if len(values) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Evaluation JSON files from scripts/eval_phase4_router.py")
    ap.add_argument("--methods", nargs="+", default=None, help="Optional subset of methods to aggregate.")
    ap.add_argument("--reference_method", default=None, help="Optional method used for delta reporting.")
    ap.add_argument("--out", default="outputs/final_policy_benchmark.json")
    args = ap.parse_args()

    selected_methods = set(args.methods or [])
    rows_by_method: dict[str, list[dict]] = defaultdict(list)
    pairwise_wins: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    file_summaries: list[dict] = []

    for path in args.inputs:
        payload = read_json(path)
        summary = payload.get("summary", {})
        filtered = {
            method: stats
            for method, stats in summary.items()
            if not selected_methods or method in selected_methods
        }
        file_summaries.append(
            {
                "path": path,
                "methods": sorted(filtered),
                "manifest": payload.get("manifest", {}),
            }
        )
        ordered = sorted(filtered.items(), key=lambda kv: float(kv[1].get("avg_utility", 0.0)), reverse=True)
        for idx, (left_method, left_stats) in enumerate(ordered):
            rows_by_method[left_method].append(
                {
                    "path": path,
                    "avg_utility": float(left_stats.get("avg_utility", 0.0)),
                    "avg_em": float(left_stats.get("avg_em", 0.0)),
                    "avg_f1_proxy": float(left_stats.get("avg_f1_proxy", 0.0)),
                    "avg_tokens": float(left_stats.get("avg_tokens", 0.0)),
                    "avg_latency_ms": float(left_stats.get("avg_latency_ms", 0.0)),
                }
            )
            for right_method, _ in ordered[idx + 1 :]:
                pairwise_wins[left_method][right_method] += 1

    aggregated: dict[str, dict] = {}
    for method, rows in rows_by_method.items():
        utilities = [row["avg_utility"] for row in rows]
        ems = [row["avg_em"] for row in rows]
        f1s = [row["avg_f1_proxy"] for row in rows]
        tokens = [row["avg_tokens"] for row in rows]
        latencies = [row["avg_latency_ms"] for row in rows]
        aggregated[method] = {
            "num_runs": len(rows),
            "mean_utility": _safe_mean(utilities),
            "std_utility": _safe_std(utilities),
            "mean_em": _safe_mean(ems),
            "std_em": _safe_std(ems),
            "mean_f1_proxy": _safe_mean(f1s),
            "std_f1_proxy": _safe_std(f1s),
            "mean_tokens": _safe_mean(tokens),
            "mean_latency_ms": _safe_mean(latencies),
            "robust_score": round(_safe_mean(utilities) - _safe_std(utilities), 4),
            "runs": rows,
        }

    if args.reference_method and args.reference_method in aggregated:
        reference_mean = aggregated[args.reference_method]["mean_utility"]
        for stats in aggregated.values():
            stats["delta_vs_reference"] = round(stats["mean_utility"] - reference_mean, 4)

    ranking = sorted(
        aggregated.items(),
        key=lambda kv: (
            kv[1]["mean_utility"],
            kv[1]["robust_score"],
            -kv[1]["mean_tokens"],
        ),
        reverse=True,
    )
    recommended_method = ranking[0][0] if ranking else None
    output = {
        "inputs": args.inputs,
        "methods": sorted(selected_methods) if selected_methods else [],
        "reference_method": args.reference_method,
        "aggregated": aggregated,
        "ranking": [{"method": method, **stats} for method, stats in ranking],
        "recommended_method": recommended_method,
        "pairwise_wins": {left: dict(rights) for left, rights in pairwise_wins.items()},
        "file_summaries": file_summaries,
    }
    write_json(args.out, output)
    print(f"Saved {args.out}")
    if recommended_method:
        best = output["aggregated"][recommended_method]
        print(
            f"Recommended method: {recommended_method} | "
            f"mean_utility={best['mean_utility']:.4f} | std_utility={best['std_utility']:.4f}"
        )


if __name__ == "__main__":
    main()
