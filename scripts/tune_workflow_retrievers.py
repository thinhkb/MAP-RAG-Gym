from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from statistics import mean

from map_rag_gym.utils.io import read_json, write_json


def _score_tuple(stats: dict) -> tuple[float, float, float]:
    return (
        float(stats.get("avg_utility", 0.0)),
        float(stats.get("avg_f1_proxy", 0.0)),
        -float(stats.get("avg_tokens", 0.0)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to outputs/retriever_eval.json")
    ap.add_argument("--out", default="outputs/workflow_retriever_policy.json")
    args = ap.parse_args()

    payload = read_json(args.input)
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for key, stats in payload.get("summary", {}).items():
        if ":" not in key:
            continue
        retriever_name, workflow_id = key.split(":", 1)
        grouped[workflow_id][retriever_name] = stats

    per_workflow_best: dict[str, dict] = {}
    best_retrievers: list[str] = []
    retriever_utilities: dict[str, list[float]] = defaultdict(list)

    for workflow_id, stats_by_retriever in sorted(grouped.items()):
        ranked = sorted(stats_by_retriever.items(), key=lambda item: _score_tuple(item[1]), reverse=True)
        if not ranked:
            continue
        best_name, best_stats = ranked[0]
        per_workflow_best[workflow_id] = {
            "best_retriever": best_name,
            "avg_utility": round(float(best_stats.get("avg_utility", 0.0)), 4),
            "all_retrievers": {
                retriever_name: {
                    "avg_utility": round(float(stats.get("avg_utility", 0.0)), 4),
                    "avg_f1_proxy": round(float(stats.get("avg_f1_proxy", 0.0)), 4),
                    "avg_tokens": round(float(stats.get("avg_tokens", 0.0)), 4),
                }
                for retriever_name, stats in sorted(stats_by_retriever.items())
            },
        }
        best_retrievers.append(best_name)
        for retriever_name, stats in stats_by_retriever.items():
            retriever_utilities[retriever_name].append(float(stats.get("avg_utility", 0.0)))

    counts = Counter(best_retrievers)
    recommended_default = None
    if counts:
        recommended_default = max(
            counts,
            key=lambda retriever_name: (counts[retriever_name], mean(retriever_utilities[retriever_name])),
        )

    workflow_retriever_overrides = {
        workflow_id: row["best_retriever"]
        for workflow_id, row in per_workflow_best.items()
        if recommended_default is not None and row["best_retriever"] != recommended_default
    }

    recommended_policy = {
        "default_retriever": recommended_default,
        "workflow_retriever_overrides": workflow_retriever_overrides,
    }
    output = {
        "source": args.input,
        "recommended_policy": recommended_policy,
        "per_workflow_best": per_workflow_best,
        "best_retriever_counts": dict(counts),
    }
    write_json(args.out, output)
    print(f"Saved {args.out}")
    print(output)


if __name__ == "__main__":
    main()
