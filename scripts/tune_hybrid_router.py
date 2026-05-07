from __future__ import annotations

import argparse

from map_rag_gym.utils.io import read_json, write_json


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Evaluation JSON from scripts/eval_phase4_router.py")
    ap.add_argument("--out", default="outputs/hybrid_threshold_sweep.json")
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.35, 0.4, 0.45, 0.5, 0.55, 0.6])
    ap.add_argument("--low_cost_thresholds", nargs="+", type=float, default=[0.45, 0.5, 0.55, 0.6, 0.65])
    ap.add_argument("--low_cost_workflows", nargs="+", default=["W1"])
    args = ap.parse_args()

    data = read_json(args.input)
    rows = data.get("per_question", [])
    if not rows:
        raise ValueError("No per_question records found in input file.")

    sweep = []
    best = None

    for threshold in args.thresholds:
        for low_cost_threshold in args.low_cost_thresholds:
            utilities = []
            fallback_count = 0
            for item in rows:
                learned = item["results"]["learned_router"]
                rule = item["results"]["rule_based"]
                learned_critic = item["results"].get("learned_router_critic", learned)
                rule_critic = item["results"].get("rule_based_critic", rule)
                workflow_id = learned["workflow_id"]
                confidence = float(learned.get("confidence", 0.0))
                active_threshold = low_cost_threshold if workflow_id in args.low_cost_workflows else threshold

                if confidence < active_threshold:
                    fallback_count += 1
                    selected = rule_critic if "rule_based_critic" in item["results"] else rule
                    mode = "rule"
                else:
                    selected = learned_critic if "learned_router_critic" in item["results"] else learned
                    mode = "learned"

                utilities.append(selected["utility_total"])

            record = {
                "min_confidence": threshold,
                "low_cost_confidence": low_cost_threshold,
                "low_cost_workflows": list(args.low_cost_workflows),
                "avg_utility": _avg(utilities),
                "fallback_count": fallback_count,
                "num_questions": len(rows),
            }
            sweep.append(record)
            if best is None or record["avg_utility"] > best["avg_utility"]:
                best = record

    output = {
        "source_eval_file": args.input,
        "num_questions": len(rows),
        "sweep": sorted(sweep, key=lambda item: item["avg_utility"], reverse=True),
        "best": best,
    }
    write_json(args.out, output)
    print("Best hybrid configuration:", best)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
