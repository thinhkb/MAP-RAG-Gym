from __future__ import annotations

import argparse
import math
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from map_rag_gym.critic.model import ProcessCritic
from map_rag_gym.utils.experiment import build_experiment_manifest, set_global_seed
from map_rag_gym.utils.io import read_json, write_json


def _group_key(row: dict) -> str:
    return str(row.get("question_id") or row.get("question") or row.get("run_id"))


def _group_holdout_split(rows: list[dict], holdout_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    groups = sorted({_group_key(row) for row in rows})
    rng = random.Random(seed)
    rng.shuffle(groups)
    holdout_size = max(1, int(len(groups) * holdout_ratio))
    holdout_groups = set(groups[:holdout_size])
    train_rows = [row for row in rows if _group_key(row) not in holdout_groups]
    eval_rows = [row for row in rows if _group_key(row) in holdout_groups]
    return train_rows, eval_rows


def _metrics(y_true: list[float], y_pred: list[float]) -> dict:
    if not y_true:
        return {"count": 0, "mae": 0.0, "rmse": 0.0, "pearson": 0.0, "spearman": 0.0}

    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    mae = float(np.mean(np.abs(yp - yt)))
    rmse = float(math.sqrt(np.mean((yp - yt) ** 2)))

    if len(yt) > 1 and float(np.std(yt)) > 0 and float(np.std(yp)) > 0:
        pearson = float(np.corrcoef(yt, yp)[0, 1])
        spearman = float(np.corrcoef(pd.Series(yt).rank(), pd.Series(yp).rank())[0, 1])
    else:
        pearson = 0.0
        spearman = 0.0

    return {
        "count": len(y_true),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "pearson": round(pearson, 4),
        "spearman": round(spearman, 4),
    }


def _evaluate_rows(rows: list[dict], preds: list[float], target: str) -> dict:
    summary = {
        "overall": _metrics([float(row[target]) for row in rows], preds),
        "per_module": {},
    }
    buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row, pred in zip(rows, preds):
        buckets[row["module"]].append((float(row[target]), float(pred)))
    for module, pairs in buckets.items():
        y_true = [left for left, _ in pairs]
        y_pred = [right for _, right in pairs]
        summary["per_module"][module] = _metrics(y_true, y_pred)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Process dataset JSON produced by scripts/build_process_dataset.py")
    ap.add_argument("--output", default="outputs/process_critic.joblib")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--target", default="blended_reward")
    ap.add_argument("--holdout_ratio", type=float, default=0.15)
    ap.add_argument("--eval_input", default=None, help="Optional separate process dataset for evaluation.")
    ap.add_argument("--modules", nargs="+", default=None, help="Optional module whitelist.")
    args = ap.parse_args()

    set_global_seed(args.seed)
    data = read_json(args.input)
    source_manifest = data.get("manifest", {})
    rows = list(data.get("examples", []))
    if args.modules:
        allowed = set(args.modules)
        rows = [row for row in rows if row.get("module") in allowed]

    if not rows:
        raise ValueError("No process examples available after filtering.")

    if args.eval_input:
        eval_data = read_json(args.eval_input)
        eval_rows = list(eval_data.get("examples", []))
        if args.modules:
            allowed = set(args.modules)
            eval_rows = [row for row in eval_rows if row.get("module") in allowed]
        train_rows = rows
    else:
        train_rows, eval_rows = _group_holdout_split(rows, holdout_ratio=args.holdout_ratio, seed=args.seed)

    if not train_rows or not eval_rows:
        raise ValueError("Train/eval split is empty. Increase data size or reduce holdout ratio.")

    critic = ProcessCritic(random_state=args.seed)
    critic.fit(train_rows, [float(row[args.target]) for row in train_rows])
    critic.save(args.output)

    eval_preds = critic.predict(eval_rows)
    eval_summary = _evaluate_rows(eval_rows, eval_preds, args.target)
    prediction_rows = []
    for row, pred in zip(eval_rows, eval_preds):
        prediction_rows.append({
            "example_id": row["example_id"],
            "module": row["module"],
            "question_id": row.get("question_id"),
            "question": row["question"],
            "action_text": row["action_text"],
            "target": round(float(row[args.target]), 4),
            "prediction": round(float(pred), 4),
            "abs_error": round(abs(float(pred) - float(row[args.target])), 4),
        })

    meta = {
        "manifest": build_experiment_manifest(
            script_name="scripts/train_process_critic.py",
            qa_path=args.input,
            dataset_name=source_manifest.get("dataset", {}).get("name", "process_dataset"),
            dataset_split=source_manifest.get("dataset", {}).get("split", "custom"),
            limit=data.get("num_examples"),
            effective_questions=len({_group_key(row) for row in train_rows}),
            seed=args.seed,
            prompt_version=source_manifest.get("reproducibility", {}).get("prompt_version", "v1"),
            router_model_path=args.output,
            settings={
                "target": args.target,
                "holdout_ratio": args.holdout_ratio,
                "eval_input": args.eval_input,
                "modules": args.modules,
            },
        ),
        "source_process_manifest": source_manifest,
        "train_counts": {
            "num_examples": len(train_rows),
            "modules": dict(Counter(row["module"] for row in train_rows)),
            "budget_modes": dict(Counter(row.get("budget_mode", "medium") for row in train_rows)),
        },
        "eval_counts": {
            "num_examples": len(eval_rows),
            "modules": dict(Counter(row["module"] for row in eval_rows)),
            "budget_modes": dict(Counter(row.get("budget_mode", "medium") for row in eval_rows)),
        },
        "evaluation": eval_summary,
        "predictions": prediction_rows[:200],
    }
    write_json(f"{args.output}.meta.json", meta)

    print("=== Process critic evaluation ===")
    print("overall:", eval_summary["overall"])
    for module, stats in sorted(eval_summary["per_module"].items()):
        print(module, stats)
    print(f"Saved {args.output}")
    print(f"Saved {args.output}.meta.json")


if __name__ == "__main__":
    main()
