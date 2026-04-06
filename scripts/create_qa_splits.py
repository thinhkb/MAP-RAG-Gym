from __future__ import annotations

import argparse
from pathlib import Path

from map_rag_gym.utils.dataset import normalize_qa_records, split_qa_records
from map_rag_gym.utils.io import read_json, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True, help="Path to a QA JSON list.")
    ap.add_argument("--out_dir", required=True, help="Output directory for train/val/test files.")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    args = ap.parse_args()

    qa = normalize_qa_records(read_json(args.qa))
    splits = split_qa_records(
        qa,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in splits.items():
        write_json(str(out_dir / f"{split_name}.json"), rows)

    write_json(
        str(out_dir / "manifest.json"),
        {
            "source_qa": str(Path(args.qa).resolve()),
            "seed": args.seed,
            "ratios": {
                "train": args.train_ratio,
                "val": args.val_ratio,
                "test": args.test_ratio,
            },
            "counts": {name: len(rows) for name, rows in splits.items()},
        },
    )
    print(f"Saved splits to {out_dir}")


if __name__ == "__main__":
    main()
