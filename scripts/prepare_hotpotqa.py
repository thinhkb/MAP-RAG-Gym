from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from map_rag_gym.utils.dataset import normalize_qa_records, split_qa_records
from map_rag_gym.utils.io import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out_dir", default="data/hotpotqa")
    ap.add_argument("--write_splits", action="store_true")
    ap.add_argument("--split_seed", type=int, default=13)
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--train_count", type=int, default=None)
    ap.add_argument("--val_count", type=int, default=None)
    ap.add_argument("--test_count", type=int, default=None)
    args = ap.parse_args()

    ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split=args.split)
    qa = []
    docs = {}
    for idx, row in enumerate(ds):
        if idx >= args.limit:
            break
        answer = row.get("answer", "")
        question = row.get("question", "")
        qa.append({
            "id": str(idx),
            "question": question,
            "answer": answer,
            "metadata": {"dataset": "hotpotqa", "source_split": args.split},
        })
        context = row.get("context", {})
        titles = context.get("title", []) if isinstance(context, dict) else []
        sentences = context.get("sentences", []) if isinstance(context, dict) else []
        for title, sent_list in zip(titles, sentences):
            text = " ".join(sent_list)
            doc_id = title.replace(" ", "_")
            if doc_id not in docs:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "title": title,
                    "text": text,
                    "metadata": {"source": "hotpotqa", "source_split": args.split},
                }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    qa = normalize_qa_records(qa)
    write_json(str(out / "qa.json"), qa)
    write_json(str(out / "corpus.json"), list(docs.values()))
    print(f"Saved {len(qa)} QA pairs and {len(docs)} docs to {out}")

    if args.write_splits:
        splits = split_qa_records(
            qa,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.split_seed,
            train_count=args.train_count,
            val_count=args.val_count,
            test_count=args.test_count,
        )
        split_dir = out / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        for split_name, rows in splits.items():
            write_json(str(split_dir / f"{split_name}.json"), rows)
        explicit_counts = all(count is not None for count in (args.train_count, args.val_count, args.test_count))
        write_json(
            str(split_dir / "manifest.json"),
            {
                "dataset": "hotpotqa",
                "source_split": args.split,
                "seed": args.split_seed,
                "ratios": {
                    "train": args.train_ratio,
                    "val": args.val_ratio,
                    "test": args.test_ratio,
                },
                "exact_counts": {
                    "train": args.train_count,
                    "val": args.val_count,
                    "test": args.test_count,
                }
                if explicit_counts
                else None,
                "counts": {name: len(rows) for name, rows in splits.items()},
                "unused_count": max(0, len(qa) - sum(len(rows) for rows in splits.values())),
            },
        )
        print(f"Saved deterministic splits to {split_dir}")


if __name__ == "__main__":
    main()
