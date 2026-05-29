from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datasets import load_dataset

from map_rag_gym.utils.dataset import normalize_qa_records, split_qa_records
from map_rag_gym.utils.io import write_json


def _sentences_to_text(sentences: Any) -> str:
    if isinstance(sentences, list):
        return " ".join(str(sentence).strip() for sentence in sentences if str(sentence).strip())
    return str(sentences or "").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cmriat/2wikimultihopqa")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--out_dir", default="data/2wikimultihopqa")
    ap.add_argument("--split_seed", type=int, default=13)
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--train_count", type=int, default=420)
    ap.add_argument("--val_count", type=int, default=90)
    ap.add_argument("--test_count", type=int, default=90)
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    qa: list[dict[str, Any]] = []
    docs: dict[str, dict[str, Any]] = {}

    for idx, row in enumerate(ds):
        if idx >= args.limit:
            break

        answers = row.get("golden_answers") or []
        if isinstance(answers, str):
            answers = [answers]
        answer = str(answers[0] if answers else "").strip()
        metadata = dict(row.get("metadata") or {})
        metadata.update(
            {
                "dataset": "2wikimultihopqa",
                "source_dataset": args.dataset,
                "source_split": args.split,
                "golden_answers": [str(item) for item in answers],
            }
        )

        qa.append(
            {
                "id": str(row.get("id") or idx),
                "question": str(row.get("question") or "").strip(),
                "answer": answer,
                "metadata": metadata,
            }
        )

        context = metadata.get("context") or {}
        titles = context.get("title", []) if isinstance(context, dict) else []
        contents = context.get("content", []) if isinstance(context, dict) else []
        for doc_idx, (title, sentences) in enumerate(zip(titles, contents)):
            title = str(title).strip()
            text = _sentences_to_text(sentences)
            if not title or not text:
                continue
            doc_id = title.replace(" ", "_")
            if doc_id in docs and docs[doc_id]["text"] != text:
                doc_id = f"{doc_id}_{row.get('id') or idx}_{doc_idx}"
            docs.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "title": title,
                    "text": text,
                    "metadata": {
                        "source": "2wikimultihopqa",
                        "source_dataset": args.dataset,
                        "source_split": args.split,
                    },
                },
            )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    qa = normalize_qa_records(qa)
    write_json(str(out / "qa.json"), qa)
    write_json(str(out / "corpus.json"), list(docs.values()))

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
    write_json(
        str(split_dir / "manifest.json"),
        {
            "dataset": "2wikimultihopqa",
            "source_dataset": args.dataset,
            "source_split": args.split,
            "seed": args.split_seed,
            "limit": args.limit,
            "exact_counts": {
                "train": args.train_count,
                "val": args.val_count,
                "test": args.test_count,
            },
            "counts": {name: len(rows) for name, rows in splits.items()},
            "unused_count": max(0, len(qa) - sum(len(rows) for rows in splits.values())),
        },
    )
    print(f"Saved {len(qa)} QA pairs and {len(docs)} docs to {out}")
    print(f"Saved deterministic splits to {split_dir}")


if __name__ == "__main__":
    main()
