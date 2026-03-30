from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from map_rag_gym.utils.io import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out_dir", default="data/hotpotqa")
    args = ap.parse_args()

    ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split=args.split)
    qa = []
    docs = {}
    for idx, row in enumerate(ds):
        if idx >= args.limit:
            break
        answer = row.get("answer", "")
        question = row.get("question", "")
        qa.append({"id": str(idx), "question": question, "answer": answer})
        context = row.get("context", {})
        titles = context.get("title", []) if isinstance(context, dict) else []
        sentences = context.get("sentences", []) if isinstance(context, dict) else []
        for title, sent_list in zip(titles, sentences):
            text = " ".join(sent_list)
            doc_id = title.replace(" ", "_")
            if doc_id not in docs:
                docs[doc_id] = {"doc_id": doc_id, "title": title, "text": text, "metadata": {"source": "hotpotqa"}}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(str(out / "qa.json"), qa)
    write_json(str(out / "corpus.json"), list(docs.values()))
    print(f"Saved {len(qa)} QA pairs and {len(docs)} docs to {out}")


if __name__ == "__main__":
    main()
