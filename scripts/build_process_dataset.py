from __future__ import annotations

import argparse
from collections import Counter

from map_rag_gym.core.schemas import Document
from map_rag_gym.critic.data import build_process_dataset
from map_rag_gym.utils.experiment import build_experiment_manifest
from map_rag_gym.utils.io import read_json, write_json


def _load_corpus_lookup(corpus_path: str) -> dict[str, Document]:
    rows = read_json(corpus_path)
    docs = [Document(**row) for row in rows]
    return {doc.doc_id: doc for doc in docs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Rollout JSON produced by scripts/batch_rollout.py")
    ap.add_argument("--corpus", default=None, help="Corpus JSON. Defaults to manifest.paths.corpus from the rollout file.")
    ap.add_argument("--out", default="outputs/process_dataset.json")
    args = ap.parse_args()

    rollout = read_json(args.input)
    source_manifest = rollout.get("manifest", {})
    corpus_path = args.corpus or source_manifest.get("paths", {}).get("corpus")
    if not corpus_path:
        raise ValueError("Missing corpus path. Pass --corpus or use a rollout file with manifest.paths.corpus.")

    corpus_lookup = _load_corpus_lookup(corpus_path)
    payload = build_process_dataset(rollout.get("runs", []), corpus_lookup)
    examples = payload["examples"]

    output = {
        "manifest": build_experiment_manifest(
            script_name="scripts/build_process_dataset.py",
            qa_path=args.input,
            corpus_path=corpus_path,
            llm_provider=source_manifest.get("llm", {}).get("provider"),
            llm_model=source_manifest.get("llm", {}).get("model"),
            dataset_name=source_manifest.get("dataset", {}).get("name", "process_dataset"),
            dataset_split=source_manifest.get("dataset", {}).get("split", "custom"),
            limit=source_manifest.get("dataset", {}).get("limit"),
            effective_questions=source_manifest.get("dataset", {}).get("effective_questions"),
            seed=source_manifest.get("reproducibility", {}).get("seed"),
            prompt_version=source_manifest.get("reproducibility", {}).get("prompt_version", "v1"),
            settings={
                "source_rollout_file": args.input,
                "reward_target": "blended_reward",
                "reward_formula": "0.7 * local_reward + 0.3 * outcome_utility",
            },
        ),
        "source_rollout_manifest": source_manifest,
        "num_runs": len(rollout.get("runs", [])),
        "num_examples": len(examples),
        "module_counts": dict(Counter(row["module"] for row in examples)),
        "module_stats": payload["module_stats"],
        "examples": examples,
    }
    write_json(args.out, output)
    print(f"Saved {len(examples)} process examples to {args.out}")


if __name__ == "__main__":
    main()
