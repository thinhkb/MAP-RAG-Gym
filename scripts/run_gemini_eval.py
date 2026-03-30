from __future__ import annotations

import argparse

from map_rag_gym.core.pipeline import MAPRAGGym
from map_rag_gym.router.rule_based import RuleBasedRouter
from map_rag_gym.utils.io import read_json, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--qa", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--model", default="gemini-2.0-flash")
    ap.add_argument("--out", default="outputs/gemini_eval.json")
    args = ap.parse_args()

    pipe = MAPRAGGym(args.corpus, llm_provider="gemini", llm_model=args.model)
    router = RuleBasedRouter()
    qa = read_json(args.qa)[: args.limit]
    runs = []
    for item in qa:
        decision = router.decide(item["question"])
        res = pipe.run(item["question"], item["answer"], decision.workflow_id, planner_reason=decision.reason)
        runs.append(res.to_dict())
        print(decision.workflow_id, item["question"], res.final_answer, res.final_scores)
    write_json(args.out, runs)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
