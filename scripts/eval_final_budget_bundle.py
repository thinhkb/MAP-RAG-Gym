from __future__ import annotations

import argparse
from collections import Counter
from statistics import mean

from map_rag_gym.core.pipeline import MAPRAGGym
from map_rag_gym.evaluation.heuristics import UTILITY_CONFIG, get_utility_profile
from map_rag_gym.router.bandit import BanditRouter
from map_rag_gym.retrieval.bm25 import LocalBM25Retriever
from map_rag_gym.retrieval.policy import parse_workflow_retriever_overrides
from map_rag_gym.router.hybrid import HybridRouter
from map_rag_gym.router.learned import LearnedRouter
from map_rag_gym.router.meta import MetaRouterGate, MetaRouterPolicy
from map_rag_gym.router.rule_based import RuleBasedRouter
from map_rag_gym.utils.dataset import normalize_qa_records
from map_rag_gym.utils.experiment import build_experiment_manifest, set_global_seed
from map_rag_gym.utils.io import read_json, write_json


def _avg(values: list[float]) -> float:
    return round(mean(values), 4) if values else 0.0


def _parse_critic_model_overrides(entries: list[str] | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for raw_entry in entries or []:
        if "=" not in raw_entry:
            raise ValueError(f"Invalid critic override '{raw_entry}'. Expected MODULE=path.")
        module_name, model_path = raw_entry.split("=", 1)
        module_name = module_name.strip().upper()
        model_path = model_path.strip()
        if not module_name or not model_path:
            raise ValueError(f"Invalid critic override '{raw_entry}'. Expected MODULE=path.")
        overrides[module_name] = model_path
    return overrides


def _summarize(runs: list[dict]) -> dict:
    return {
        "num_runs": len(runs),
        "avg_utility": _avg([r["final_scores"].get("utility_total", 0.0) for r in runs]),
        "avg_em": _avg([r["final_scores"].get("em", 0.0) for r in runs]),
        "avg_f1_proxy": _avg([r["final_scores"].get("f1_proxy", 0.0) for r in runs]),
        "avg_process_score": _avg([r["final_scores"].get("process_score", 0.0) for r in runs]),
        "avg_tokens": _avg([r["total_cost"].get("tokens", 0.0) for r in runs]),
        "avg_retrieval_calls": _avg([r["total_cost"].get("retrieval_calls", 0.0) for r in runs]),
        "avg_latency_ms": _avg([r["total_cost"].get("latency_ms", 0.0) for r in runs]),
        "workflow_counts": dict(Counter(r["workflow_id"] for r in runs)),
    }


def _critic_candidate_counts(workflow_id: str, n_candidates: int) -> dict[str, int]:
    workflow_key = str(workflow_id).upper()
    if n_candidates <= 1 or workflow_key in {"W1", "W3"}:
        return {"QR": 1, "AG": 1}
    return {"QR": n_candidates, "AG": n_candidates}


def _run_selected_method(
    method_name: str,
    question: str,
    answer: str,
    budget_mode: str,
    *,
    pipe: MAPRAGGym,
    pipe_critic: MAPRAGGym | None,
    rule_based: RuleBasedRouter,
    learned: LearnedRouter,
    bandit_router: BanditRouter | None,
    hybrid_router: HybridRouter | None,
    meta_router: MetaRouterPolicy | None,
    n_candidates: int,
    critic_n_candidates: int,
    bandit_gate_baseline_workflow: str,
    bandit_gate_min_advantage: float,
    bandit_gate_min_confidence: float,
    bandit_gate_allowed_workflows: list[str],
) -> dict:
    active_pipe = pipe_critic if method_name.endswith("_critic") and pipe_critic else pipe
    active_module_candidate_counts = None
    active_n_candidates = critic_n_candidates if method_name.endswith("_critic") else n_candidates

    if method_name.startswith("fixed_"):
        workflow_id = method_name.replace("fixed_", "", 1).replace("_critic", "")
        if method_name.endswith("_critic"):
            active_module_candidate_counts = _critic_candidate_counts(workflow_id, active_n_candidates)
        run = active_pipe.run(
            question,
            answer,
            workflow_id,
            planner_reason=f"bundle:{method_name}",
            n_candidates=active_n_candidates,
            budget_mode=budget_mode,
            module_candidate_counts=active_module_candidate_counts,
        )
        return run.to_dict()

    if method_name == "rule_based" or method_name == "rule_based_critic":
        decision = rule_based.decide(question, budget_mode=budget_mode)
        if method_name.endswith("_critic"):
            active_module_candidate_counts = _critic_candidate_counts(decision.workflow_id, active_n_candidates)
        run = active_pipe.run(
            question,
            answer,
            decision.workflow_id,
            planner_reason=f"bundle:{decision.reason}",
            n_candidates=active_n_candidates,
            budget_mode=budget_mode,
            module_candidate_counts=active_module_candidate_counts,
        )
        payload = run.to_dict()
        payload.setdefault("metadata", {})["router_confidence"] = decision.confidence
        return payload

    if method_name == "learned_router" or method_name == "learned_router_critic":
        workflow_id, confidence = learned.predict(question, budget_mode=budget_mode)
        if method_name.endswith("_critic"):
            active_module_candidate_counts = _critic_candidate_counts(workflow_id, active_n_candidates)
        run = active_pipe.run(
            question,
            answer,
            workflow_id,
            planner_reason=f"bundle:learned({confidence:.4f})",
            n_candidates=active_n_candidates,
            budget_mode=budget_mode,
            module_candidate_counts=active_module_candidate_counts,
        )
        payload = run.to_dict()
        payload.setdefault("metadata", {})["router_confidence"] = confidence
        return payload

    if method_name == "hybrid_router" or method_name == "hybrid_router_critic":
        if hybrid_router is None:
            raise ValueError("Hybrid router requested but no hybrid configuration was provided.")
        decision = hybrid_router.decide(question, budget_mode=budget_mode)
        if method_name.endswith("_critic"):
            active_module_candidate_counts = _critic_candidate_counts(decision.workflow_id, active_n_candidates)
        run = active_pipe.run(
            question,
            answer,
            decision.workflow_id,
            planner_reason=f"bundle:{decision.reason}",
            n_candidates=active_n_candidates,
            budget_mode=budget_mode,
            module_candidate_counts=active_module_candidate_counts,
        )
        payload = run.to_dict()
        payload.setdefault("metadata", {})["router_confidence"] = decision.confidence
        return payload

    if method_name == "bandit_router" or method_name == "bandit_router_critic":
        if bandit_router is None:
            raise ValueError("Bandit router requested but no bandit router model was provided.")
        workflow_id, confidence, scores = bandit_router.predict_with_scores(question, budget_mode=budget_mode)
        if method_name.endswith("_critic"):
            active_module_candidate_counts = _critic_candidate_counts(workflow_id, active_n_candidates)
        run = active_pipe.run(
            question,
            answer,
            workflow_id,
            planner_reason=f"bundle:bandit({confidence:.4f})",
            n_candidates=active_n_candidates,
            budget_mode=budget_mode,
            module_candidate_counts=active_module_candidate_counts,
        )
        payload = run.to_dict()
        payload.setdefault("metadata", {})["router_confidence"] = confidence
        payload["metadata"]["bandit_scores"] = {wf: round(score, 4) for wf, score in scores.items()}
        return payload

    if method_name == "gated_bandit_router" or method_name == "gated_bandit_router_critic":
        if bandit_router is None:
            raise ValueError("Gated bandit router requested but no bandit router model was provided.")
        workflow_id, confidence, scores, gate_meta = bandit_router.predict_with_gate(
            question,
            budget_mode=budget_mode,
            baseline_workflow=bandit_gate_baseline_workflow,
            minimum_advantage=bandit_gate_min_advantage,
            minimum_confidence=bandit_gate_min_confidence,
            allowed_switch_workflows=bandit_gate_allowed_workflows,
        )
        if method_name.endswith("_critic"):
            active_module_candidate_counts = _critic_candidate_counts(workflow_id, active_n_candidates)
        run = active_pipe.run(
            question,
            answer,
            workflow_id,
            planner_reason=f"bundle:gated_bandit({confidence:.4f})",
            n_candidates=active_n_candidates,
            budget_mode=budget_mode,
            module_candidate_counts=active_module_candidate_counts,
        )
        payload = run.to_dict()
        payload.setdefault("metadata", {})["router_confidence"] = confidence
        payload["metadata"]["bandit_scores"] = {wf: round(score, 4) for wf, score in scores.items()}
        payload["metadata"]["bandit_gate"] = gate_meta
        return payload

    if method_name == "meta_router" or method_name == "meta_router_critic":
        if meta_router is None:
            raise ValueError("Meta router requested but no meta router model was provided.")
        decision = meta_router.decide(question, budget_mode=budget_mode)
        if method_name.endswith("_critic"):
            active_module_candidate_counts = _critic_candidate_counts(decision.workflow_id, active_n_candidates)
        run = active_pipe.run(
            question,
            answer,
            decision.workflow_id,
            planner_reason=f"bundle:{decision.reason}",
            n_candidates=active_n_candidates,
            budget_mode=budget_mode,
            module_candidate_counts=active_module_candidate_counts,
        )
        payload = run.to_dict()
        payload.setdefault("metadata", {})["router_confidence"] = decision.confidence
        return payload

    raise ValueError(f"Unsupported method '{method_name}' in policy bundle.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--qa", required=True)
    ap.add_argument("--policy_bundle", required=True)
    ap.add_argument("--router_model", required=True)
    ap.add_argument("--llm_provider", default="dummy")
    ap.add_argument("--llm_model", default=None)
    ap.add_argument("--retriever_type", default="bm25", choices=["bm25", "tfidf", "hybrid"])
    ap.add_argument("--retriever_bm25_weight", type=float, default=0.5)
    ap.add_argument("--workflow_retriever_overrides", nargs="+", default=[])
    ap.add_argument("--critic_model", default=None)
    ap.add_argument("--critic_modules", nargs="+", default=["QR", "AG"])
    ap.add_argument("--critic_model_overrides", nargs="+", default=[])
    ap.add_argument("--hybrid_min_confidence", type=float, default=None)
    ap.add_argument("--hybrid_low_cost_confidence", type=float, default=None)
    ap.add_argument("--hybrid_low_cost_workflows", nargs="+", default=["W1"])
    ap.add_argument("--bandit_router_model", default=None)
    ap.add_argument("--bandit_gate_baseline_workflow", default="W3")
    ap.add_argument("--bandit_gate_min_advantage", type=float, default=0.0)
    ap.add_argument("--bandit_gate_min_confidence", type=float, default=0.0)
    ap.add_argument("--bandit_gate_allowed_workflows", nargs="+", default=[])
    ap.add_argument("--meta_router_model", default=None)
    ap.add_argument("--budget_modes", nargs="+", default=None, help="Subset of budget modes from the bundle.")
    ap.add_argument("--dataset_name", default=None)
    ap.add_argument("--dataset_split", default=None)
    ap.add_argument("--prompt_version", default="v1")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--n_candidates", type=int, default=1)
    ap.add_argument("--critic_n_candidates", type=int, default=3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="outputs/final_budget_bundle_eval.json")
    args = ap.parse_args()

    set_global_seed(args.seed)
    workflow_retriever_overrides = parse_workflow_retriever_overrides(args.workflow_retriever_overrides)
    critic_model_overrides = _parse_critic_model_overrides(args.critic_model_overrides)
    qa = normalize_qa_records(read_json(args.qa))[: args.limit]
    bundle = read_json(args.policy_bundle)
    budget_policies = bundle.get("budget_policies", {})
    selected_budget_modes = [mode.lower() for mode in (args.budget_modes or sorted(budget_policies))]

    pipe = MAPRAGGym(
        args.corpus,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        retriever_type=args.retriever_type,
        retriever_bm25_weight=args.retriever_bm25_weight,
        workflow_retriever_overrides=workflow_retriever_overrides,
    )
    pipe_critic = (
        MAPRAGGym(
            args.corpus,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            retriever_type=args.retriever_type,
            retriever_bm25_weight=args.retriever_bm25_weight,
            workflow_retriever_overrides=workflow_retriever_overrides,
            critic_model_path=args.critic_model,
            critic_modules=args.critic_modules,
            critic_model_overrides=critic_model_overrides,
        )
        if args.critic_model or critic_model_overrides
        else None
    )
    learned = LearnedRouter(random_state=args.seed)
    learned.load(args.router_model)
    rule_based = RuleBasedRouter()
    hybrid_router = (
        HybridRouter(
            learned_router=learned,
            rule_router=rule_based,
            min_confidence=args.hybrid_min_confidence,
            low_cost_workflow_confidence=args.hybrid_low_cost_confidence,
            low_cost_workflows=args.hybrid_low_cost_workflows,
        )
        if args.hybrid_min_confidence is not None and args.hybrid_low_cost_confidence is not None
        else None
    )
    bandit_router = None
    bandit_router_cache: dict[str, BanditRouter] = {}
    if args.bandit_router_model:
        bandit_router = BanditRouter(random_state=args.seed)
        bandit_router.load(args.bandit_router_model)
        bandit_router.attach_probe_retriever(LocalBM25Retriever(args.corpus))
        bandit_router_cache[args.bandit_router_model] = bandit_router
    meta_router = None
    if args.meta_router_model:
        gate = MetaRouterGate(random_state=args.seed)
        gate.load(args.meta_router_model)
        meta_router = MetaRouterPolicy(gate=gate, learned_router=learned, rule_router=rule_based)

    manifest = build_experiment_manifest(
        script_name="scripts/eval_final_budget_bundle.py",
        qa_path=args.qa,
        corpus_path=args.corpus,
        llm_provider=pipe.llm_provider,
        llm_model=pipe.llm_model,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        limit=args.limit,
        effective_questions=len(qa),
        seed=args.seed,
        prompt_version=args.prompt_version,
        router_model_path=args.router_model,
        utility_config=UTILITY_CONFIG,
        settings={
            "policy_bundle": args.policy_bundle,
            "budget_modes": selected_budget_modes,
            "n_candidates": args.n_candidates,
            "critic_n_candidates": args.critic_n_candidates,
            "retriever_type": args.retriever_type,
            "retriever_bm25_weight": args.retriever_bm25_weight,
            "workflow_retriever_overrides": workflow_retriever_overrides,
            "critic_model": args.critic_model,
            "critic_modules": args.critic_modules if args.critic_model else [],
            "critic_model_overrides": critic_model_overrides,
            "hybrid_min_confidence": args.hybrid_min_confidence,
            "hybrid_low_cost_confidence": args.hybrid_low_cost_confidence,
            "hybrid_low_cost_workflows": args.hybrid_low_cost_workflows if hybrid_router else [],
            "bandit_router_model": args.bandit_router_model,
            "bandit_gate_baseline_workflow": args.bandit_gate_baseline_workflow,
            "bandit_gate_min_advantage": args.bandit_gate_min_advantage,
            "bandit_gate_min_confidence": args.bandit_gate_min_confidence,
            "bandit_gate_allowed_workflows": args.bandit_gate_allowed_workflows,
            "meta_router_model": args.meta_router_model,
        },
    )

    budget_results: dict[str, dict] = {}
    for budget_mode in selected_budget_modes:
        policy = budget_policies.get(budget_mode)
        if not policy:
            raise ValueError(f"Budget mode '{budget_mode}' not found in {args.policy_bundle}.")
        method_name = policy.get("recommended_method")
        if not method_name:
            raise ValueError(f"Budget mode '{budget_mode}' has no recommended_method in bundle.")
        router_settings = policy.get("router_settings", {})
        budget_bandit_gate_baseline_workflow = router_settings.get("bandit_gate_baseline_workflow", args.bandit_gate_baseline_workflow)
        budget_bandit_gate_min_advantage = float(router_settings.get("bandit_gate_min_advantage", args.bandit_gate_min_advantage))
        budget_bandit_gate_min_confidence = float(router_settings.get("bandit_gate_min_confidence", args.bandit_gate_min_confidence))
        budget_bandit_gate_allowed_workflows = list(router_settings.get("bandit_gate_allowed_workflows", args.bandit_gate_allowed_workflows))

        budget_bandit_router_model = router_settings.get("bandit_router_model", args.bandit_router_model)
        active_bandit_router = None
        if budget_bandit_router_model:
            if budget_bandit_router_model not in bandit_router_cache:
                router = BanditRouter(random_state=args.seed)
                router.load(budget_bandit_router_model)
                router.attach_probe_retriever(LocalBM25Retriever(args.corpus))
                bandit_router_cache[budget_bandit_router_model] = router
            active_bandit_router = bandit_router_cache[budget_bandit_router_model]

        runs = []
        per_question = []
        for item in qa:
            payload = _run_selected_method(
                method_name,
                item["question"],
                item["answer"],
                budget_mode,
                pipe=pipe,
                pipe_critic=pipe_critic,
                rule_based=rule_based,
                learned=learned,
                bandit_router=active_bandit_router,
                hybrid_router=hybrid_router,
                meta_router=meta_router,
                n_candidates=args.n_candidates,
                critic_n_candidates=args.critic_n_candidates,
                bandit_gate_baseline_workflow=budget_bandit_gate_baseline_workflow,
                bandit_gate_min_advantage=budget_bandit_gate_min_advantage,
                bandit_gate_min_confidence=budget_bandit_gate_min_confidence,
                bandit_gate_allowed_workflows=budget_bandit_gate_allowed_workflows,
            )
            payload.setdefault("metadata", {})["question_id"] = item["id"]
            payload["metadata"]["dataset_split"] = manifest["dataset"]["split"]
            payload["metadata"]["budget_mode"] = budget_mode
            runs.append(payload)
            per_question.append(
                {
                    "question_id": item["id"],
                    "question": item["question"],
                    "gold_answer": item["answer"],
                    "selected_method": method_name,
                    "workflow_id": payload["workflow_id"],
                    "utility_total": payload["final_scores"]["utility_total"],
                    "em": payload["final_scores"]["em"],
                    "f1_proxy": payload["final_scores"]["f1_proxy"],
                }
            )

        summary = _summarize(runs)
        budget_results[budget_mode] = {
            "recommended_method": method_name,
            "constraints": policy.get("constraints", {}),
            "router_settings": router_settings,
            "source_eval_file": policy.get("source_eval_file"),
            "summary": summary,
            "utility_profile": get_utility_profile(budget_mode),
            "per_question": per_question,
        }
        print(
            f"{budget_mode}: {method_name} | utility={summary['avg_utility']:.4f} | "
            f"tokens={summary['avg_tokens']:.1f} | latency={summary['avg_latency_ms']:.1f}"
        )

    payload = {
        "manifest": manifest,
        "budget_results": budget_results,
    }
    write_json(args.out, payload)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
