from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from statistics import mean

from map_rag_gym.core.pipeline import MAPRAGGym
from map_rag_gym.evaluation.heuristics import get_utility_profile
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

ALL_METHODS = {
    "rule_based",
    "rule_based_critic",
    "learned_router",
    "learned_router_critic",
    "hybrid_router",
    "hybrid_router_critic",
    "bandit_router",
    "bandit_router_critic",
    "gated_bandit_router",
    "gated_bandit_router_critic",
    "meta_router",
    "meta_router_critic",
}


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


def _avg(values: list[float]) -> float:
    return round(mean(values), 4) if values else 0.0


def _normalize_methods(raw_methods: list[str] | None, fixed_workflows: list[str]) -> set[str]:
    if not raw_methods:
        return set()
    normalized: set[str] = set()
    valid = ALL_METHODS | {f"fixed_{wf}" for wf in fixed_workflows} | {f"fixed_{wf}_critic" for wf in fixed_workflows}
    for method in raw_methods:
        if method not in valid:
            raise ValueError(f"Unknown method '{method}'. Valid options: {sorted(valid)}")
        normalized.add(method)
    return normalized


def _summarize(runs: list[dict]) -> dict:
    if not runs:
        return {
            "num_runs": 0,
            "avg_utility": 0.0,
            "avg_em": 0.0,
            "avg_f1_proxy": 0.0,
            "avg_process_score": 0.0,
            "avg_tokens": 0.0,
            "avg_retrieval_calls": 0.0,
            "avg_latency_ms": 0.0,
            "workflow_counts": {},
        }
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--qa", required=True)
    ap.add_argument("--router_model", required=True)
    ap.add_argument("--llm_provider", default="dummy")
    ap.add_argument("--llm_model", default=None)
    ap.add_argument("--retriever_type", default="bm25", choices=["bm25", "tfidf", "hybrid"])
    ap.add_argument("--retriever_bm25_weight", type=float, default=0.5)
    ap.add_argument("--workflow_retriever_overrides", nargs="+", default=[], help="Optional overrides like W6=hybrid W3=bm25.")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--n_candidates", type=int, default=1)
    ap.add_argument("--critic_n_candidates", type=int, default=3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--critic_model", default=None)
    ap.add_argument("--critic_modules", nargs="+", default=["QR", "AG"])
    ap.add_argument("--critic_model_overrides", nargs="+", default=[], help="Optional per-module overrides such as QR=path RA=path.")
    ap.add_argument("--hybrid_min_confidence", type=float, default=None)
    ap.add_argument("--hybrid_low_cost_confidence", type=float, default=None)
    ap.add_argument("--hybrid_low_cost_workflows", nargs="+", default=["W1"])
    ap.add_argument("--bandit_router_model", default=None)
    ap.add_argument("--bandit_gate_baseline_workflow", default="W3")
    ap.add_argument("--bandit_gate_min_advantage", type=float, default=0.0)
    ap.add_argument("--bandit_gate_min_confidence", type=float, default=0.0)
    ap.add_argument("--bandit_gate_allowed_workflows", nargs="+", default=[])
    ap.add_argument("--meta_router_model", default=None)
    ap.add_argument("--budget_mode", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--dataset_name", default=None)
    ap.add_argument("--dataset_split", default=None)
    ap.add_argument("--prompt_version", default="v1")
    ap.add_argument("--fixed_workflows", nargs="+", default=["W1", "W2", "W3", "W6"])
    ap.add_argument("--methods", nargs="+", default=None, help="Optional subset such as fixed_W3 hybrid_router.")
    ap.add_argument("--out", default="outputs/router_eval.json")
    args = ap.parse_args()

    set_global_seed(args.seed)
    selected_methods = _normalize_methods(args.methods, args.fixed_workflows)
    workflow_retriever_overrides = parse_workflow_retriever_overrides(args.workflow_retriever_overrides)
    critic_model_overrides = _parse_critic_model_overrides(args.critic_model_overrides)
    qa = normalize_qa_records(read_json(args.qa))[: args.limit]
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
    if args.bandit_router_model:
        bandit_router = BanditRouter(random_state=args.seed)
        bandit_router.load(args.bandit_router_model)
        bandit_router.attach_probe_retriever(LocalBM25Retriever(args.corpus))
    meta_router = None
    if args.meta_router_model:
        gate = MetaRouterGate(random_state=args.seed)
        gate.load(args.meta_router_model)
        meta_router = MetaRouterPolicy(gate=gate, learned_router=learned, rule_router=rule_based)
    manifest = build_experiment_manifest(
        script_name="scripts/eval_phase4_router.py",
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
        utility_config=get_utility_profile(args.budget_mode),
        settings={
            "fixed_workflows": args.fixed_workflows,
            "methods": sorted(selected_methods) if selected_methods else [],
            "budget_mode": args.budget_mode,
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

    buckets: dict[str, list[dict]] = defaultdict(list)
    per_question: list[dict] = []

    def wants(method_name: str) -> bool:
        return not selected_methods or method_name in selected_methods

    for item in qa:
        q = item["question"]
        a = item["answer"]
        entry: dict = {"question_id": item["id"], "question": q, "gold_answer": a, "results": {}}

        for wf in args.fixed_workflows:
            method_name = f"fixed_{wf}"
            critic_method_name = f"fixed_{wf}_critic"
            if wants(method_name):
                run = pipe.run(q, a, wf, planner_reason=f"fixed:{wf}", n_candidates=args.n_candidates, budget_mode=args.budget_mode)
                payload = run.to_dict()
                payload.setdefault("metadata", {})["question_id"] = item["id"]
                payload["metadata"]["dataset_split"] = manifest["dataset"]["split"]
                buckets[method_name].append(payload)
                entry["results"][method_name] = {
                    "workflow_id": wf,
                    "utility_total": payload["final_scores"]["utility_total"],
                    "em": payload["final_scores"]["em"],
                    "f1_proxy": payload["final_scores"]["f1_proxy"],
                }
            if pipe_critic and wants(critic_method_name):
                critic_run = pipe_critic.run(
                    q,
                    a,
                    wf,
                    planner_reason=f"fixed:{wf}+critic",
                    n_candidates=args.critic_n_candidates,
                    budget_mode=args.budget_mode,
                    module_candidate_counts=_critic_candidate_counts(wf, args.critic_n_candidates),
                )
                critic_payload = critic_run.to_dict()
                critic_payload.setdefault("metadata", {})["question_id"] = item["id"]
                critic_payload["metadata"]["dataset_split"] = manifest["dataset"]["split"]
                buckets[critic_method_name].append(critic_payload)
                entry["results"][critic_method_name] = {
                    "workflow_id": wf,
                    "utility_total": critic_payload["final_scores"]["utility_total"],
                    "em": critic_payload["final_scores"]["em"],
                    "f1_proxy": critic_payload["final_scores"]["f1_proxy"],
                }

        rb = None
        if wants("rule_based") or wants("rule_based_critic"):
            rb = rule_based.decide(q, budget_mode=args.budget_mode)
        if rb and wants("rule_based"):
            rb_run = pipe.run(q, a, rb.workflow_id, planner_reason=f"rule:{rb.reason}", n_candidates=args.n_candidates, budget_mode=args.budget_mode)
            rb_payload = rb_run.to_dict()
            rb_payload.setdefault("metadata", {})["question_id"] = item["id"]
            rb_payload["metadata"]["dataset_split"] = manifest["dataset"]["split"]
            buckets["rule_based"].append(rb_payload)
            entry["results"]["rule_based"] = {
                "workflow_id": rb.workflow_id,
                "utility_total": rb_payload["final_scores"]["utility_total"],
                "em": rb_payload["final_scores"]["em"],
                "f1_proxy": rb_payload["final_scores"]["f1_proxy"],
            }
        if rb and pipe_critic and wants("rule_based_critic"):
            rb_critic_run = pipe_critic.run(
                q,
                a,
                rb.workflow_id,
                planner_reason=f"rule+critic:{rb.reason}",
                n_candidates=args.critic_n_candidates,
                budget_mode=args.budget_mode,
                module_candidate_counts=_critic_candidate_counts(rb.workflow_id, args.critic_n_candidates),
            )
            rb_critic_payload = rb_critic_run.to_dict()
            rb_critic_payload.setdefault("metadata", {})["question_id"] = item["id"]
            rb_critic_payload["metadata"]["dataset_split"] = manifest["dataset"]["split"]
            buckets["rule_based_critic"].append(rb_critic_payload)
            entry["results"]["rule_based_critic"] = {
                "workflow_id": rb.workflow_id,
                "utility_total": rb_critic_payload["final_scores"]["utility_total"],
                "em": rb_critic_payload["final_scores"]["em"],
                "f1_proxy": rb_critic_payload["final_scores"]["f1_proxy"],
            }

        pred = None
        prob = None
        if any(
            wants(method_name)
            for method_name in (
                "learned_router",
                "learned_router_critic",
                "hybrid_router",
                "hybrid_router_critic",
                "bandit_router",
                "bandit_router_critic",
                "gated_bandit_router",
                "gated_bandit_router_critic",
                "meta_router",
                "meta_router_critic",
            )
        ):
            pred, prob = learned.predict(q, budget_mode=args.budget_mode)

        if pred is not None and prob is not None and wants("learned_router"):
            lr_run = pipe.run(q, a, pred, planner_reason=f"learned:{prob:.4f}", n_candidates=args.n_candidates, budget_mode=args.budget_mode)
            lr_payload = lr_run.to_dict()
            lr_payload.setdefault("metadata", {})["router_confidence"] = prob
            lr_payload["metadata"]["question_id"] = item["id"]
            lr_payload["metadata"]["dataset_split"] = manifest["dataset"]["split"]
            buckets["learned_router"].append(lr_payload)
            entry["results"]["learned_router"] = {
                "workflow_id": pred,
                "confidence": round(prob, 4),
                "utility_total": lr_payload["final_scores"]["utility_total"],
                "em": lr_payload["final_scores"]["em"],
                "f1_proxy": lr_payload["final_scores"]["f1_proxy"],
            }
        if pred is not None and prob is not None and pipe_critic and wants("learned_router_critic"):
            lr_critic_run = pipe_critic.run(
                q,
                a,
                pred,
                planner_reason=f"learned+critic:{prob:.4f}",
                n_candidates=args.critic_n_candidates,
                budget_mode=args.budget_mode,
                module_candidate_counts=_critic_candidate_counts(pred, args.critic_n_candidates),
            )
            lr_critic_payload = lr_critic_run.to_dict()
            lr_critic_payload.setdefault("metadata", {})["router_confidence"] = prob
            lr_critic_payload["metadata"]["question_id"] = item["id"]
            lr_critic_payload["metadata"]["dataset_split"] = manifest["dataset"]["split"]
            buckets["learned_router_critic"].append(lr_critic_payload)
            entry["results"]["learned_router_critic"] = {
                "workflow_id": pred,
                "confidence": round(prob, 4),
                "utility_total": lr_critic_payload["final_scores"]["utility_total"],
                "em": lr_critic_payload["final_scores"]["em"],
                "f1_proxy": lr_critic_payload["final_scores"]["f1_proxy"],
            }
        if hybrid_router and (wants("hybrid_router") or wants("hybrid_router_critic")):
            hd = hybrid_router.decide(q, budget_mode=args.budget_mode)
            if wants("hybrid_router"):
                hybrid_run = pipe.run(q, a, hd.workflow_id, planner_reason=hd.reason, n_candidates=args.n_candidates, budget_mode=args.budget_mode)
                hybrid_payload = hybrid_run.to_dict()
                hybrid_payload.setdefault("metadata", {})["question_id"] = item["id"]
                hybrid_payload["metadata"]["dataset_split"] = manifest["dataset"]["split"]
                hybrid_payload["metadata"]["hybrid_confidence"] = hd.confidence
                buckets["hybrid_router"].append(hybrid_payload)
                entry["results"]["hybrid_router"] = {
                    "workflow_id": hd.workflow_id,
                    "confidence": round(hd.confidence, 4),
                    "utility_total": hybrid_payload["final_scores"]["utility_total"],
                    "em": hybrid_payload["final_scores"]["em"],
                    "f1_proxy": hybrid_payload["final_scores"]["f1_proxy"],
                }
            if pipe_critic and wants("hybrid_router_critic"):
                hybrid_critic_run = pipe_critic.run(
                    q,
                    a,
                    hd.workflow_id,
                    planner_reason=f"{hd.reason}+critic",
                    n_candidates=args.critic_n_candidates,
                    budget_mode=args.budget_mode,
                    module_candidate_counts=_critic_candidate_counts(hd.workflow_id, args.critic_n_candidates),
                )
                hybrid_critic_payload = hybrid_critic_run.to_dict()
                hybrid_critic_payload.setdefault("metadata", {})["question_id"] = item["id"]
                hybrid_critic_payload["metadata"]["dataset_split"] = manifest["dataset"]["split"]
                hybrid_critic_payload["metadata"]["hybrid_confidence"] = hd.confidence
                buckets["hybrid_router_critic"].append(hybrid_critic_payload)
                entry["results"]["hybrid_router_critic"] = {
                    "workflow_id": hd.workflow_id,
                    "confidence": round(hd.confidence, 4),
                    "utility_total": hybrid_critic_payload["final_scores"]["utility_total"],
                    "em": hybrid_critic_payload["final_scores"]["em"],
                    "f1_proxy": hybrid_critic_payload["final_scores"]["f1_proxy"],
                }
        if bandit_router and (wants("bandit_router") or wants("bandit_router_critic")):
            bandit_workflow, bandit_confidence, bandit_scores = bandit_router.predict_with_scores(
                q,
                budget_mode=args.budget_mode,
            )
            if wants("bandit_router"):
                bandit_run = pipe.run(q, a, bandit_workflow, planner_reason=f"bandit:{bandit_confidence:.4f}", n_candidates=args.n_candidates, budget_mode=args.budget_mode)
                bandit_payload = bandit_run.to_dict()
                bandit_payload.setdefault("metadata", {})["question_id"] = item["id"]
                bandit_payload["metadata"]["dataset_split"] = manifest["dataset"]["split"]
                bandit_payload["metadata"]["bandit_confidence"] = bandit_confidence
                bandit_payload["metadata"]["bandit_scores"] = {workflow_id: round(score, 4) for workflow_id, score in bandit_scores.items()}
                buckets["bandit_router"].append(bandit_payload)
                entry["results"]["bandit_router"] = {
                    "workflow_id": bandit_workflow,
                    "confidence": round(bandit_confidence, 4),
                    "utility_total": bandit_payload["final_scores"]["utility_total"],
                    "em": bandit_payload["final_scores"]["em"],
                    "f1_proxy": bandit_payload["final_scores"]["f1_proxy"],
                }
            if pipe_critic and wants("bandit_router_critic"):
                bandit_critic_run = pipe_critic.run(
                    q,
                    a,
                    bandit_workflow,
                    planner_reason=f"bandit+critic:{bandit_confidence:.4f}",
                    n_candidates=args.critic_n_candidates,
                    budget_mode=args.budget_mode,
                    module_candidate_counts=_critic_candidate_counts(bandit_workflow, args.critic_n_candidates),
                )
                bandit_critic_payload = bandit_critic_run.to_dict()
                bandit_critic_payload.setdefault("metadata", {})["question_id"] = item["id"]
                bandit_critic_payload["metadata"]["dataset_split"] = manifest["dataset"]["split"]
                bandit_critic_payload["metadata"]["bandit_confidence"] = bandit_confidence
                bandit_critic_payload["metadata"]["bandit_scores"] = {workflow_id: round(score, 4) for workflow_id, score in bandit_scores.items()}
                buckets["bandit_router_critic"].append(bandit_critic_payload)
                entry["results"]["bandit_router_critic"] = {
                    "workflow_id": bandit_workflow,
                    "confidence": round(bandit_confidence, 4),
                    "utility_total": bandit_critic_payload["final_scores"]["utility_total"],
                    "em": bandit_critic_payload["final_scores"]["em"],
                    "f1_proxy": bandit_critic_payload["final_scores"]["f1_proxy"],
                }
        if bandit_router and (wants("gated_bandit_router") or wants("gated_bandit_router_critic")):
            gated_workflow, gated_confidence, gated_scores, gated_meta = bandit_router.predict_with_gate(
                q,
                budget_mode=args.budget_mode,
                baseline_workflow=args.bandit_gate_baseline_workflow,
                minimum_advantage=args.bandit_gate_min_advantage,
                minimum_confidence=args.bandit_gate_min_confidence,
                allowed_switch_workflows=args.bandit_gate_allowed_workflows,
            )
            if wants("gated_bandit_router"):
                gated_run = pipe.run(
                    q,
                    a,
                    gated_workflow,
                    planner_reason=f"gated_bandit:{gated_confidence:.4f}",
                    n_candidates=args.n_candidates,
                    budget_mode=args.budget_mode,
                )
                gated_payload = gated_run.to_dict()
                gated_payload.setdefault("metadata", {})["question_id"] = item["id"]
                gated_payload["metadata"]["dataset_split"] = manifest["dataset"]["split"]
                gated_payload["metadata"]["bandit_confidence"] = gated_confidence
                gated_payload["metadata"]["bandit_scores"] = {workflow_id: round(score, 4) for workflow_id, score in gated_scores.items()}
                gated_payload["metadata"]["bandit_gate"] = gated_meta
                buckets["gated_bandit_router"].append(gated_payload)
                entry["results"]["gated_bandit_router"] = {
                    "workflow_id": gated_workflow,
                    "confidence": round(gated_confidence, 4),
                    "utility_total": gated_payload["final_scores"]["utility_total"],
                    "em": gated_payload["final_scores"]["em"],
                    "f1_proxy": gated_payload["final_scores"]["f1_proxy"],
                }
            if pipe_critic and wants("gated_bandit_router_critic"):
                gated_critic_run = pipe_critic.run(
                    q,
                    a,
                    gated_workflow,
                    planner_reason=f"gated_bandit+critic:{gated_confidence:.4f}",
                    n_candidates=args.critic_n_candidates,
                    budget_mode=args.budget_mode,
                    module_candidate_counts=_critic_candidate_counts(gated_workflow, args.critic_n_candidates),
                )
                gated_critic_payload = gated_critic_run.to_dict()
                gated_critic_payload.setdefault("metadata", {})["question_id"] = item["id"]
                gated_critic_payload["metadata"]["dataset_split"] = manifest["dataset"]["split"]
                gated_critic_payload["metadata"]["bandit_confidence"] = gated_confidence
                gated_critic_payload["metadata"]["bandit_scores"] = {workflow_id: round(score, 4) for workflow_id, score in gated_scores.items()}
                gated_critic_payload["metadata"]["bandit_gate"] = gated_meta
                buckets["gated_bandit_router_critic"].append(gated_critic_payload)
                entry["results"]["gated_bandit_router_critic"] = {
                    "workflow_id": gated_workflow,
                    "confidence": round(gated_confidence, 4),
                    "utility_total": gated_critic_payload["final_scores"]["utility_total"],
                    "em": gated_critic_payload["final_scores"]["em"],
                    "f1_proxy": gated_critic_payload["final_scores"]["f1_proxy"],
                }
        if meta_router and (wants("meta_router") or wants("meta_router_critic")):
            md = meta_router.decide(q, budget_mode=args.budget_mode)
            if wants("meta_router"):
                meta_run = pipe.run(q, a, md.workflow_id, planner_reason=md.reason, n_candidates=args.n_candidates, budget_mode=args.budget_mode)
                meta_payload = meta_run.to_dict()
                meta_payload.setdefault("metadata", {})["question_id"] = item["id"]
                meta_payload["metadata"]["dataset_split"] = manifest["dataset"]["split"]
                meta_payload["metadata"]["meta_confidence"] = md.confidence
                buckets["meta_router"].append(meta_payload)
                entry["results"]["meta_router"] = {
                    "workflow_id": md.workflow_id,
                    "confidence": round(md.confidence, 4),
                    "utility_total": meta_payload["final_scores"]["utility_total"],
                    "em": meta_payload["final_scores"]["em"],
                    "f1_proxy": meta_payload["final_scores"]["f1_proxy"],
                }
            if pipe_critic and wants("meta_router_critic"):
                meta_critic_run = pipe_critic.run(
                    q,
                    a,
                    md.workflow_id,
                    planner_reason=f"{md.reason}+critic",
                    n_candidates=args.critic_n_candidates,
                    budget_mode=args.budget_mode,
                    module_candidate_counts=_critic_candidate_counts(md.workflow_id, args.critic_n_candidates),
                )
                meta_critic_payload = meta_critic_run.to_dict()
                meta_critic_payload.setdefault("metadata", {})["question_id"] = item["id"]
                meta_critic_payload["metadata"]["dataset_split"] = manifest["dataset"]["split"]
                meta_critic_payload["metadata"]["meta_confidence"] = md.confidence
                buckets["meta_router_critic"].append(meta_critic_payload)
                entry["results"]["meta_router_critic"] = {
                    "workflow_id": md.workflow_id,
                    "confidence": round(md.confidence, 4),
                    "utility_total": meta_critic_payload["final_scores"]["utility_total"],
                    "em": meta_critic_payload["final_scores"]["em"],
                    "f1_proxy": meta_critic_payload["final_scores"]["f1_proxy"],
                }

        per_question.append(entry)

    summary = {name: _summarize(runs) for name, runs in buckets.items()}

    print("=== Router evaluation summary ===")
    ranked = sorted(summary.items(), key=lambda kv: kv[1]["avg_utility"], reverse=True)
    for name, stats in ranked:
        print(
            f"{name:14s} | utility={stats['avg_utility']:.4f} | em={stats['avg_em']:.4f} | "
            f"f1={stats['avg_f1_proxy']:.4f} | tokens={stats['avg_tokens']:.1f} | workflows={stats['workflow_counts']}"
        )

    payload = {
        "manifest": manifest,
        "settings": {
            "limit": args.limit,
            "llm_provider": pipe.llm_provider,
            "llm_model": pipe.llm_model,
            "n_candidates": args.n_candidates,
            "critic_n_candidates": args.critic_n_candidates,
            "seed": args.seed,
            "retriever_type": args.retriever_type,
            "retriever_bm25_weight": args.retriever_bm25_weight,
            "workflow_retriever_overrides": workflow_retriever_overrides,
            "fixed_workflows": args.fixed_workflows,
            "methods": sorted(selected_methods) if selected_methods else [],
            "budget_mode": args.budget_mode,
            "router_model": args.router_model,
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
        "summary": summary,
        "per_question": per_question,
    }
    write_json(args.out, payload)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
