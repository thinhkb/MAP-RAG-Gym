from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from map_rag_gym.router.budget import ALLOWED_WORKFLOWS_BY_BUDGET, normalize_budget_mode
from map_rag_gym.utils.experiment import build_experiment_manifest
from map_rag_gym.utils.io import read_json, write_json


DEFAULT_ALPHA_BY_BUDGET = {
    "low": 0.1,
    "medium": 1.0,
    "high": 1.0,
}

DEFAULT_PREFERENCE_MARGIN_BY_BUDGET = {
    "low": 0.02,
    "medium": 0.0,
    "high": 0.0,
}

DEFAULT_PREFERRED_WORKFLOWS_BY_BUDGET = {
    "low": ["W3"],
    "medium": [],
    "high": [],
}

DEFAULT_POLICY_METHOD_BY_BUDGET = {
    "low": "gated_bandit_router",
    "medium": "bandit_router",
    "high": "bandit_router",
}

DEFAULT_GATE_BASELINE_WORKFLOW_BY_BUDGET = {
    "low": "W3",
}

DEFAULT_GATE_MIN_ADVANTAGE_BY_BUDGET = {
    "low": 0.2,
}

DEFAULT_GATE_MIN_CONFIDENCE_BY_BUDGET = {
    "low": 0.65,
}

DEFAULT_GATE_ALLOWED_WORKFLOWS_BY_BUDGET = {
    "low": ["W1"],
}

DEFAULT_CONSTRAINTS_BY_BUDGET = {
    "low": {
        "max_tokens": 110.0,
        "max_latency_ms": 900.0,
        "max_retrieval_calls": None,
    },
    "medium": {
        "max_tokens": None,
        "max_latency_ms": None,
        "max_retrieval_calls": None,
    },
    "high": {
        "max_tokens": None,
        "max_latency_ms": None,
        "max_retrieval_calls": None,
    },
}

SUPPORTED_POLICY_METHODS = {"bandit_router", "gated_bandit_router"}


def _parse_key_values(entries: list[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_entry in entries or []:
        if "=" not in raw_entry:
            raise ValueError(f"Invalid entry '{raw_entry}'. Expected KEY=VALUE.")
        key, value = raw_entry.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Invalid entry '{raw_entry}'. Expected KEY=VALUE.")
        parsed[key] = value
    return parsed


def _parse_budget_float_overrides(entries: list[str] | None, defaults: dict[str, float]) -> dict[str, float]:
    values = dict(defaults)
    for key, raw_value in _parse_key_values(entries).items():
        values[normalize_budget_mode(key)] = float(raw_value)
    return values


def _parse_budget_string_overrides(entries: list[str] | None, defaults: dict[str, str]) -> dict[str, str]:
    values = dict(defaults)
    for key, raw_value in _parse_key_values(entries).items():
        values[normalize_budget_mode(key)] = raw_value
    return values


def _parse_budget_workflow_overrides(
    entries: list[str] | None,
    defaults: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    values = {budget: list(workflows) for budget, workflows in (defaults or {}).items()}
    for key, raw_value in _parse_key_values(entries).items():
        workflows = [item.strip().upper() for item in raw_value.split(",") if item.strip()]
        if not workflows:
            raise ValueError(f"Budget workflow override for '{key}' is empty.")
        values[normalize_budget_mode(key)] = workflows
    return values


def _workflow_counts(rollout_path: str) -> Counter[str]:
    data = read_json(rollout_path)
    return Counter(str(run.get("workflow_id", "")).upper() for run in data.get("runs", []))


def _infer_candidate_workflows(
    *,
    budget_mode: str,
    rollout_path: str,
    explicit_workflows: dict[str, list[str]],
) -> list[str]:
    if budget_mode in explicit_workflows:
        return explicit_workflows[budget_mode]
    present_workflows = set(_workflow_counts(rollout_path))
    allowed = ALLOWED_WORKFLOWS_BY_BUDGET[budget_mode]
    workflows = sorted(present_workflows & allowed)
    if len(workflows) < 2:
        raise ValueError(
            f"Need at least two candidate workflows for {budget_mode}; "
            f"found {workflows} in {rollout_path}."
        )
    return workflows


def _run_command(command: list[str], *, dry_run: bool) -> dict[str, Any]:
    printable = " ".join(command)
    if dry_run:
        return {
            "command": printable,
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": "",
            "dry_run": True,
        }
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    return {
        "command": printable,
        "returncode": result.returncode,
        "stdout_tail": "\n".join(result.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(result.stderr.splitlines()[-20:]),
        "dry_run": False,
    }


def _read_holdout_eval(meta_path: str) -> dict[str, Any]:
    meta = read_json(meta_path)
    holdout = meta.get("holdout_policy_eval", {})
    return {
        "avg_policy_utility": holdout.get("avg_policy_utility"),
        "avg_oracle_utility": holdout.get("avg_oracle_utility"),
        "avg_regret": holdout.get("avg_regret"),
        "exact_best_rate": holdout.get("exact_best_rate"),
        "num_questions": holdout.get("num_questions"),
    }


def _policy_payload(
    *,
    budget_mode: str,
    model_path: str,
    source_rollout: str,
    holdout_eval: dict[str, Any],
    recommended_method: str,
    gate_settings: dict[str, Any],
) -> dict[str, Any]:
    router_settings: dict[str, Any] = {
        "bandit_router_model": model_path,
    }
    if recommended_method == "gated_bandit_router":
        router_settings.update(gate_settings)
    return {
        "budget_mode": budget_mode,
        "recommended_method": recommended_method,
        "constraints": dict(DEFAULT_CONSTRAINTS_BY_BUDGET[budget_mode]),
        "source_eval_file": source_rollout,
        "router_settings": router_settings,
        "offline_rl": {
            "stage": "macro_candidate_policy",
            "holdout_policy_eval": holdout_eval,
            "promotion_required": "Evaluate on validation/test and promote only if non-regressive vs frozen bundle.",
        },
    }


def _bundle_payload(policy_paths: dict[str, str], policies: dict[str, dict]) -> dict[str, Any]:
    table = []
    for budget_mode, policy in sorted(policies.items()):
        table.append(
            {
                "budget_mode": budget_mode,
                "recommended_method": policy.get("recommended_method"),
                "constraints": policy.get("constraints", {}),
                "router_settings": policy.get("router_settings", {}),
                "source_eval_file": policy.get("source_eval_file"),
            }
        )
    return {
        "source_policy_files": [policy_paths[budget_mode] for budget_mode in sorted(policy_paths)],
        "source_eval_files": [policies[budget_mode].get("source_eval_file") for budget_mode in sorted(policies)],
        "budget_policies": policies,
        "budget_modes": sorted(policies),
        "policy_table": table,
        "offline_rl": {
            "stage": "candidate_bundle",
            "promotion_required": "Run eval_final_budget_bundle.py before replacing the frozen rl_ready bundle.",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default="outputs/full_system_rl_package.json")
    ap.add_argument("--out_dir", default="outputs/offline_full_system_rl")
    ap.add_argument("--base_router_model", default="outputs/router_hotpot_budget_calibrated.joblib")
    ap.add_argument("--probe_corpus", default="data/hotpotqa_large/corpus.json")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--holdout_ratio", type=float, default=0.15)
    ap.add_argument("--alpha_by_budget", nargs="+", default=[])
    ap.add_argument("--preference_margin_by_budget", nargs="+", default=[])
    ap.add_argument("--preferred_workflows_by_budget", nargs="+", default=[])
    ap.add_argument("--candidate_workflows_by_budget", nargs="+", default=["high=W2,W3"])
    ap.add_argument("--policy_method_by_budget", nargs="+", default=[])
    ap.add_argument("--gate_baseline_workflow_by_budget", nargs="+", default=[])
    ap.add_argument("--gate_min_advantage_by_budget", nargs="+", default=[])
    ap.add_argument("--gate_min_confidence_by_budget", nargs="+", default=[])
    ap.add_argument("--gate_allowed_workflows_by_budget", nargs="+", default=[])
    ap.add_argument("--budget_modes", nargs="+", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    package = read_json(args.package)
    stage = package.get("stage", {})
    if not stage.get("ready_for_offline_full_system_rl") and not args.force:
        raise ValueError(
            f"{args.package} is not ready for offline full-system RL. "
            "Pass --force only for debugging."
        )
    if stage.get("ready_for_online_full_system_rl"):
        raise ValueError("This trainer is for offline RL only; online RL should use a separate guarded loop.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha_by_budget = _parse_budget_float_overrides(args.alpha_by_budget, DEFAULT_ALPHA_BY_BUDGET)
    preference_margin_by_budget = _parse_budget_float_overrides(
        args.preference_margin_by_budget,
        DEFAULT_PREFERENCE_MARGIN_BY_BUDGET,
    )
    preferred_workflows_by_budget = _parse_budget_workflow_overrides(
        args.preferred_workflows_by_budget,
        DEFAULT_PREFERRED_WORKFLOWS_BY_BUDGET,
    )
    candidate_workflows_by_budget = _parse_budget_workflow_overrides(args.candidate_workflows_by_budget)
    policy_method_by_budget = _parse_budget_string_overrides(
        args.policy_method_by_budget,
        DEFAULT_POLICY_METHOD_BY_BUDGET,
    )
    gate_baseline_workflow_by_budget = _parse_budget_string_overrides(
        args.gate_baseline_workflow_by_budget,
        DEFAULT_GATE_BASELINE_WORKFLOW_BY_BUDGET,
    )
    gate_min_advantage_by_budget = _parse_budget_float_overrides(
        args.gate_min_advantage_by_budget,
        DEFAULT_GATE_MIN_ADVANTAGE_BY_BUDGET,
    )
    gate_min_confidence_by_budget = _parse_budget_float_overrides(
        args.gate_min_confidence_by_budget,
        DEFAULT_GATE_MIN_CONFIDENCE_BY_BUDGET,
    )
    gate_allowed_workflows_by_budget = _parse_budget_workflow_overrides(
        args.gate_allowed_workflows_by_budget,
        DEFAULT_GATE_ALLOWED_WORKFLOWS_BY_BUDGET,
    )

    coverage = package.get("macro_layer", {}).get("counterfactual_rollout_coverage", {})
    requested_budgets = [normalize_budget_mode(mode) for mode in (args.budget_modes or sorted(coverage))]
    if not requested_budgets:
        raise ValueError("No budget modes available in package macro coverage.")

    train_runs = {}
    policy_paths: dict[str, str] = {}
    policies: dict[str, dict] = {}
    for budget_mode in requested_budgets:
        rollout_info = coverage.get(budget_mode)
        if not rollout_info:
            raise ValueError(f"Package has no counterfactual rollout coverage for budget '{budget_mode}'.")
        rollout_path = rollout_info.get("path")
        if not rollout_path:
            raise ValueError(f"Package rollout coverage for budget '{budget_mode}' has no path.")
        workflows = _infer_candidate_workflows(
            budget_mode=budget_mode,
            rollout_path=rollout_path,
            explicit_workflows=candidate_workflows_by_budget,
        )
        recommended_method = policy_method_by_budget.get(budget_mode, "bandit_router")
        if recommended_method not in SUPPORTED_POLICY_METHODS:
            raise ValueError(
                f"Unsupported policy method '{recommended_method}' for {budget_mode}. "
                f"Expected one of {sorted(SUPPORTED_POLICY_METHODS)}."
            )
        gate_settings = {}
        if recommended_method == "gated_bandit_router":
            baseline_workflow = str(gate_baseline_workflow_by_budget.get(budget_mode, "W3")).upper()
            if baseline_workflow not in workflows:
                raise ValueError(
                    f"Gate baseline workflow '{baseline_workflow}' for {budget_mode} "
                    f"is not in candidate workflows {workflows}."
                )
            allowed_switch_workflows = gate_allowed_workflows_by_budget.get(
                budget_mode,
                [workflow for workflow in workflows if workflow != baseline_workflow],
            )
            gate_settings = {
                "bandit_gate_baseline_workflow": baseline_workflow,
                "bandit_gate_min_advantage": float(gate_min_advantage_by_budget.get(budget_mode, 0.0)),
                "bandit_gate_min_confidence": float(gate_min_confidence_by_budget.get(budget_mode, 0.0)),
                "bandit_gate_allowed_workflows": [str(workflow).upper() for workflow in allowed_switch_workflows],
            }
        output_model = str(out_dir / f"macro_bandit_{budget_mode}.joblib")
        command = [
            sys.executable,
            "scripts/train_bandit_router.py",
            "--input",
            rollout_path,
            "--output",
            output_model,
            "--budget_mode",
            budget_mode,
            "--allowed_workflows",
            *workflows,
            "--holdout_ratio",
            str(args.holdout_ratio),
            "--alpha",
            str(alpha_by_budget[budget_mode]),
            "--preference_margin",
            str(preference_margin_by_budget[budget_mode]),
            "--seed",
            str(args.seed),
        ]
        preferred_workflows = preferred_workflows_by_budget.get(budget_mode, [])
        if preferred_workflows:
            command.extend(["--preferred_workflows", *preferred_workflows])
        if args.base_router_model:
            command.extend(["--base_router_model", args.base_router_model])
        if args.probe_corpus:
            command.extend(["--probe_corpus", args.probe_corpus])

        run_result = _run_command(command, dry_run=args.dry_run)
        train_runs[budget_mode] = {
            "rollout_path": rollout_path,
            "candidate_workflows": workflows,
            "model_path": output_model,
            "meta_path": f"{output_model}.meta.json",
            "alpha": alpha_by_budget[budget_mode],
            "preference_margin": preference_margin_by_budget[budget_mode],
            "preferred_workflows": preferred_workflows,
            "recommended_method": recommended_method,
            "gate_settings": gate_settings,
            "subprocess": run_result,
        }
        if run_result["returncode"] not in (0, None):
            raise RuntimeError(
                f"Training failed for budget '{budget_mode}' with return code {run_result['returncode']}.\n"
                f"stderr tail:\n{run_result['stderr_tail']}"
            )
        holdout_eval = {} if args.dry_run else _read_holdout_eval(f"{output_model}.meta.json")
        policy = _policy_payload(
            budget_mode=budget_mode,
            model_path=output_model,
            source_rollout=rollout_path,
            holdout_eval=holdout_eval,
            recommended_method=recommended_method,
            gate_settings=gate_settings,
        )
        policy_path = str(out_dir / f"budget_policy_{budget_mode}_offline_rl_candidate.json")
        if not args.dry_run:
            write_json(policy_path, policy)
        policy_paths[budget_mode] = policy_path
        policies[budget_mode] = policy

    candidate_bundle_path = str(out_dir / "final_budget_policy_bundle_offline_rl_candidate.json")
    if not args.dry_run:
        write_json(candidate_bundle_path, _bundle_payload(policy_paths, policies))

    manifest = build_experiment_manifest(
        script_name="scripts/train_offline_full_system_rl.py",
        qa_path=args.package,
        dataset_name="offline_full_system_rl",
        dataset_split="train",
        effective_questions=sum(int(coverage[budget].get("num_questions", 0)) for budget in requested_budgets),
        seed=args.seed,
        router_model_path=candidate_bundle_path,
        settings={
            "package": args.package,
            "out_dir": str(out_dir),
            "base_router_model": args.base_router_model,
            "probe_corpus": args.probe_corpus,
            "holdout_ratio": args.holdout_ratio,
            "budget_modes": requested_budgets,
            "policy_method_by_budget": policy_method_by_budget,
            "gate_baseline_workflow_by_budget": gate_baseline_workflow_by_budget,
            "gate_min_advantage_by_budget": gate_min_advantage_by_budget,
            "gate_min_confidence_by_budget": gate_min_confidence_by_budget,
            "gate_allowed_workflows_by_budget": gate_allowed_workflows_by_budget,
            "dry_run": args.dry_run,
        },
    )
    report = {
        "manifest": manifest,
        "stage": {
            "offline_full_system_rl_opened": True,
            "online_rl_enabled": False,
            "candidate_bundle": candidate_bundle_path,
            "behavior_bundle": package.get("macro_layer", {}).get("policy_bundle"),
            "promotion_required": "Evaluate candidate bundle on held-out val/test before replacing behavior bundle.",
        },
        "macro_training": train_runs,
        "micro_reward_models": package.get("micro_layer", {}).get("critic_models", {}),
        "direct_critic_deployment": {
            "enabled": False,
            "reason": "The full-system RL package marks direct critic deployment as not ready.",
            "checks": package.get("micro_layer", {}).get("direct_critic_evaluations", {}),
        },
        "next_commands": {
            "evaluate_candidate_bundle": (
                "python scripts/eval_final_budget_bundle.py "
                "--corpus data/hotpotqa_large/corpus.json "
                "--qa data/hotpotqa_large/splits/test.json "
                "--dataset_split test --dataset_name hotpotqa_large "
                f"--policy_bundle {candidate_bundle_path} "
                "--router_model outputs/router_hotpot_budget_calibrated.joblib "
                "--llm_provider ollama --llm_model llama3.2 "
                "--hybrid_min_confidence 0.55 --hybrid_low_cost_confidence 0.55 "
                "--hybrid_low_cost_workflows W1 --budget_modes low medium high "
                "--limit 90 --seed 13 "
                f"--out {out_dir / 'final_budget_policy_test_eval_offline_rl_candidate.json'}"
            ),
            "compare_with_frozen_bundle": (
                "python scripts/build_final_project_report.py "
                f"--policy_bundle {candidate_bundle_path} "
                f"--test_eval {out_dir / 'final_budget_policy_test_eval_offline_rl_candidate.json'} "
                "--reference_test_eval outputs/final_budget_policy_test_eval_rl_ready.json "
                "--micro_eval_stage high=outputs/router_eval_large_budget_high_switch_qrag_det.json "
                f"--out {out_dir / 'final_project_report_offline_rl_candidate.json'}"
            ),
        },
    }
    report_path = str(out_dir / "offline_full_system_rl_training_report.json")
    if not args.dry_run:
        write_json(report_path, report)

    print("=== Offline full-system RL training ===")
    print(f"opened=True | online_rl_enabled=False | budgets={requested_budgets}")
    for budget_mode, row in train_runs.items():
        holdout = policies[budget_mode].get("offline_rl", {}).get("holdout_policy_eval", {})
        print(
            f"{budget_mode}: method={row['recommended_method']} | workflows={row['candidate_workflows']} | "
            f"model={row['model_path']} | "
            f"holdout_utility={holdout.get('avg_policy_utility')} | regret={holdout.get('avg_regret')} | "
            f"best_rate={holdout.get('exact_best_rate')}"
        )
    print(f"Candidate bundle: {candidate_bundle_path}")
    print(f"Training report: {report_path}")


if __name__ == "__main__":
    main()
