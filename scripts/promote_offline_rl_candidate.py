from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from map_rag_gym.router.budget import normalize_budget_mode
from map_rag_gym.utils.experiment import build_experiment_manifest
from map_rag_gym.utils.io import read_json, write_json


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Promote an offline RL candidate bundle to become the new frozen bundle. "
            "This script validates the promotion gate, backs up the current frozen "
            "files, copies the candidate into the frozen path, and writes a promotion report."
        ),
    )
    ap.add_argument(
        "--candidate_bundle",
        default="outputs/offline_full_system_rl_guarded/final_budget_policy_bundle_offline_rl_candidate.json",
        help="Path to the candidate bundle to promote.",
    )
    ap.add_argument(
        "--candidate_eval",
        default="outputs/offline_full_system_rl_guarded/final_budget_policy_test_eval_offline_rl_candidate.json",
        help="Path to the candidate test evaluation.",
    )
    ap.add_argument(
        "--candidate_report",
        default="outputs/offline_full_system_rl_guarded/final_project_report_offline_rl_candidate.json",
        help="Path to the candidate project report.",
    )
    ap.add_argument(
        "--promotion_check",
        default="outputs/offline_full_system_rl_guarded/promotion_check.json",
        help="Path to the promotion check result.",
    )
    ap.add_argument(
        "--frozen_bundle",
        default="outputs/final_budget_policy_bundle_rl_ready.json",
        help="Path to the current frozen bundle (will be backed up then overwritten).",
    )
    ap.add_argument(
        "--frozen_eval",
        default="outputs/final_budget_policy_test_eval_rl_ready.json",
        help="Path to the current frozen evaluation (will be backed up then overwritten).",
    )
    ap.add_argument(
        "--frozen_report",
        default="outputs/final_project_report_rl_ready.json",
        help="Path to the current frozen report (will be backed up then overwritten).",
    )
    ap.add_argument(
        "--rl_package",
        default="outputs/full_system_rl_package.json",
        help="Path to the full-system RL package to update.",
    )
    ap.add_argument(
        "--backup_suffix",
        default="_pre_promotion",
        help="Suffix for backup files.",
    )
    ap.add_argument(
        "--skip_gate_check",
        action="store_true",
        help="Skip the promotion gate check (not recommended).",
    )
    ap.add_argument(
        "--out",
        default="outputs/promotion_report.json",
        help="Path to write the promotion report.",
    )
    args = ap.parse_args()

    # ── 1. Validate promotion gate ──────────────────────────────────────
    promotion_check = read_json(args.promotion_check)
    if not promotion_check.get("ready_to_promote") and not args.skip_gate_check:
        blocked = promotion_check.get("blocked_budgets", [])
        raise ValueError(
            f"Promotion gate has NOT passed. Blocked budgets: {blocked}. "
            "Fix regressions before promoting or use --skip_gate_check."
        )
    print("[OK] Promotion gate passed.")

    # ── 2. Read candidate and frozen data ───────────────────────────────
    candidate_bundle = read_json(args.candidate_bundle)
    candidate_eval = read_json(args.candidate_eval)
    
    frozen_bundle = read_json(args.frozen_bundle) if Path(args.frozen_bundle).exists() else {}
    frozen_eval = read_json(args.frozen_eval) if Path(args.frozen_eval).exists() else {}

    # Collect before/after comparison
    budget_comparison = {}
    for budget_mode in sorted(set(candidate_eval.get("budget_results", {}))):
        cand_result = candidate_eval["budget_results"][budget_mode]
        froz_result = frozen_eval.get("budget_results", {}).get(budget_mode, {})
        cand_summary = cand_result.get("summary", {})
        froz_summary = froz_result.get("summary", {})
        cand_utility = float(cand_summary.get("avg_utility", 0.0))
        froz_utility = float(froz_summary.get("avg_utility", 0.0))
        budget_comparison[budget_mode] = {
            "old_method": froz_result.get("recommended_method"),
            "new_method": cand_result.get("recommended_method"),
            "old_utility": round(froz_utility, 4),
            "new_utility": round(cand_utility, 4),
            "delta": round(cand_utility - froz_utility, 4),
            "old_tokens": froz_summary.get("avg_tokens"),
            "new_tokens": cand_summary.get("avg_tokens"),
            "old_workflow_counts": froz_summary.get("workflow_counts", {}),
            "new_workflow_counts": cand_summary.get("workflow_counts", {}),
        }

    # ── 3. Back up current frozen files ─────────────────────────────────
    backup_paths = {}
    for label, path in [
        ("frozen_bundle", args.frozen_bundle),
        ("frozen_eval", args.frozen_eval),
        ("frozen_report", args.frozen_report),
    ]:
        src = Path(path)
        if src.exists():
            backup = src.with_stem(src.stem + args.backup_suffix)
            shutil.copy2(src, backup)
            backup_paths[label] = str(backup)
            print(f"  Backed up {src.name} -> {backup.name}")
        else:
            backup_paths[label] = None
            print(f"  Skipped backup for {src.name} (not found)")

    # ── 4. Promote: overwrite frozen with candidate ─────────────────────
    # 4a. Update the candidate bundle to remove offline_rl metadata
    #     and write it as the new frozen bundle
    promoted_bundle = _strip_offline_rl_metadata(candidate_bundle)
    write_json(args.frozen_bundle, promoted_bundle)
    print(f"[OK] Promoted bundle -> {args.frozen_bundle}")

    # 4b. Copy candidate eval as new frozen eval
    shutil.copy2(args.candidate_eval, args.frozen_eval)
    print(f"[OK] Promoted eval -> {args.frozen_eval}")

    # 4c. Copy candidate report as new frozen report (if it exists)
    if Path(args.candidate_report).exists():
        shutil.copy2(args.candidate_report, args.frozen_report)
        print(f"[OK] Promoted report -> {args.frozen_report}")

    # ── 5. Update RL package to reference the new frozen bundle ─────────
    if Path(args.rl_package).exists():
        rl_package = read_json(args.rl_package)
        rl_package["macro_layer"]["policy_bundle"] = args.frozen_bundle
        # Update budget policies to match the promoted bundle
        for budget_mode, policy in promoted_bundle.get("budget_policies", {}).items():
            if budget_mode in rl_package.get("macro_layer", {}).get("budget_policies", {}):
                pkg_policy = rl_package["macro_layer"]["budget_policies"][budget_mode]
                pkg_policy["recommended_method"] = policy.get("recommended_method")
                pkg_policy["router_settings"] = policy.get("router_settings", {})
                pkg_policy["constraints"] = policy.get("constraints", {})
        # Update evaluation section
        if "evaluation" in rl_package:
            eval_section = rl_package["evaluation"]
            eval_section["final_budget_eval"] = {}
            for budget_mode, result in candidate_eval.get("budget_results", {}).items():
                summary = result.get("summary", {})
                eval_section["final_budget_eval"][budget_mode] = {
                    "recommended_method": result.get("recommended_method"),
                    "num_runs": summary.get("num_runs"),
                    "avg_utility": summary.get("avg_utility"),
                    "avg_em": summary.get("avg_em"),
                    "avg_f1_proxy": summary.get("avg_f1_proxy"),
                    "avg_tokens": summary.get("avg_tokens"),
                    "avg_latency_ms": summary.get("avg_latency_ms"),
                    "workflow_counts": summary.get("workflow_counts", {}),
                }
        write_json(args.rl_package, rl_package)
        print(f"[OK] Updated RL package -> {args.rl_package}")

    # ── 6. Write promotion report ───────────────────────────────────────
    report = {
        "manifest": build_experiment_manifest(
            script_name="scripts/promote_offline_rl_candidate.py",
            qa_path=args.candidate_eval,
            router_model_path=args.frozen_bundle,
            dataset_name="promotion",
            dataset_split="promote",
            effective_questions=sum(
                int(candidate_eval.get("budget_results", {}).get(bm, {}).get("summary", {}).get("num_runs", 0))
                for bm in budget_comparison
            ),
            settings={
                "candidate_bundle": args.candidate_bundle,
                "candidate_eval": args.candidate_eval,
                "frozen_bundle": args.frozen_bundle,
                "frozen_eval": args.frozen_eval,
                "promotion_check": args.promotion_check,
                "skip_gate_check": args.skip_gate_check,
            },
        ),
        "promoted": True,
        "budget_comparison": budget_comparison,
        "backup_paths": backup_paths,
        "promoted_paths": {
            "frozen_bundle": args.frozen_bundle,
            "frozen_eval": args.frozen_eval,
            "frozen_report": args.frozen_report,
        },
        "summary": (
            "Guarded offline RL candidate has been promoted to the new frozen bundle. "
            "The previous frozen bundle has been backed up."
        ),
    }
    write_json(args.out, report)

    # ── 7. Print summary ────────────────────────────────────────────────
    print("\n=== Promotion Summary ===")
    for budget_mode, comp in sorted(budget_comparison.items()):
        print(
            f"  {budget_mode}: {comp['old_method']} -> {comp['new_method']} | "
            f"utility {comp['old_utility']:.4f} -> {comp['new_utility']:.4f} "
            f"(delta={comp['delta']:+.4f})"
        )
    print(f"\nSaved promotion report -> {args.out}")
    print("Done. The promoted bundle is now the frozen reference for future candidates.")


def _strip_offline_rl_metadata(bundle: dict) -> dict:
    """Remove offline_rl metadata from a candidate bundle for promotion."""
    import copy

    promoted = copy.deepcopy(bundle)
    # Remove top-level offline_rl key
    promoted.pop("offline_rl", None)
    # Remove per-budget offline_rl keys
    for budget_mode, policy in promoted.get("budget_policies", {}).items():
        policy.pop("offline_rl", None)
    return promoted


if __name__ == "__main__":
    main()
