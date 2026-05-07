from __future__ import annotations

import argparse

from map_rag_gym.utils.io import read_json, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", required=True, help="Budget policy JSON files from scripts/select_budget_policy.py")
    ap.add_argument("--out", default="outputs/final_budget_policy_bundle.json")
    args = ap.parse_args()

    bundle: dict[str, dict] = {}
    source_files: list[str] = []
    source_eval_files: list[str] = []

    for path in args.policies:
        payload = read_json(path)
        budget_mode = str(payload.get("budget_mode") or "").strip().lower()
        if not budget_mode:
            raise ValueError(f"Policy file {path} does not define budget_mode.")
        bundle[budget_mode] = {
            "recommended_method": payload.get("recommended_method"),
            "constraints": payload.get("constraints", {}),
            "router_settings": payload.get("router_settings", {}),
            "source_policy_file": path,
            "source_eval_file": payload.get("source_eval_file"),
            "pareto_frontier": payload.get("pareto_frontier", []),
        }
        source_files.append(path)
        source_eval = payload.get("source_eval_file")
        if source_eval:
            source_eval_files.append(source_eval)

    output = {
        "source_policy_files": source_files,
        "source_eval_files": source_eval_files,
        "budget_policies": bundle,
        "budget_modes": sorted(bundle),
        "policy_table": [
            {
                "budget_mode": budget_mode,
                "recommended_method": info.get("recommended_method"),
                "constraints": info.get("constraints", {}),
                "router_settings": info.get("router_settings", {}),
                "source_eval_file": info.get("source_eval_file"),
            }
            for budget_mode, info in sorted(bundle.items())
        ],
    }
    write_json(args.out, output)

    print("Final budget policy table:")
    for row in output["policy_table"]:
        print(
            f"{row['budget_mode']}: {row['recommended_method']} "
            f"| constraints={row['constraints']} | source={row['source_eval_file']}"
        )
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
