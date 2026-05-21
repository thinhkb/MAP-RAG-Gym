# Report visualization summary

## Key talking points

- Best final budget by utility: `high` with utility `0.4375`.
- High-budget result: utility `0.4375`, EM `0.2778`, F1 proxy `0.3220`, tokens `176.53`.
- Bandit gate: avg regret `0.0352`, exact-best rate `0.8047`.
- Critic validation: QR Spearman `0.6745`, AG Spearman `0.2112`.
- Online gate: `True`, deployment mode `selective_critic_online`.

## Caveat for paper comparison

The paper comparison figure is only a reference. The papers report official HotpotQA EM/F1, while this project uses a local HotpotQA-large split, local LLM settings, and `f1_proxy` plus utility/cost metrics.

## Generated files

- `01_budget_quality.png`
- `02_budget_costs.png`
- `03_macro_quality_cost_tradeoff.png`
- `04_workflow_distribution.png`
- `05_micro_critic_summary.png`
- `06_critic_deployment_tradeoff.png`
- `07_selective_critic_gate.png`
- `08_high_bandit_cv.png`
- `09_hotpotqa_paper_reference.png`
- `10_hotpotqa_paper_reference_table.png`
- `11_budget_summary_table.png`
- `12_system_gate_table.png`
