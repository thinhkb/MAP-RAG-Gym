# Experimental Setup and Training Results Summary

Generated at UTC: `2026-06-10T14:28:01+00:00`
Output directory: `outputs/hotpotqa_large_gemma3`

## 1. Experimental Setup

| Item | Value |
| --- | --- |
| Dataset/package name | full_system_rl |
| Split/stage | gate |
| Effective questions | 270 |
| Policy bundle | outputs/hotpotqa_large_gemma3/final_budget_policy_bundle.json |
| Final evaluation | outputs/hotpotqa_large_gemma3/final_budget_policy_test_eval.json |
| Final report | outputs/hotpotqa_large_gemma3/final_project_report.json |

### Counterfactual Rollout Coverage

| Budget | Questions | Runs | Workflow counts | Workflow avg utility | Has counterfactuals |
| --- | --- | --- | --- | --- | --- |
| high | 420 | 840 | {'W2': 420, 'W3': 420} | {'W2': 0.1892, 'W3': 0.2683} | Yes |
| low | 420 | 1680 | {'W1': 420, 'W2': 420, 'W3': 420, 'W6': 420} | {'W1': 0.1017, 'W2': 0.1375, 'W3': 0.2235, 'W6': 0.1107} | Yes |
| medium | 420 | 1680 | {'W1': 420, 'W2': 420, 'W3': 420, 'W6': 420} | {'W1': 0.1017, 'W2': 0.1375, 'W3': 0.2235, 'W6': 0.1107} | Yes |

## 2. Macro Budget Policy Results

| Budget | Method | Runs | Utility | EM | F1 | Tokens | Latency ms | Workflows | Zero utility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| low | gated_bandit_router | 90 | 0.168 | 0.1444 | 0.1726 | 90.09 | 887.52 | {'W3': 52, 'W1': 38} | 0.7667 |
| medium | bandit_router | 90 | 0.2161 | 0.1444 | 0.1748 | 180.97 | 1923.32 | {'W3': 20, 'W6': 23, 'W2': 28, 'W1': 19} | 0.3889 |
| high | bandit_router | 90 | 0.2353 | 0.1333 | 0.1567 | 204.24 | 1722.43 | {'W3': 32, 'W2': 58} | 0.0 |

## 3. High-Budget Bandit Training

| Metric | Value |
| --- | --- |
| Source | outputs/hotpotqa_large_gemma3/cv_ensemble_report.json |
| Model | outputs/hotpotqa_large_gemma3/cv_ensemble_high_bandit.joblib |
| Best config | ens_ra10.0_n150_d4_lr005_b0.4 |
| Average regret | 0.0193 |
| Exact best rate | 0.8333 |
| Meets online threshold | Yes |

## 4. Micro Critic Training Results

| Module | Train examples | Eval examples | MAE | RMSE | Pearson | Spearman | Ready offline reward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AG | 4283 | 756 | 0.2203 | 0.295 | 0.2987 | 0.2112 | False |
| QR | 1071 | 189 | 0.0955 | 0.1202 | 0.705 | 0.6745 | True |

## 5. Selective Critic Verification

| Metric | Value |
| --- | --- |
| Baseline utility | 0.2441 |
| Baseline tokens | 237.6444 |
| Full critic utility | 0.2329 |
| Full critic tokens | 564.3 |
| Best passing gate | 1 |
| Best passing token multiplier | 1 |
| Best passing utility vs base | 0 |
| Online strategy | - |
| Meets online threshold | - |
| Verified on holdout | - |

## 6. RL Readiness / Deployment Status

| Metric | Value |
| --- | --- |
| Offline RL ready | No |
| Online RL ready | No |
| Deployment mode | online_rl_candidate |
| Recommended next stage | fix_offline_rl_blockers |
| Deployment recommendation | - |
| Offline blockers | Final project report has not opened offline RL.; Not every budget mode uses an adaptive macro policy.; Micro critic reward models are missing or below quality thresholds. |
| Online blockers |  |

## 7. Source Files

- `full_system_rl_package`: `outputs/hotpotqa_large_gemma3/full_system_rl_package.json`
- `macro_budget_metrics`: `outputs/hotpotqa_large_gemma3/metrics/metrics_macro_budget_summary.csv`
- `micro_critic_metrics`: `outputs/hotpotqa_large_gemma3/metrics/metrics_micro_critic_summary.csv`
- `cv_ensemble_report`: `outputs/hotpotqa_large_gemma3/cv_ensemble_report.json`
- `selective_critic_verification`: `outputs/hotpotqa_large_gemma3/selective_critic_verification.json`
- `regate_report`: `outputs/hotpotqa_large_gemma3/regate_report.json`
