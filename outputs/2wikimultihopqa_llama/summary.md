# Experimental Setup and Training Results Summary

Generated at UTC: `2026-06-10T14:28:00+00:00`
Output directory: `outputs/2wikimultihopqa_llama`

## 1. Experimental Setup

| Item | Value |
| --- | --- |
| Dataset/package name | full_system_rl |
| Split/stage | gate |
| Effective questions | 270 |
| Policy bundle | outputs/2wikimultihopqa/final_budget_policy_bundle.json |
| Final evaluation | outputs/2wikimultihopqa/final_budget_policy_test_eval.json |
| Final report | outputs/2wikimultihopqa/final_project_report.json |

### Counterfactual Rollout Coverage

| Budget | Questions | Runs | Workflow counts | Workflow avg utility | Has counterfactuals |
| --- | --- | --- | --- | --- | --- |
| high | 420 | 840 | {'W2': 420, 'W3': 420} | {'W2': 0.213, 'W3': 0.189} | Yes |
| low | 420 | 1680 | {'W1': 420, 'W2': 420, 'W3': 420, 'W6': 420} | {'W1': 0.2227, 'W2': 0.1551, 'W3': 0.1468, 'W6': 0.1191} | Yes |
| medium | 420 | 1680 | {'W1': 420, 'W2': 420, 'W3': 420, 'W6': 420} | {'W1': 0.2227, 'W2': 0.1551, 'W3': 0.1468, 'W6': 0.1191} | Yes |

## 2. Macro Budget Policy Results

| Budget | Method | Runs | Utility | EM | F1 | Tokens | Latency ms | Workflows | Zero utility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| low | gated_bandit_router | 90 | 0.168 | 0.1444 | 0.1726 | 90.09 | 887.52 | {'W3': 52, 'W1': 38} | 0.7667 |
| medium | bandit_router | 90 | 0.2161 | 0.1444 | 0.1748 | 180.97 | 1923.32 | {'W3': 20, 'W6': 23, 'W2': 28, 'W1': 19} | 0.3889 |
| high | bandit_router | 90 | 0.2353 | 0.1333 | 0.1567 | 204.24 | 1722.43 | {'W3': 32, 'W2': 58} | 0.0 |

## 3. High-Budget Bandit Training

| Metric | Value |
| --- | --- |
| Source | outputs/2wikimultihopqa_llama/cv_ensemble_report.json |
| Model | outputs/2wikimultihopqa_llama/cv_ensemble_high_bandit.joblib |
| Best config | gbt_n50_d2_lr003 |
| Average regret | 0.0419 |
| Exact best rate | 0.7928 |
| Meets online threshold | No |

## 4. Micro Critic Training Results

| Module | Train examples | Eval examples | MAE | RMSE | Pearson | Spearman | Ready offline reward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AG | 4283 | 756 | 0.2203 | 0.295 | 0.2987 | 0.2112 | False |
| QR | 1071 | 189 | 0.0955 | 0.1202 | 0.705 | 0.6745 | True |

## 5. Selective Critic Verification

| Metric | Value |
| --- | --- |
| Baseline utility | 0.2364 |
| Baseline tokens | 231.4333 |
| Full critic utility | 0.2216 |
| Full critic tokens | 601.7889 |
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

- `full_system_rl_package`: `outputs/2wikimultihopqa_llama/full_system_rl_package.json`
- `macro_budget_metrics`: `outputs/2wikimultihopqa_llama/metrics/metrics_macro_budget_summary.csv`
- `micro_critic_metrics`: `outputs/2wikimultihopqa_llama/metrics/metrics_micro_critic_summary.csv`
- `cv_ensemble_report`: `outputs/2wikimultihopqa_llama/cv_ensemble_report.json`
- `selective_critic_verification`: `outputs/2wikimultihopqa_llama/selective_critic_verification.json`
- `regate_report`: `outputs/2wikimultihopqa_llama/regate_report.json`
