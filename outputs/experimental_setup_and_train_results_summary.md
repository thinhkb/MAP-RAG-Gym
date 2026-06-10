# Experimental Setup and Training Results Summary

Generated at UTC: `2026-06-10T14:51:34+00:00`
Output directory: `outputs`

## 1. Experimental Setup

| Item | Value |
| --- | --- |
| Dataset/package name | full_system_rl |
| Split/stage | gate |
| Effective questions | 270 |
| Policy bundle | outputs/final_budget_policy_bundle.json |
| Final evaluation | outputs/final_budget_policy_test_eval.json |
| Final report | outputs/final_project_report.json |

### Counterfactual Rollout Coverage

| Budget | Questions | Runs | Workflow counts | Workflow avg utility | Has counterfactuals |
| --- | --- | --- | --- | --- | --- |
| high | 420 | 840 | {'W2': 420, 'W3': 420} | {'W2': 0.3666, 'W3': 0.3924} | Yes |
| low | 420 | 1680 | {'W1': 420, 'W2': 420, 'W3': 420, 'W6': 420} | {'W1': 0.2212, 'W2': 0.3056, 'W3': 0.3433, 'W6': 0.2547} | Yes |
| medium | 420 | 1680 | {'W1': 420, 'W2': 420, 'W3': 420, 'W6': 420} | {'W1': 0.2212, 'W2': 0.3056, 'W3': 0.3433, 'W6': 0.2547} | Yes |

## 2. Macro Budget Policy Results

| Budget | Method | Runs | Utility | EM | F1 | Tokens | Latency ms | Workflows | Zero utility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| low | gated_bandit_router | 90 | 0.2776 | 0.2222 | 0.272 | 104.46 | 761.99 | {'W1': 25, 'W3': 65} | 0.6 |
| medium | bandit_router | 90 | 0.3613 | 0.2444 | 0.3011 | 116.4 | 907.93 | {'W1': 36, 'W3': 43, 'W2': 6, 'W6': 5} | 0.4444 |
| high | bandit_router | 90 | 0.4375 | 0.2778 | 0.322 | 176.53 | 1151.62 | {'W3': 53, 'W2': 37} | 0.0 |

## 3. High-Budget Bandit Training

| Metric | Value |
| --- | --- |
| Source | outputs/regate_report.json |
| Model | outputs/improved_high_bandit.joblib |
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
| Baseline utility | 0.405 |
| Baseline tokens | 218.4222 |
| Full critic utility | 0.3915 |
| Full critic tokens | 522.0444 |
| Best passing gate | 1 |
| Best passing token multiplier | 1 |
| Best passing utility vs base | 0 |
| Online strategy | confidence_gate=1.0 |
| Meets online threshold | Yes |
| Verified on holdout | Yes |

## 6. RL Readiness / Deployment Status

| Metric | Value |
| --- | --- |
| Offline RL ready | No |
| Online RL ready | Yes |
| Deployment mode | selective_critic_online |
| Recommended next stage | online_full_system_rl_with_selective_critic |
| Deployment recommendation | Online RL with selective critic is ready. |
| Offline blockers | Final project report has not opened offline RL.; Not every budget mode uses an adaptive macro policy.; Micro critic reward models are missing or below quality thresholds. |
| Online blockers |  |

## 7. Source Files

- `full_system_rl_package`: `outputs/full_system_rl_package.json`
- `macro_budget_metrics`: `outputs/metrics/metrics_macro_budget_summary.csv`
- `micro_critic_metrics`: `outputs/metrics/metrics_micro_critic_summary.csv`
- `cv_ensemble_report`: `outputs/cv_ensemble_report.json`
- `selective_critic_verification`: `outputs/selective_critic_verification.json`
- `regate_report`: `outputs/regate_report.json`
