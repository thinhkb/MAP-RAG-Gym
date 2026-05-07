# MAP-RAG-Gym

Project này ghép hai ý tưởng:
- `MAO-ARAG` ở tầng `macro`: planner/router chọn workflow theo `query + budget + cost`.
- `RAG-Gym` ở tầng `micro`: critic chấm chất lượng step/candidate action bên trong workflow.

Trạng thái hiện tại:
- Đã mở `offline full-system RL`.
- Chưa mở `online RL`.
- Bundle đang được giữ làm mốc an toàn là `outputs/final_budget_policy_bundle_rl_ready.json`.
- Candidate bundle của offline RL đã train xong nhưng chưa được promote vì bị regression ở budget `low`.

## Kiến trúc hiện tại

### Macro layer
- `low`: ưu tiên rẻ, bundle hiện tại chọn `hybrid_router`.
- `medium`: trade-off chất lượng/chi phí, bundle hiện tại chọn `gated_bandit_router`.
- `high`: ưu tiên chất lượng, bundle hiện tại chọn `bandit_router` trên cặp `W2/W3`.

### Micro layer
- Critic đang sẵn sàng làm `offline reward model` cho:
  - `QR`: dùng `local_reward`
  - `AG`: dùng `blended_reward`
- Critic chưa được bật mặc định để rerank online vì vẫn làm utility giảm nhẹ và tăng cost.

### Workflow library

| Workflow | Steps | Ý nghĩa |
| --- | --- | --- |
| `W1` | `AG` | Direct answer, rẻ nhất |
| `W2` | `QR -> RA -> AG` | Rewrite rồi retrieve |
| `W3` | `RA -> DS -> AG` | Retrieve rồi select evidence |
| `W4` | `QDP -> RA -> AS` | Parallel decomposition |
| `W5` | `QDS -> QR -> RA -> AS` | Serial decomposition |
| `W6` | `DRAFT -> REFLECT -> RA -> AG` | Reflective retrieval |

## Kết quả chính

### 1. Frozen bundle hiện tại là bundle tốt nhất và an toàn nhất

File:
- `outputs/final_budget_policy_bundle_rl_ready.json`
- `outputs/final_budget_policy_test_eval_rl_ready.json`
- `outputs/final_project_report_rl_ready.json`

Kết quả test 90 câu mỗi budget:

| Budget | Policy | Utility | So với static reference |
| --- | --- | ---: | ---: |
| `low` | `hybrid_router` | `0.2899` | `+0.0031` |
| `medium` | `gated_bandit_router` | `0.3849` | `+0.0145` |
| `high` | `bandit_router` | `0.4544` | `+0.0307` |

Kết luận:
- Adaptive macro routing đã thắng static reference ở cả `low/medium/high`.
- Frontier theo budget là hợp lý: `low < medium < high`.
- Kiến trúc hiện tại đã đúng tinh thần `MAO-ARAG + RAG-Gym` ở mức routing + reward modeling offline.

### 2. Offline full-system RL đã được mở

File:
- `outputs/full_system_rl_package.json`

Gate hiện tại:
- `ready_for_offline_full_system_rl = true`
- `ready_for_online_full_system_rl = false`
- `deployment_mode = offline_reward_model_only`

Ý nghĩa:
- Có thể train offline policy improvement trên tầng macro.
- Có thể dùng critic làm reward model ở tầng micro.
- Chưa được bật direct online rerank/update.

### 3. Candidate offline RL đã train xong nhưng chưa nên promote

File:
- `outputs/offline_full_system_rl/final_budget_policy_bundle_offline_rl_candidate.json`
- `outputs/offline_full_system_rl/final_budget_policy_test_eval_offline_rl_candidate.json`
- `outputs/offline_full_system_rl/offline_full_system_rl_training_report.json`

So sánh candidate offline RL với frozen bundle:

| Budget | Frozen | Offline RL candidate | Delta |
| --- | ---: | ---: | ---: |
| `low` | `0.2899` | `0.2811` | `-0.0088` |
| `medium` | `0.3849` | `0.3859` | `+0.0010` |
| `high` | `0.4544` | `0.4559` | `+0.0015` |

Kết luận:
- `medium` và `high` nhích lên nhẹ.
- `low` bị giảm rõ hơn mức gain ở hai budget còn lại.
- Vì vậy candidate offline RL hiện tại **chưa đủ điều kiện để thay frozen bundle**.

### 4. Critic micro đã đủ tốt cho offline reward modeling, nhưng chưa đủ tốt cho online deployment

File:
- `outputs/process_critic_budget_qr_local.joblib.meta.json`
- `outputs/process_critic_budget_ag.joblib.meta.json`
- `outputs/router_eval_large_budget_high_w2w3_det_critic2_val.json`

Chất lượng critic:
- `QR`: `spearman = 0.6669`
- `AG`: `spearman = 0.3543`

Kiểm tra direct critic deployment trên high-budget:
- `bandit_router`: utility `0.4397`, tokens `156.8`
- `bandit_router_critic`: utility `0.4362`, tokens `229.0`
- Gap utility: `-0.0035`
- Token multiplier: `1.4601x`

Kết luận:
- Critic đủ tốt để dùng làm `offline reward model`.
- Critic **chưa đủ tốt để bật rerank online mặc định**.

## Những gì đã làm được

- Đã xây dựng workflow library `W1` đến `W6`.
- Đã có router rule-based, learned, hybrid, bandit, gated-bandit.
- Đã đưa `budget_mode` vào utility và routing.
- Đã build benchmark lớn hơn với `420 train / 90 val / 90 test`.
- Đã train critic budget-aware cho `QR` và `AG`.
- Đã chọn được frozen final bundle thắng static reference trên held-out test.
- Đã mở được gate `offline full-system RL`.
- Đã train xong candidate macro policies cho offline RL.

## Những gì chưa làm được

- Chưa có candidate offline RL nào thắng frozen bundle trên cả 3 budget cùng lúc.
- Chưa giữ được chất lượng `low-budget` sau bước offline policy improvement.
- Chưa đưa critic vào online rerank mà vẫn giữ utility không giảm.
- Chưa mở `online RL` toàn hệ thống.
- High-budget macro bandit vẫn còn holdout regret khá cao, nên chưa phù hợp cho online update.

## Kế hoạch tiếp theo

### Ưu tiên 1: sửa regression ở `low-budget`
- Ràng chặt hơn cost cap khi train offline RL candidate cho `low`.
- Giữ bias về `W3` mạnh hơn, tránh bandit chuyển sang `W1` quá nhiều.
- Chỉ promote candidate nếu `low` không còn bị âm so với frozen bundle.

### Ưu tiên 2: làm mạnh hơn high-bandit để chuẩn bị online RL
- Cải thiện feature/context cho high bandit.
- Giảm holdout regret và tăng exact-best rate trên budget `high`.
- Chỉ mở online update khi macro bandit đủ ổn định trên holdout.

### Ưu tiên 3: giảm cost của critic deployment
- Thử critic theo module hẹp hơn, thay vì bật toàn pipeline.
- Thử chỉ critic ở `AG`, hoặc chỉ trên một subset query khó.
- Tối ưu `critic_n_candidates` và gate để token multiplier thấp hơn.

## Các file đang được giữ lại

### Frozen path
- `outputs/final_budget_policy_bundle_rl_ready.json`
- `outputs/final_budget_policy_test_eval_rl_ready.json`
- `outputs/final_budget_policy_test_eval_v2_det90.json`
- `outputs/final_project_report_rl_ready.json`
- `outputs/full_system_rl_package.json`

### Data và rollout để tái lập
- `outputs/hotpotqa_large_train_rollouts.json`
- `outputs/hotpotqa_large_train_rollouts_high_w2w3_det.json`
- `outputs/hotpotqa_large_process_train_budget.json`

### Critic và macro eval còn dùng
- `outputs/process_critic_budget_qr_local.joblib.meta.json`
- `outputs/process_critic_budget_ag.joblib.meta.json`
- `outputs/router_eval_large_budget_low_bandit_v2.json`
- `outputs/router_eval_large_budget_medium_bandit.json`
- `outputs/router_eval_large_budget_high_w2w3_det_val.json`
- `outputs/router_eval_large_budget_high_w2w3_det_test.json`
- `outputs/router_eval_large_budget_high_switch_qrag_det.json`
- `outputs/router_eval_large_budget_high_w2w3_det_critic2_val.json`

### Offline full-system RL candidate
- `outputs/offline_full_system_rl/final_budget_policy_bundle_offline_rl_candidate.json`
- `outputs/offline_full_system_rl/final_budget_policy_test_eval_offline_rl_candidate.json`
- `outputs/offline_full_system_rl/offline_full_system_rl_training_report.json`

## Lệnh còn cần dùng

### 1. Re-evaluate frozen bundle
```bash
python scripts/eval_final_budget_bundle.py --corpus data/hotpotqa_large/corpus.json --qa data/hotpotqa_large/splits/test.json --dataset_split test --dataset_name hotpotqa_large --policy_bundle outputs/final_budget_policy_bundle_rl_ready.json --router_model outputs/router_hotpot_budget_calibrated.joblib --llm_provider ollama --llm_model llama3.2 --hybrid_min_confidence 0.55 --hybrid_low_cost_confidence 0.55 --hybrid_low_cost_workflows W1 --budget_modes low medium high --limit 90 --seed 13 --out outputs/final_budget_policy_test_eval_rl_ready.json
```

### 2. Rebuild the full-system RL gate
```bash
python scripts/build_full_system_rl_package.py --policy_bundle outputs/final_budget_policy_bundle_rl_ready.json --final_eval outputs/final_budget_policy_test_eval_rl_ready.json --final_report outputs/final_project_report_rl_ready.json --reference_eval outputs/final_budget_policy_test_eval_v2_det90.json --macro_rollout low=outputs/hotpotqa_large_train_rollouts.json medium=outputs/hotpotqa_large_train_rollouts.json high=outputs/hotpotqa_large_train_rollouts_high_w2w3_det.json --critic_model QR=outputs/process_critic_budget_qr_local.joblib AG=outputs/process_critic_budget_ag.joblib --critic_meta QR=outputs/process_critic_budget_qr_local.joblib.meta.json AG=outputs/process_critic_budget_ag.joblib.meta.json --direct_critic_eval high=outputs/router_eval_large_budget_high_w2w3_det_critic2_val.json --out outputs/full_system_rl_package.json
```

### 3. Train offline full-system RL candidate
```bash
python scripts/train_offline_full_system_rl.py --package outputs/full_system_rl_package.json --out_dir outputs/offline_full_system_rl --base_router_model outputs/router_hotpot_budget_calibrated.joblib --probe_corpus data/hotpotqa_large/corpus.json --seed 13
```

### 4. Evaluate offline RL candidate before promotion
```bash
python scripts/eval_final_budget_bundle.py --corpus data/hotpotqa_large/corpus.json --qa data/hotpotqa_large/splits/test.json --dataset_split test --dataset_name hotpotqa_large --policy_bundle outputs/offline_full_system_rl/final_budget_policy_bundle_offline_rl_candidate.json --router_model outputs/router_hotpot_budget_calibrated.joblib --llm_provider ollama --llm_model llama3.2 --hybrid_min_confidence 0.55 --hybrid_low_cost_confidence 0.55 --hybrid_low_cost_workflows W1 --budget_modes low medium high --limit 90 --seed 13 --out outputs/offline_full_system_rl/final_budget_policy_test_eval_offline_rl_candidate.json
```

## Kết luận cuối cùng

Project hiện tại đã đạt đúng tinh thần `MAO-ARAG + RAG-Gym` ở mức:
- `macro`: chọn workflow theo query/budget/cost
- `micro`: critic chấm step/candidate action
- `system`: đã có gate rõ ràng để phân biệt `offline RL ready` và `online RL ready`

Trạng thái cuối:
- `offline full-system RL`: **đã mở**
- `online RL`: **chưa mở**
- Bundle nên dùng hiện tại: **frozen rl_ready bundle**
- Offline RL candidate: **đã train xong nhưng chưa promote**
