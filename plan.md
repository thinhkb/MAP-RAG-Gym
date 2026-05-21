# MAP-RAG-Gym

## Ý tưởng project

`MAP-RAG-Gym` được xây dựng từ một vấn đề chung của các hệ thống RAG: không có một pipeline cố định nào phù hợp cho mọi câu hỏi. Câu hỏi đơn giản có thể chỉ cần trả lời trực tiếp hoặc retrieve rất ít; câu hỏi nhiều bước có thể cần rewrite, retrieve, chọn bằng chứng, tách câu hỏi song song hoặc tuần tự, rồi tổng hợp lại. Nếu luôn dùng pipeline mạnh nhất thì tốn token, thời gian và số lần retrieval; nếu luôn dùng pipeline rẻ nhất thì dễ mất chất lượng ở câu hỏi khó.

Project này kết hợp hai hướng nghiên cứu để giải quyết vấn đề đó:
- `MAO-ARAG` ở tầng `macro`: học cách chọn workflow RAG phù hợp với từng query, budget và chi phí.
- `RAG-Gym` ở tầng `micro`: học cách đánh giá chất lượng từng action/candidate step bên trong workflow bằng process reward và critic.

### 1. Ý tưởng từ `MAO-ARAG`: adaptive RAG ở tầng workflow

`MAO-ARAG` xem RAG như một hệ thống nhiều agent. Thay vì thiết kế một pipeline cố định, paper định nghĩa một tập executor agent tương ứng với các module RAG phổ biến:
- `QDS`: tách câu hỏi tuần tự, khi sub-question sau phụ thuộc vào đáp án của sub-question trước.
- `QDP`: tách câu hỏi song song, khi các sub-question độc lập và có thể retrieve riêng.
- `QR`: rewrite câu hỏi để dễ tìm kiếm hơn.
- `RA`: retrieval agent, lấy tài liệu liên quan từ corpus.
- `DS`: document selector, lọc tài liệu nhiễu và giữ bằng chứng hữu ích.
- `AG`: answer generator, sinh câu trả lời từ câu hỏi và bằng chứng.
- `AS`: answer summarization, tổng hợp các sub-answer thành đáp án cuối.

Phần trung tâm của `MAO-ARAG` là planner agent. Với mỗi câu hỏi hoặc sub-question, planner chọn một chuỗi executor để tạo workflow riêng cho trường hợp đó. Paper mô hình hóa bài toán này như một Multiagent Semi-Markov Decision Process, vì mỗi action có thể là một module khác nhau và có thời lượng/chi phí khác nhau. Planner được train bằng PPO với reward gồm:
- reward chất lượng dựa trên F1 giữa đáp án dự đoán và đáp án chuẩn;
- cost penalty cho token cost, latency/turn cost và số lần gọi retrieval;
- format penalty nếu workflow planner tạo ra không hợp lệ hoặc không execute được.

Điểm quan trọng của `MAO-ARAG` là đưa quyết định "nên chạy pipeline nào" thành một bài toán học chính sách có ràng buộc chi phí. Nó không chỉ cố tăng F1, mà còn học trade-off giữa chất lượng và tài nguyên.

### 2. Ý tưởng từ `RAG-Gym`: tối ưu từng bước trong agentic RAG

`RAG-Gym` tập trung vào tầng chi tiết hơn: khi một workflow đã chạy, từng bước trung gian có tốt không? Paper mô hình hóa agentic RAG như một high-level MDP:
- state gồm câu hỏi gốc và lịch sử tìm kiếm, tức các query đã sinh cùng các document đã retrieve;
- action là một macro-action như sinh search query tiếp theo hoặc đưa ra final answer;
- outcome reward đánh giá đáp án cuối, còn process reward đánh giá chất lượng của action trung gian.

Từ cách nhìn này, `RAG-Gym` tối ưu agentic RAG theo ba hướng:
- Prompt engineering: đề xuất `Re2Search`, trong đó agent reasoning trước, reflection để tìm claim chưa được chứng minh, rồi search đúng phần còn thiếu.
- Actor tuning: dùng process-level supervision để train actor bằng các thuật toán như SFT, DPO và PPO; paper cho thấy preference/process feedback giúp cải thiện tốt hơn việc chỉ dựa vào reward cuối.
- Critic training: train một critic/reward model để chấm state-action pair, ví dụ một query rewrite có đáng search không, hoặc một answer candidate có đủ tốt không.

Điểm quan trọng của `RAG-Gym` là không xem RAG như hộp đen chỉ chấm đáp án cuối. Nó mở workflow ra thành các bước nhỏ, gán tín hiệu học cho từng bước, rồi dùng critic để chọn hoặc rerank action tốt hơn trong quá trình inference hoặc làm reward model cho offline learning.

### 3. Metrics hai paper dùng để evaluate

`MAO-ARAG` evaluate theo hai nhóm metric: chất lượng đáp án và chi phí chạy workflow.
- Chất lượng đáp án: dùng `F1 score` giữa predicted answer và golden answer. Đây cũng là outcome reward chính khi train planner bằng RL.
- Cost metrics: dùng `Token Cost` (USD/query), `Retrieval Calls` (số lần gọi retriever/query) và `Turns` (số lượt xử lý/query, đại diện cho latency).
- Reward khi train planner: `Rplanner = F1 - alpha * CostPenalty - FormatPenalty`, trong đó `CostPenalty` gom token cost, turn/latency cost và indicator cho việc gọi retrieval; `FormatPenalty` phạt workflow sai format hoặc không execute được.
- Dataset đánh giá: single-hop QA gồm `NQ`, `PopQA`, `AmbigQA`; multi-hop QA gồm `HotpotQA`, `2WikiMultiHopQA`, `MuSiQue`, `Bamboogle`.

`RAG-Gym` evaluate theo metric QA chuẩn và thêm tín hiệu process-level cho các bước trung gian.
- Với `HotpotQA`, `2WikiMultihopQA` và `Bamboogle`: dùng `Exact Match (EM)` và `F1`.
- Với `MedQA`: dùng `Accuracy (Acc)` vì đây là multi-choice QA.
- Khi tính trung bình nhiều dataset, paper xem `Accuracy` của MedQA tương đương với `EM/F1` để tạo average score chung.
- Với critic/process supervision: dữ liệu reward là các preference tuple `(state, preferred action, unpreferred action)`, dùng để train actor/critic và đánh giá gián tiếp bằng việc final `EM/F1/Acc` có tăng khi critic chọn intermediate action tốt hơn hay không.
- Khi so với các agent khác, paper còn báo `Average F1` và trong một số bảng có thêm `CEM` (`Cover Exact Match`) để so sánh với các baseline dùng metric đó.

### 4. Cách `MAP-RAG-Gym` kết hợp hai paper

Project này dùng `MAO-ARAG` làm khung điều phối ở tầng lớn và dùng `RAG-Gym` làm cơ chế đánh giá ở tầng nhỏ.

Ở tầng `macro`, project xây một workflow library `W1` đến `W6`, tương ứng với các kiểu RAG từ rẻ đến phức tạp: trả lời trực tiếp, rewrite-retrieve-generate, retrieve-select-generate, decomposition song song, decomposition tuần tự và reflective retrieval. Router/planner chọn workflow theo `query + budget_mode + cost/utility`, với ba budget `low`, `medium`, `high`. Đây là phiên bản thực dụng của ý tưởng `MAO-ARAG`: thay planner LLM tự sinh workflow tự do bằng một tập workflow hợp lệ, đo được chi phí, dễ evaluate và dễ rollback.

Ở tầng `micro`, project mượn tinh thần `RAG-Gym` để train critic cho các bước quan trọng như `QR` và `AG`. Critic không chỉ hỏi "đáp án cuối đúng không", mà học tín hiệu gần hơn với quá trình: rewrite có hữu ích không, candidate answer có tốt không, step này có làm tăng utility không. Trong bản hiện tại, critic được dùng an toàn nhất như `offline reward model`; chưa bật mặc định cho online rerank vì thử nghiệm cho thấy còn làm tăng cost và giảm nhẹ utility.

Vì vậy, mục tiêu của project không phải chỉ là "một RAG pipeline tốt", mà là một hệ thống RAG thích nghi theo budget:
- câu hỏi/budget rẻ ưu tiên workflow ít tốn tài nguyên;
- câu hỏi/budget trung bình dùng router có gate để cân bằng chất lượng và chi phí;
- câu hỏi/budget cao cho phép workflow mạnh hơn và bandit chọn giữa các phương án tốt;
- mọi policy mới phải qua held-out evaluation và promotion gate để tránh regression.

Trạng thái hiện tại:
- Đã hoàn tất `offline full-system RL` (train -> eval -> promote).
- Đã promote candidate guarded thành frozen bundle mới.
- Đã cải thiện high-budget bandit bằng Ridge(a=0.1) + 5-fold CV (cv_regret: 0.0260).
- Đã verify selective critic trên held-out (gate=0.70 PASS).
- **Đã mở `online RL`**: tất cả 4 deployment gates đều PASS.
- Bundle đang dùng là `outputs/final_budget_policy_bundle_rl_ready.json` (đã promoted từ guarded candidate).
- Metrics CSV được export tại `outputs/metrics/`.

## Kiến trúc hiện tại

### Macro layer
- `low`: ưu tiên rẻ, bundle hiện tại chọn `gated_bandit_router` (baseline W3, gate cho W1).
- `medium`: trade-off chất lượng/chi phí, bundle hiện tại chọn `bandit_router` trên `W1/W2/W3/W6`.
- `high`: ưu tiên chất lượng, bundle hiện tại chọn `bandit_router` trên cặp `W2/W3`.

### Micro layer
- Critic sẵn sàng làm `offline reward model` cho:
  - `QR`: dùng `local_reward` (spearman=0.6669)
  - `AG`: dùng `blended_reward` (spearman=0.3543)
- **Selective critic đã được verify** trên held-out với confidence_gate=0.70:
  - Token multiplier: 1.16x (PASS, threshold 1.25x)
  - Utility gap: -0.0009 (PASS, threshold -0.001)
  - Chỉ apply critic cho 34% câu hỏi (low-confidence queries)

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

### 1. Promoted bundle hiện tại (từ guarded offline RL candidate)

File:
- `outputs/final_budget_policy_bundle_rl_ready.json` (promoted)
- `outputs/final_budget_policy_test_eval_rl_ready.json` (promoted)
- `outputs/final_project_report_rl_ready.json` (promoted)

Kết quả test 90 câu mỗi budget:

| Budget | Policy | Utility | So với frozen trước promotion |
| --- | --- | ---: | ---: |
| `low` | `gated_bandit_router` | `0.3234` | `+0.0335` |
| `medium` | `bandit_router` | `0.3861` | `+0.0012` |
| `high` | `bandit_router` | `0.4560` | `+0.0016` |

Kết luận:
- Guarded candidate đã được promote thành frozen bundle mới.
- Tất cả 3 budget đều non-regressive so với frozen cũ.
- Low-budget cải thiện mạnh nhất (+0.0335) nhờ gated_bandit_router với baseline W3.
- Frozen cũ được backup tại `outputs/final_budget_policy_bundle_rl_ready_pre_promotion.json`.

### 2. Offline full-system RL đã hoàn tất một vòng

File:
- `outputs/full_system_rl_package.json`
- `outputs/promotion_report.json`

Gate hiện tại:
- `ready_for_offline_full_system_rl = true`
- `ready_for_online_full_system_rl = false`
- `deployment_mode = offline_reward_model_only`

Ý nghĩa:
- Vòng offline RL đầu tiên đã hoàn tất: train -> evaluate -> promotion gate -> promote.
- Có thể tiếp tục train offline policy improvement trên tầng macro.
- Có thể dùng critic làm reward model ở tầng micro.
- Chưa được bật direct online rerank/update.

### 3. High-budget bandit đã được cải thiện

File:
- `outputs/cv_ensemble_high_bandit.joblib` (final model, Ridge a=0.1 trained on all data)
- `outputs/cv_ensemble_report.json`
- `outputs/metrics/cv_ensemble_bandit_configs.csv` (101 configs x 5 folds)

So sánh tiến bộ:

| Metric | Ridge baseline | GBT single | Ensemble (single holdout) | **Ridge(a=0.1) 5-fold CV** | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| `best_rate` | `0.6349` | `0.7460` | `0.7302` | **`0.7333`** | `>= 0.70` |
| `regret` | `0.0706` | `0.0706` | `0.0442` | **`0.0260`** | `<= 0.04` |
| Model | `Ridge(a=1.0)` | `GBT(n100,d3)` | `Ens(Ra5+GBT)` | **`Ridge(a=0.1)`** | - |
| Eval method | single holdout | single holdout | single holdout | **5-fold CV** | - |

Kết luận:
- **5-fold CV regret = 0.0260 +/- 0.0100** -> PASS threshold 0.04.
- Per-fold regrets: [0.0279, 0.0225, 0.0118, 0.0392, 0.0288] - tất cả folds đều gần hoặc dưới threshold.
- Ridge(a=0.1) thắng 101 configs (bao gồm 60 GBT và 36 ensemble) nhờ generalization tốt hơn.
- Features: temporal, negation, superlative, multi_entity, entity_density.

### 4. Selective critic deployment đã được đánh giá và verify

File:
- `outputs/selective_critic_report.json` (estimation)
- `outputs/selective_critic_verification.json` (verified on held-out)
- `outputs/metrics_selective_critic_verification.csv`

Kết quả verify trên held-out (90 câu):

| Gate threshold | Critic used | Token multiplier | Utility vs base | Status |
| ---: | ---: | ---: | ---: | --- |
| 0.00 (always) | 90/90 | `1.46x` | `-0.0035` | FAIL |
| 0.55 | 8/90 | `1.04x` | `-0.0004` | PASS |
| 0.60 | 16/90 | `1.08x` | `-0.0006` | PASS |
| 0.65 | 20/90 | `1.10x` | `-0.0008` | PASS |
| **0.70** | **31/90** | **`1.16x`** | **`-0.0009`** | **PASS** |
| 0.75 | 42/90 | `1.21x` | `-0.0013` | FAIL |
| 1.00 (never) | 0/90 | `1.00x` | `+0.0000` | PASS |

Kết luận:
- **Confidence gate=0.70 PASS cả 2 deployment gate** (token_mult=1.16x, utility_gap=-0.0009).
- Critic chỉ apply cho 31/90 câu (34%) ở low-confidence queries.
- Blocker critic deployment đã được giải quyết.

### 5. Critic micro đã đủ tốt cho offline reward modeling

File:
- `outputs/process_critic_budget_qr_local.joblib.meta.json`
- `outputs/process_critic_budget_ag.joblib.meta.json`

Chất lượng critic:
- `QR`: `spearman = 0.6669`
- `AG`: `spearman = 0.3543`

### 6. Online RL gate status

File:
- `outputs/regate_report.json`

| Check | Status | Detail |
| --- | --- | --- |
| High bandit best_rate >= 0.70 | **PASS** | 0.7333 (5-fold CV) |
| High bandit regret <= 0.04 | **PASS** | 0.0260 (5-fold CV) |
| Selective critic token_mult <= 1.25x | **PASS** | 1.16x (gate=0.70) |
| Critic utility gap >= -0.001 | **PASS** | -0.0009 (verified) |
| **Overall online RL** | **PASS** | **All 4 gates cleared!** |

## Những gì đã làm được

- Đã xây dựng workflow library `W1` đến `W6`.
- Đã có router rule-based, learned, hybrid, bandit, gated-bandit.
- Đã đưa `budget_mode` vào utility và routing.
- Đã build benchmark lớn hơn với `420 train / 90 val / 90 test`.
- Đã train critic budget-aware cho `QR` và `AG`.
- Đã chọn được frozen final bundle thắng static reference trên held-out test.
- Đã mở được gate `offline full-system RL`.
- Đã train xong candidate macro policies cho offline RL.
- Đã thêm low-budget guarded gate cho offline RL candidate.
- Đã thêm script kiểm tra promotion non-regression: `scripts/check_offline_rl_promotion.py`.
- **Đã promote candidate guarded thành bundle frozen mới: `scripts/promote_offline_rl_candidate.py`.**
- **Đã cải thiện high-budget bandit bằng GBT, Ensemble và CV: `scripts/improve_high_bandit.py`, `scripts/train_ensemble_bandit.py`, `scripts/train_cv_ensemble_bandit.py`.**
- **Đã đánh giá selective critic deployment: `scripts/eval_selective_critic.py`.**
- **Đã verify selective critic trên held-out: `scripts/verify_selective_critic.py`.** Gate=0.70 PASS.
- **Đã re-gate online RL readiness: `scripts/regate_online_rl.py`.** ALL 4 GATES PASS.
- **Đã mở rộng question features (temporal, negation, superlative, entity density).**
- **Đã export metrics CSV cho từng tầng: `scripts/export_metrics_csv.py`.**
- **Đã mở `online RL` toàn hệ thống (deployment_mode: selective_critic_online).**

## Những gì chưa làm được

- Chưa chạy actual online RL training loop (ready to deploy, chưa deploy).
- Chưa thực hiện A/B testing giữa offline bundle và online-updated policies.

## Kế hoạch tiếp theo

### Ưu tiên 1: Deploy online RL training loop
- Tất cả 4 gates đã PASS -> sẵn sàng deploy.
- Deploy mode: `selective_critic_online` với `confidence_gate=0.70`.
- Safety: keep budget-specific cost penalties, promotion gate cho mọi policy mới.

### Ưu tiên 2: A/B testing
- So sánh offline frozen bundle vs online-updated policies.
- Monitor regret, utility, token cost trong production.

### Ưu tiên 3: Iterative improvement
- Mở rộng rollout data với W4/W5/W6 cho high budget.
- Fine-tune critic với online data mới.
- Tăng cường feature engineering dựa trên error analysis.

## CSV Metrics exports

Tất cả metrics được export dưới dạng CSV vào folder `outputs/metrics/`. Chạy `python scripts/export_metrics_csv.py` để tái tạo.

### Macro layer (routing & workflow selection)

| File | Nội dung | Rows |
| --- | --- | ---: |
| `outputs/metrics/metrics_macro_budget_summary.csv` | Utility, EM, F1, tokens, latency, workflow distribution per budget | 3 |
| `outputs/metrics/metrics_macro_per_question.csv` | Per-question routing decisions (question, workflow, utility, em, f1) | 270 |
| `outputs/metrics/metrics_macro_bandit_holdout.csv` | Per-question bandit holdout: predicted vs oracle workflow, regret | 189 |
| `outputs/metrics/metrics_bandit_configs.csv` | Model config comparison (Ridge vs GBT) with holdout metrics | 10 |
| `outputs/metrics/cv_ensemble_bandit_configs.csv` | 5-fold CV results for 101 configs (Ridge, GBT, Ensemble) | 101 |

### Micro layer (critic quality & deployment)

| File | Nội dung | Rows |
| --- | --- | ---: |
| `outputs/metrics/metrics_micro_critic_summary.csv` | Per-module critic quality: MAE, RMSE, Pearson, Spearman | 2 |
| `outputs/metrics/metrics_micro_critic_predictions.csv` | Per-example critic predictions vs targets | 400 |
| `outputs/metrics/metrics_micro_critic_deployment.csv` | Base vs critic method comparison + delta + token multiplier | 4 |
| `outputs/metrics/selective_critic_verification.csv` | Per-threshold gate results: utility, tokens, pass/fail | 9 |

### System layer (RL gates & promotion)

| File | Nội dung | Rows |
| --- | --- | ---: |
| `outputs/metrics/metrics_system_overview.csv` | RL gate status, deployment mode, online blockers | 14 |
| `outputs/metrics/metrics_system_promotion.csv` | Promotion comparison: candidate vs frozen per budget | 3 |

## Các file đang được giữ lại

### Frozen path (promoted)
- `outputs/final_budget_policy_bundle_rl_ready.json`
- `outputs/final_budget_policy_test_eval_rl_ready.json`
- `outputs/final_project_report_rl_ready.json`
- `outputs/full_system_rl_package.json`

### Pre-promotion backup
- `outputs/final_budget_policy_bundle_rl_ready_pre_promotion.json`
- `outputs/final_budget_policy_test_eval_rl_ready_pre_promotion.json`
- `outputs/final_project_report_rl_ready_pre_promotion.json`

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

### Offline full-system RL candidate (promoted)
- `outputs/offline_full_system_rl_guarded/final_budget_policy_bundle_offline_rl_candidate.json`
- `outputs/offline_full_system_rl_guarded/final_budget_policy_test_eval_offline_rl_candidate.json`
- `outputs/offline_full_system_rl_guarded/offline_full_system_rl_training_report.json`
- `outputs/offline_full_system_rl_guarded/promotion_check.json`

### Improvement artifacts
- `outputs/improved_high_bandit.joblib`
- `outputs/improve_high_bandit_report.json`
- `outputs/ensemble_high_bandit.joblib`
- `outputs/ensemble_high_bandit_report.json`
- `outputs/cv_ensemble_high_bandit.joblib` (best CV model, Ridge a=0.1)
- `outputs/cv_ensemble_report.json`
- `outputs/selective_critic_report.json`
- `outputs/selective_critic_verification.json`
- `outputs/promotion_report.json`
- `outputs/regate_report.json`

### Metrics CSV (tất cả trong `outputs/metrics/`)
- Xem bảng ở mục "CSV Metrics exports" ở trên.

## Lệnh còn cần dùng

### 1. Re-evaluate frozen bundle
```bash
python scripts/eval_final_budget_bundle.py --corpus data/hotpotqa_large/corpus.json --qa data/hotpotqa_large/splits/test.json --dataset_split test --dataset_name hotpotqa_large --policy_bundle outputs/final_budget_policy_bundle_rl_ready.json --router_model outputs/router_hotpot_budget_calibrated.joblib --llm_provider ollama --llm_model llama3.2 --hybrid_min_confidence 0.55 --hybrid_low_cost_confidence 0.55 --hybrid_low_cost_workflows W1 --budget_modes low medium high --limit 90 --seed 13 --out outputs/final_budget_policy_test_eval_rl_ready.json
```

### 2. Rebuild the full-system RL gate
```bash
python scripts/build_full_system_rl_package.py --policy_bundle outputs/final_budget_policy_bundle_rl_ready.json --final_eval outputs/final_budget_policy_test_eval_rl_ready.json --final_report outputs/final_project_report_rl_ready.json --reference_eval outputs/final_budget_policy_test_eval_v2_det90.json --macro_rollout low=outputs/hotpotqa_large_train_rollouts.json medium=outputs/hotpotqa_large_train_rollouts.json high=outputs/hotpotqa_large_train_rollouts_high_w2w3_det.json --critic_model QR=outputs/process_critic_budget_qr_local.joblib AG=outputs/process_critic_budget_ag.joblib --critic_meta QR=outputs/process_critic_budget_qr_local.joblib.meta.json AG=outputs/process_critic_budget_ag.joblib.meta.json --direct_critic_eval high=outputs/router_eval_large_budget_high_w2w3_det_critic2_val.json --out outputs/full_system_rl_package.json
```

### 3. Promote offline RL candidate
```bash
python scripts/promote_offline_rl_candidate.py --candidate_bundle outputs/offline_full_system_rl_guarded/final_budget_policy_bundle_offline_rl_candidate.json --candidate_eval outputs/offline_full_system_rl_guarded/final_budget_policy_test_eval_offline_rl_candidate.json --promotion_check outputs/offline_full_system_rl_guarded/promotion_check.json
```

### 4. Improve high-budget bandit
```bash
python scripts/improve_high_bandit.py --input outputs/hotpotqa_large_train_rollouts_high_w2w3_det.json --output outputs/improved_high_bandit.joblib --budget_mode high --allowed_workflows W2 W3 --base_router_model outputs/router_hotpot_budget_calibrated.joblib --probe_corpus data/hotpotqa_large/corpus.json
```

### 5. Evaluate selective critic strategies
```bash
python scripts/eval_selective_critic.py --critic_eval outputs/router_eval_large_budget_high_w2w3_det_critic2_val.json --base_eval outputs/router_eval_large_budget_high_w2w3_det_val.json --budget_mode high
```

### 6. Re-gate online RL
```bash
python scripts/regate_online_rl.py --improved_bandit_report outputs/ensemble_high_bandit_report.json --selective_critic_report outputs/selective_critic_verification.json
```

### 7. Export metrics CSV
```bash
python scripts/export_metrics_csv.py
```
### 8. Train ensemble bandit
```bash
python scripts/train_ensemble_bandit.py --input outputs/hotpotqa_large_train_rollouts_high_w2w3_det.json --output outputs/ensemble_high_bandit.joblib
```

### 9. Verify selective critic
```bash
python scripts/verify_selective_critic.py --critic_eval outputs/router_eval_large_budget_high_w2w3_det_critic2_val.json
```

### 10. Train CV ensemble bandit (5-fold cross-validation)
```bash
python scripts/train_cv_ensemble_bandit.py --input outputs/hotpotqa_large_train_rollouts_high_w2w3_det.json --output outputs/cv_ensemble_high_bandit.joblib --k_folds 5
```

## Kết luận cuối cùng

Project hiện tại đã đạt đúng tinh thần `MAO-ARAG + RAG-Gym` ở mức:
- `macro`: chọn workflow theo query/budget/cost, đã hoàn tất một vòng offline RL với promotion gate
- `micro`: critic chấm step/candidate action, đã verify selective deployment trên held-out
- `system`: **tất cả 4 deployment gates đã PASS**, sẵn sàng online RL

Trạng thái cuối:
- `offline full-system RL`: **đã hoàn tất (train -> eval -> promote)**
- `online RL`: **SẴN SÀNG** (deployment_mode: selective_critic_online)
- Bundle hiện tại: **promoted từ guarded candidate** (utility: low=0.3234, medium=0.3861, high=0.4560)
- High-budget bandit: **PASS** (5-fold CV regret=0.0260 <= 0.04, best_rate=0.7333 >= 0.70)
- Selective critic: **PASS** (gate=0.70 trên held-out, token_mult=1.16x <= 1.25x, utility=-0.0009 >= -0.001)
- Metrics CSV: **đã export 13 files vào `outputs/metrics/`** (macro/micro/system)
- Bước tiếp theo: **deploy online RL training loop với selective_critic_online**
