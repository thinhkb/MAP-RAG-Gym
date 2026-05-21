# MAP-RAG-Gym

> **Adaptive Multi-Workflow RAG with Process Evaluation**  
> Kết hợp `MAO-ARAG` (macro workflow routing) và `RAG-Gym` (micro process critic) để xây dựng hệ thống RAG thích nghi theo budget.

---

## Chay toan bo pipeline (1 lenh)

Script `run_all.ps1` (Windows) va `run_all.sh` (Linux/Mac) tu dong chay toan bo 9 phase theo thu tu, bo qua cac buoc da co output, va log tat ca ra `outputs/run_all.log`.

**Buoc 1:** Mo file `run_all.ps1`, chinh LLM o dau file:

```
$LLM_PROVIDER = "ollama"    # hoac "gemini"
$LLM_MODEL    = "llama3.2"  # hoac "gemini-2.0-flash"
```

**Buoc 2:** Chay:

```powershell
# Windows (PowerShell):
.venv\Scripts\activate
.\run_all.ps1
```

```bash
# Linux / Mac:
source .venv/bin/activate
chmod +x run_all.sh && ./run_all.sh
```

> Uoc tinh thoi gian toan bo voi Ollama llama3.2:
> Phase 1 Rollout (420 cau x2): ~3-6 gio | Phase 4-5 Eval: ~1-2 gio | Phase 6 CV bandit: ~2 gio | **Tong: 6-10 gio**
>
> De test nhanh: doi `$TRAIN_LIMIT = 20` va `$EVAL_LIMIT = 10` trong script (~15 phut).

Ket qua CSV de bao cao (trong `outputs/metrics/`):

| File | Noi dung |
|---|---|
| `metrics_macro_budget_summary.csv` | Utility per budget (low/medium/high) |
| `cv_ensemble_bandit_configs.csv` | Bandit regret & best_rate (5-fold CV) |
| `metrics_micro_critic_summary.csv` | Critic Spearman QR/AG |
| `metrics_selective_critic_verification.csv` | Selective critic gate table |
| `metrics_system_overview.csv` | Online RL gate status |

---

## Mục lục

1. [Yêu cầu hệ thống](#1-yêu-cầu-hệ-thống)
2. [Cài đặt môi trường](#2-cài-đặt-môi-trường)
3. [Cấu hình API & LLM](#3-cấu-hình-api--llm)
4. [Chuẩn bị dữ liệu](#4-chuẩn-bị-dữ-liệu)
5. [Phase 1 — Rollout & Benchmark](#5-phase-1--rollout--benchmark)
6. [Phase 2 — Train Router](#6-phase-2--train-router)
7. [Phase 3 — Train Critic (Micro Layer)](#7-phase-3--train-critic-micro-layer)
8. [Phase 4 — Build & Evaluate Budget Bundle](#8-phase-4--build--evaluate-budget-bundle)
9. [Phase 5 — Offline Full-System RL](#9-phase-5--offline-full-system-rl)
10. [Phase 6 — Improve High-Budget Bandit](#10-phase-6--improve-high-budget-bandit)
11. [Phase 7 — Selective Critic Verification](#11-phase-7--selective-critic-verification)
12. [Phase 8 — Re-gate & Online RL](#12-phase-8--re-gate--online-rl)
13. [Export Metrics CSV](#13-export-metrics-csv)
14. [Chạy nhanh (Quick Run)](#14-chạy-nhanh-quick-run)
15. [Cấu trúc thư mục](#15-cấu-trúc-thư-mục)
16. [Ghi chú & Troubleshooting](#16-ghi-chú--troubleshooting)

---

## 1. Yêu cầu hệ thống

| Thành phần | Yêu cầu tối thiểu |
|---|---|
| Python | >= 3.10 |
| RAM | >= 8 GB |
| Ollama | >= 0.3 (để dùng local LLM) |
| Gemini API | Key hợp lệ (nếu dùng `--llm_provider gemini`) |

### Cài Ollama (local LLM backend)

```bash
# Windows: tải từ https://ollama.com/download
# Sau khi cài, pull model:
ollama pull llama3.2
```

---

## 2. Cài đặt môi trường

```bash
# 1. Clone repo (nếu chưa có)
git clone <repo-url>
cd MAP-RAG-Gym

# 2. Tạo virtual environment
python -m venv .venv

# 3. Kích hoạt venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 4. Cài đặt package và dependencies
pip install -e .
```

> **Lưu ý:** Lệnh `pip install -e .` sẽ cài tất cả dependencies từ `pyproject.toml` (pydantic, scikit-learn, rank-bm25, numpy, pandas, tqdm, requests, python-dotenv, datasets, PyYAML).

---

## 3. Cấu hình API & LLM

Tạo file `.env` ở thư mục gốc (hoặc chỉnh sửa file có sẵn):

```env
GEMINI_API_KEY=<your-gemini-api-key-here>
OLLAMA_BASE_URL=http://localhost:11434
```

> - Dùng **Ollama** (local, miễn phí): thêm `--llm_provider ollama --llm_model llama3.2` vào các lệnh.  
> - Dùng **Gemini API**: thêm `--llm_provider gemini --llm_model gemini-2.0-flash` vào các lệnh.

---

## 4. Chuẩn bị dữ liệu

Dữ liệu đã có sẵn trong thư mục `data/`. **Không cần tải thêm.**

```
data/
├── hotpotqa/               # Dataset nhỏ (~100 QA)
│   ├── corpus.json
│   ├── qa.json
│   └── splits/
│       ├── train.json
│       ├── val.json
│       └── test.json
├── hotpotqa_large/         # Dataset lớn (~600 QA) — dùng cho training chính
│   ├── corpus.json
│   ├── qa.json
│   └── splits/
│       ├── train.json      # 420 câu
│       ├── val.json        # 90 câu
│       └── test.json       # 90 câu
└── sample/                 # Dataset mẫu rất nhỏ (~5 QA) để test nhanh
    ├── corpus.json
    └── qa.json
```

*(Tùy chọn)* Nếu muốn tái tạo splits từ đầu:
```bash
python scripts/prepare_hotpotqa.py
python scripts/create_qa_splits.py
```

---

## 5. Phase 1 — Rollout & Benchmark

### 5.1. Demo nhanh với dataset sample

```bash
python scripts/run_phase1_demo.py
```

### 5.2. Batch rollout toàn bộ training set (dataset lớn)

Đây là bước **quan trọng nhất** — sinh rollout data cho toàn bộ workflow. Quá trình này **mất nhiều thời gian** (có thể hàng giờ tùy LLM backend).

```bash
# Rollout cho medium/low budget (W1, W2, W3, W6)
python scripts/batch_rollout.py \
  --corpus data/hotpotqa_large/corpus.json \
  --qa data/hotpotqa_large/splits/train.json \
  --dataset_name hotpotqa_large \
  --llm_provider ollama \
  --llm_model llama3.2 \
  --workflows W1 W2 W3 W6 \
  --budget_modes low medium high \
  --limit 420 \
  --seed 42 \
  --out outputs/hotpotqa_large_train_rollouts.json
```

```bash
# Rollout riêng cho high budget (W2, W3) với deterministic sampling
python scripts/batch_rollout.py \
  --corpus data/hotpotqa_large/corpus.json \
  --qa data/hotpotqa_large/splits/train.json \
  --dataset_name hotpotqa_large \
  --llm_provider ollama \
  --llm_model llama3.2 \
  --workflows W2 W3 \
  --budget_modes high \
  --limit 420 \
  --seed 42 \
  --out outputs/hotpotqa_large_train_rollouts_high_w2w3_det.json
```

> **Tip:** Thêm `--limit 20` để test nhanh trước khi chạy full.

### 5.3. Build process dataset (cho critic training)

```bash
python scripts/build_process_dataset.py \
  --rollout outputs/hotpotqa_large_train_rollouts.json \
  --out outputs/hotpotqa_large_process_train_budget.json
```

---

## 6. Phase 2 — Train Router

### 6.1. Train bandit router

```bash
python scripts/train_bandit_router.py \
  --input outputs/hotpotqa_large_train_rollouts.json \
  --budget_mode low medium high \
  --out_prefix outputs/bandit_router
```

### 6.2. Train meta router

```bash
python scripts/train_meta_router.py \
  --input outputs/hotpotqa_large_train_rollouts.json \
  --out outputs/router_hotpot_budget_calibrated.joblib
```

### 6.3. (Tùy chọn) Tune router

```bash
# Tune hybrid router
python scripts/tune_hybrid_router.py \
  --rollout outputs/hotpotqa_large_train_rollouts.json

# Tune switch policy
python scripts/tune_switch_policy.py \
  --rollout outputs/hotpotqa_large_train_rollouts.json

# Tune bandit gate
python scripts/tune_bandit_gate.py \
  --rollout outputs/hotpotqa_large_train_rollouts.json

# Tune workflow retrievers
python scripts/tune_workflow_retrievers.py \
  --corpus data/hotpotqa_large/corpus.json \
  --qa data/hotpotqa_large/splits/val.json
```

---

## 7. Phase 3 — Train Critic (Micro Layer)

```bash
# Train critic cho QR (query rewrite)
python scripts/train_process_critic.py \
  --input outputs/hotpotqa_large_process_train_budget.json \
  --module QR \
  --reward_type local \
  --out outputs/process_critic_budget_qr_local.joblib

# Train critic cho AG (answer generation)
python scripts/train_process_critic.py \
  --input outputs/hotpotqa_large_process_train_budget.json \
  --module AG \
  --reward_type blended \
  --out outputs/process_critic_budget_ag.joblib
```

---

## 8. Phase 4 — Build & Evaluate Budget Bundle

### 8.1. Select best policy per budget

```bash
python scripts/select_budget_policy.py \
  --rollout outputs/hotpotqa_large_train_rollouts.json \
  --router_model outputs/router_hotpot_budget_calibrated.joblib \
  --out outputs/final_budget_policy_bundle.json
```

### 8.2. Evaluate trên validation set

```bash
python scripts/eval_phase4_router.py \
  --corpus data/hotpotqa_large/corpus.json \
  --qa data/hotpotqa_large/splits/val.json \
  --dataset_split val \
  --dataset_name hotpotqa_large \
  --policy_bundle outputs/final_budget_policy_bundle.json \
  --router_model outputs/router_hotpot_budget_calibrated.joblib \
  --llm_provider ollama \
  --llm_model llama3.2 \
  --budget_modes low medium high \
  --limit 90 \
  --seed 13 \
  --out outputs/router_eval_large_budget_val.json
```

### 8.3. Train phase4 router

```bash
python scripts/train_phase4_router.py \
  --rollout outputs/hotpotqa_large_train_rollouts.json \
  --out_bundle outputs/final_budget_policy_bundle.json
```

### 8.4. Evaluate trên test set (frozen bundle)

```bash
python scripts/eval_final_budget_bundle.py \
  --corpus data/hotpotqa_large/corpus.json \
  --qa data/hotpotqa_large/splits/test.json \
  --dataset_split test \
  --dataset_name hotpotqa_large \
  --policy_bundle outputs/final_budget_policy_bundle.json \
  --router_model outputs/router_hotpot_budget_calibrated.joblib \
  --llm_provider ollama \
  --llm_model llama3.2 \
  --hybrid_min_confidence 0.55 \
  --hybrid_low_cost_confidence 0.55 \
  --hybrid_low_cost_workflows W1 \
  --budget_modes low medium high \
  --limit 90 \
  --seed 13 \
  --out outputs/final_budget_policy_test_eval.json
```

---

## 9. Phase 5 — Offline Full-System RL

### 9.1. Build full-system RL package

```bash
python scripts/build_full_system_rl_package.py \
  --policy_bundle outputs/final_budget_policy_bundle.json \
  --final_eval outputs/final_budget_policy_test_eval.json \
  --final_report outputs/final_project_report.json \
  --macro_rollout \
      low=outputs/hotpotqa_large_train_rollouts.json \
      medium=outputs/hotpotqa_large_train_rollouts.json \
      high=outputs/hotpotqa_large_train_rollouts_high_w2w3_det.json \
  --critic_model \
      QR=outputs/process_critic_budget_qr_local.joblib \
      AG=outputs/process_critic_budget_ag.joblib \
  --critic_meta \
      QR=outputs/process_critic_budget_qr_local.joblib.meta.json \
      AG=outputs/process_critic_budget_ag.joblib.meta.json \
  --out outputs/full_system_rl_package.json
```

### 9.2. Train offline RL

```bash
python scripts/train_offline_full_system_rl.py \
  --rl_package outputs/full_system_rl_package.json \
  --rollout outputs/hotpotqa_large_train_rollouts.json \
  --out_dir outputs/offline_full_system_rl_guarded
```

### 9.3. Evaluate RL candidate

```bash
python scripts/eval_final_budget_bundle.py \
  --corpus data/hotpotqa_large/corpus.json \
  --qa data/hotpotqa_large/splits/test.json \
  --dataset_split test \
  --dataset_name hotpotqa_large \
  --policy_bundle outputs/offline_full_system_rl_guarded/final_budget_policy_bundle_offline_rl_candidate.json \
  --router_model outputs/router_hotpot_budget_calibrated.joblib \
  --llm_provider ollama \
  --llm_model llama3.2 \
  --budget_modes low medium high \
  --limit 90 \
  --seed 13 \
  --out outputs/offline_full_system_rl_guarded/final_budget_policy_test_eval_offline_rl_candidate.json
```

### 9.4. Kiểm tra promotion gate

```bash
python scripts/check_offline_rl_promotion.py \
  --candidate_eval outputs/offline_full_system_rl_guarded/final_budget_policy_test_eval_offline_rl_candidate.json \
  --frozen_eval outputs/final_budget_policy_test_eval.json \
  --out outputs/offline_full_system_rl_guarded/promotion_check.json
```

### 9.5. Promote candidate (nếu gate PASS)

```bash
python scripts/promote_offline_rl_candidate.py \
  --candidate_bundle outputs/offline_full_system_rl_guarded/final_budget_policy_bundle_offline_rl_candidate.json \
  --candidate_eval outputs/offline_full_system_rl_guarded/final_budget_policy_test_eval_offline_rl_candidate.json \
  --promotion_check outputs/offline_full_system_rl_guarded/promotion_check.json
```

> Sau khi promote, bundle hiện tại sẽ được lưu vào `outputs/final_budget_policy_bundle_rl_ready.json`.

---

## 10. Phase 6 — Improve High-Budget Bandit

### 10.1. Train improved bandit (Ridge)

```bash
python scripts/improve_high_bandit.py \
  --input outputs/hotpotqa_large_train_rollouts_high_w2w3_det.json \
  --output outputs/improved_high_bandit.joblib \
  --budget_mode high \
  --allowed_workflows W2 W3 \
  --base_router_model outputs/router_hotpot_budget_calibrated.joblib \
  --probe_corpus data/hotpotqa_large/corpus.json
```

### 10.2. Train ensemble bandit

```bash
python scripts/train_ensemble_bandit.py \
  --input outputs/hotpotqa_large_train_rollouts_high_w2w3_det.json \
  --output outputs/ensemble_high_bandit.joblib
```

### 10.3. Train CV ensemble bandit (5-fold, **recommended**)

```bash
python scripts/train_cv_ensemble_bandit.py \
  --input outputs/hotpotqa_large_train_rollouts_high_w2w3_det.json \
  --output outputs/cv_ensemble_high_bandit.joblib \
  --k_folds 5
```

> Mục tiêu: `cv_regret <= 0.04` và `best_rate >= 0.70`.

---

## 11. Phase 7 — Selective Critic Verification

### 11.1. Eval selective critic strategies

Cần có evaluation với critic trước:

```bash
python scripts/eval_phase4_router.py \
  --corpus data/hotpotqa_large/corpus.json \
  --qa data/hotpotqa_large/splits/val.json \
  --dataset_split val \
  --dataset_name hotpotqa_large \
  --policy_bundle outputs/final_budget_policy_bundle_rl_ready.json \
  --router_model outputs/router_hotpot_budget_calibrated.joblib \
  --critic_model outputs/process_critic_budget_ag.joblib \
  --llm_provider ollama \
  --llm_model llama3.2 \
  --budget_modes high \
  --limit 90 \
  --seed 13 \
  --out outputs/router_eval_large_budget_high_w2w3_det_critic2_val.json
```

```bash
python scripts/eval_selective_critic.py \
  --critic_eval outputs/router_eval_large_budget_high_w2w3_det_critic2_val.json \
  --base_eval outputs/router_eval_large_budget_high_w2w3_det_val.json \
  --budget_mode high
```

### 11.2. Verify selective critic trên held-out

```bash
python scripts/verify_selective_critic.py \
  --critic_eval outputs/router_eval_large_budget_high_w2w3_det_critic2_val.json
```

> Mục tiêu: `token_multiplier <= 1.25x` và `utility_gap >= -0.001` tại `confidence_gate=0.70`.

---

## 12. Phase 8 — Re-gate & Online RL

```bash
python scripts/regate_online_rl.py \
  --improved_bandit_report outputs/cv_ensemble_report.json \
  --selective_critic_report outputs/selective_critic_verification.json
```

> Nếu tất cả 4 gates PASS, hệ thống sẽ chuyển sang `deployment_mode: selective_critic_online`.

---

## 13. Export Metrics CSV

```bash
python scripts/export_metrics_csv.py
```

Kết quả được lưu vào `outputs/metrics/` gồm:

| File | Nội dung |
|---|---|
| `metrics_macro_budget_summary.csv` | Utility, EM, F1, tokens, latency theo budget |
| `metrics_macro_per_question.csv` | Per-question routing decisions |
| `metrics_macro_bandit_holdout.csv` | Bandit holdout: predicted vs oracle |
| `metrics_bandit_configs.csv` | So sánh config Ridge vs GBT |
| `cv_ensemble_bandit_configs.csv` | 5-fold CV results (101 configs) |
| `metrics_micro_critic_summary.csv` | MAE, RMSE, Pearson, Spearman theo module |
| `metrics_micro_critic_predictions.csv` | Per-example critic predictions |
| `metrics_micro_critic_deployment.csv` | Base vs critic method comparison |
| `selective_critic_verification.csv` | Per-threshold gate results |
| `metrics_system_overview.csv` | RL gate status, deployment mode |
| `metrics_system_promotion.csv` | Promotion comparison per budget |

---

## 14. Chạy nhanh (Quick Run)

Nếu bạn muốn kiểm tra toàn bộ pipeline nhanh với dataset nhỏ:

```bash
# Bước 1: Demo phase 1
python scripts/run_phase1_demo.py

# Bước 2: Demo phase 2 router
python scripts/run_phase2_router_demo.py

# Bước 3: Demo phase 3 evaluate
python scripts/run_phase3_evaluate.py
```

---

## 15. Cấu trúc thư mục

```
MAP-RAG-Gym/
├── .env                        # API keys (tạo thủ công)
├── .gitignore
├── pyproject.toml              # Package config & dependencies
├── plan.md                     # Chi tiết kỹ thuật và kết quả
├── README.md                   # File này
│
├── data/
│   ├── hotpotqa/               # Dataset nhỏ
│   ├── hotpotqa_large/         # Dataset lớn (dùng chính)
│   └── sample/                 # Dataset mẫu
│
├── scripts/                    # Tất cả scripts chạy pipeline
│   ├── batch_rollout.py        # Sinh rollout data
│   ├── train_*.py              # Train các model
│   ├── eval_*.py               # Evaluate
│   ├── build_*.py              # Build packages/bundles
│   ├── tune_*.py               # Hyperparameter tuning
│   ├── check_*.py              # Kiểm tra promotion gates
│   ├── promote_*.py            # Promote candidate
│   ├── verify_*.py             # Verify deployment
│   ├── regate_online_rl.py     # Re-gate online RL
│   ├── export_metrics_csv.py   # Export metrics
│   └── run_phase*.py           # Demo scripts
│
├── src/
│   └── map_rag_gym/            # Core package
│       ├── core/               # Workflow definitions (W1-W6)
│       ├── router/             # Router & bandit logic
│       ├── critic/             # Process critic
│       ├── retrieval/          # BM25 + retrieval backends
│       ├── executors/          # QDS, QDP, QR, RA, DS, AG, AS
│       ├── evaluation/         # Metrics (F1, EM, utility)
│       ├── llm/                # LLM backends (Ollama, Gemini)
│       ├── prompts/            # Prompt templates
│       └── utils/              # Helpers
│
└── outputs/                    # Kết quả sinh ra (được tạo khi chạy)
    ├── *.json                  # Bundles, reports, eval results
    ├── *.joblib                # Trained models
    ├── offline_full_system_rl_guarded/
    └── metrics/                # CSV exports
```

---

## 16. Ghi chú & Troubleshooting

### Ollama không kết nối được

```bash
# Kiểm tra Ollama đang chạy
ollama list

# Khởi động lại Ollama
ollama serve
```

### Lỗi import `map_rag_gym`

```bash
# Đảm bảo đã cài package ở edit mode
pip install -e .

# Kiểm tra venv đã activate
.venv\Scripts\activate   # Windows
```

### Thiếu file trong `outputs/`

Các file trong `outputs/` được tạo ra khi chạy pipeline. Nếu thiếu file nào, hãy chạy lại bước tương ứng theo thứ tự trong README này. Thứ tự phụ thuộc:

```
batch_rollout → build_process_dataset → train_* → eval_* → build_full_system_rl_package → train_offline_rl → promote → improve_bandit → verify_critic → regate
```

### Muốn dùng Gemini thay Ollama

Thay `--llm_provider ollama --llm_model llama3.2` bằng:
```bash
--llm_provider gemini --llm_model gemini-2.0-flash
```

Đảm bảo `GEMINI_API_KEY` đã được set trong `.env`.

### Chạy nhanh để test (không tốn thời gian)

Thêm `--limit 5` vào bất kỳ lệnh nào để chỉ xử lý 5 câu hỏi đầu tiên.

---

*Xem chi tiết kỹ thuật, kết quả thực nghiệm và kiến trúc hệ thống tại [plan.md](plan.md).*
