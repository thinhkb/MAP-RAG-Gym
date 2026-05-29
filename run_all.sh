#!/usr/bin/env bash
# =============================================================================
# MAP-RAG-Gym - Full Pipeline (Bash)
# Su dung: chmod +x run_all.sh && ./run_all.sh
# =============================================================================
set -euo pipefail

# â”€â”€ CAU HINH â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
LLM_PROVIDER="ollama"
LLM_MODEL="llama3.2"
TRAIN_LIMIT=420   # Doi thanh 20 de test nhanh
EVAL_LIMIT=90     # Doi thanh 10 de test nhanh
SEED=42
EVAL_SEED=13
CV_FOLDS=5
DATASET_NAME="2wikimultihopqa"
DATA_DIR="data/${DATASET_NAME}"
OUT_DIR="outputs/${DATASET_NAME}"
# --- HET CAU HINH ------------------------------------------------------------

# Tu dong tim python trong venv (ho tro Windows Git Bash va Linux/Mac)
if [[ -f ".venv/Scripts/python" ]]; then
    PY=".venv/Scripts/python"
elif [[ -f ".venv/bin/python" ]]; then
    PY=".venv/bin/python"
else
    echo "ERROR: Khong tim thay python trong .venv/"
    echo "Chay: python -m venv .venv && pip install -e ."
    exit 1
fi
echo "Using Python: $PY"

LOG="${OUT_DIR}/run_all.log"
STEP_NUM=0
START_TOTAL=$SECONDS

mkdir -p ${OUT_DIR}/offline_full_system_rl_guarded ${OUT_DIR}/metrics

# â”€â”€ Helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
log() {
    local ts; ts=$(date "+%H:%M:%S")
    local line="[$ts] $*"
    echo "$line"
    echo "$line" >> "$LOG"
}

step() {
    local name="$1"
    local out_file="$2"
    shift 2
    STEP_NUM=$((STEP_NUM + 1))
    if [[ -n "$out_file" && -f "$out_file" ]]; then
        log "[SKIP] Buoc $STEP_NUM - $name (da co: $out_file)"
        return 0
    fi
    log ""
    log "======================================================"
    log "  Buoc $STEP_NUM - $name"
    log "======================================================"
    local t0=$SECONDS
    if ! "$@"; then
        log "[LOI] Buoc $STEP_NUM that bai!"
        exit 1
    fi
    log "[OK] Buoc $STEP_NUM hoan thanh trong $((SECONDS - t0))s"
}

log "======================================================"
log "   MAP-RAG-Gym Full Pipeline | $LLM_PROVIDER/$LLM_MODEL"
log "   Train=$TRAIN_LIMIT | Eval=$EVAL_LIMIT"
log "======================================================"

# =============================================================================
# PHASE 1 - ROLLOUT
# =============================================================================

step "Rollout W1/W2/W3/W6 medium budget" \
    "${OUT_DIR}/hotpotqa_large_train_rollouts.json" \
    "$PY" scripts/batch_rollout.py \
    --corpus ${DATA_DIR}/corpus.json \
    --qa ${DATA_DIR}/splits/train.json \
    --dataset_name "$DATASET_NAME" \
    --dataset_split train \
    --llm_provider "$LLM_PROVIDER" \
    --llm_model "$LLM_MODEL" \
    --workflow_ids W1 W2 W3 W6 \
    --budget_mode medium \
    --limit "$TRAIN_LIMIT" \
    --seed "$SEED" \
    --out ${OUT_DIR}/hotpotqa_large_train_rollouts.json \
    --resume

step "Rollout W2/W3 high budget det" \
    "${OUT_DIR}/hotpotqa_large_train_rollouts_high_w2w3_det.json" \
    "$PY" scripts/batch_rollout.py \
    --corpus ${DATA_DIR}/corpus.json \
    --qa ${DATA_DIR}/splits/train.json \
    --dataset_name "$DATASET_NAME" \
    --dataset_split train \
    --llm_provider "$LLM_PROVIDER" \
    --llm_model "$LLM_MODEL" \
    --workflow_ids W2 W3 \
    --budget_mode high \
    --limit "$TRAIN_LIMIT" \
    --seed "$SEED" \
    --out ${OUT_DIR}/hotpotqa_large_train_rollouts_high_w2w3_det.json \
    --resume

step "Build process dataset" \
    "${OUT_DIR}/hotpotqa_large_process_train_budget.json" \
    "$PY" scripts/build_process_dataset.py \
    --input ${OUT_DIR}/hotpotqa_large_train_rollouts.json \
    --out ${OUT_DIR}/hotpotqa_large_process_train_budget.json

# =============================================================================
# PHASE 2 - TRAIN ROUTER
# Phase4 router (LearnedRouter) phai chay TRUOC meta router vi meta router can no
# =============================================================================

step "Train phase4 router (LearnedRouter budget-aware)" \
    "${OUT_DIR}/router_hotpot_phase4.joblib" \
    "$PY" scripts/train_phase4_router.py \
    --input ${OUT_DIR}/hotpotqa_large_train_rollouts.json \
    --output ${OUT_DIR}/router_hotpot_phase4.joblib \
    --budget_modes low medium high \
    --allowed_workflows W1 W2 W3 W6 \
    --seed "$EVAL_SEED"

step "Train meta router gate" \
    "${OUT_DIR}/router_hotpot_budget_calibrated.joblib" \
    "$PY" scripts/train_meta_router.py \
    --rollouts ${OUT_DIR}/hotpotqa_large_train_rollouts.json \
    --router_model ${OUT_DIR}/router_hotpot_phase4.joblib \
    --output ${OUT_DIR}/router_hotpot_budget_calibrated.joblib

# =============================================================================
# PHASE 3 - TRAIN CRITIC
# train_process_critic.py args: --input --output --target --modules
# QR: --target local_reward  --modules QR
# AG: --target blended_reward --modules AG
# =============================================================================

step "Train critic QR (local_reward)" \
    "${OUT_DIR}/process_critic_budget_qr_local.joblib" \
    "$PY" scripts/train_process_critic.py \
    --input ${OUT_DIR}/hotpotqa_large_process_train_budget.json \
    --output ${OUT_DIR}/process_critic_budget_qr_local.joblib \
    --target local_reward \
    --modules QR

step "Train critic AG (blended_reward)" \
    "${OUT_DIR}/process_critic_budget_ag.joblib" \
    "$PY" scripts/train_process_critic.py \
    --input ${OUT_DIR}/hotpotqa_large_process_train_budget.json \
    --output ${OUT_DIR}/process_critic_budget_ag.joblib \
    --target blended_reward \
    --modules AG

# =============================================================================
# PHASE 4 - EVALUATE & BUILD BUNDLE
# eval_phase4_router.py args: --corpus --qa --router_model (required)
#   --budget_mode (singular!) --out
# Chay eval cho tung budget rieng biet, sau do select va ghep bundle
# =============================================================================

step "Eval all methods val set (low)" \
    "${OUT_DIR}/router_eval_large_budget_low_val.json" \
    "$PY" scripts/eval_phase4_router.py \
    --corpus ${DATA_DIR}/corpus.json \
    --qa ${DATA_DIR}/splits/val.json \
    --dataset_split val \
    --dataset_name "$DATASET_NAME" \
    --router_model ${OUT_DIR}/router_hotpot_phase4.joblib \
    --llm_provider "$LLM_PROVIDER" \
    --llm_model "$LLM_MODEL" \
    --budget_mode low \
    --limit "$EVAL_LIMIT" \
    --seed "$EVAL_SEED" \
    --out ${OUT_DIR}/router_eval_large_budget_low_val.json

step "Eval all methods val set (medium)" \
    "${OUT_DIR}/router_eval_large_budget_medium_val.json" \
    "$PY" scripts/eval_phase4_router.py \
    --corpus ${DATA_DIR}/corpus.json \
    --qa ${DATA_DIR}/splits/val.json \
    --dataset_split val \
    --dataset_name "$DATASET_NAME" \
    --router_model ${OUT_DIR}/router_hotpot_phase4.joblib \
    --llm_provider "$LLM_PROVIDER" \
    --llm_model "$LLM_MODEL" \
    --budget_mode medium \
    --limit "$EVAL_LIMIT" \
    --seed "$EVAL_SEED" \
    --out ${OUT_DIR}/router_eval_large_budget_medium_val.json

step "Eval all methods val set (high)" \
    "${OUT_DIR}/router_eval_large_budget_high_val.json" \
    "$PY" scripts/eval_phase4_router.py \
    --corpus ${DATA_DIR}/corpus.json \
    --qa ${DATA_DIR}/splits/val.json \
    --dataset_split val \
    --dataset_name "$DATASET_NAME" \
    --router_model ${OUT_DIR}/router_hotpot_phase4.joblib \
    --llm_provider "$LLM_PROVIDER" \
    --llm_model "$LLM_MODEL" \
    --budget_mode high \
    --limit "$EVAL_LIMIT" \
    --seed "$EVAL_SEED" \
    --out ${OUT_DIR}/router_eval_large_budget_high_val.json

step "Select best policy (low)" \
    "${OUT_DIR}/policy_selection_low.json" \
    "$PY" scripts/select_budget_policy.py \
    --input ${OUT_DIR}/router_eval_large_budget_low_val.json \
    --out ${OUT_DIR}/policy_selection_low.json

step "Select best policy (medium)" \
    "${OUT_DIR}/policy_selection_medium.json" \
    "$PY" scripts/select_budget_policy.py \
    --input ${OUT_DIR}/router_eval_large_budget_medium_val.json \
    --out ${OUT_DIR}/policy_selection_medium.json

step "Select best policy (high)" \
    "${OUT_DIR}/policy_selection_high.json" \
    "$PY" scripts/select_budget_policy.py \
    --input ${OUT_DIR}/router_eval_large_budget_high_val.json \
    --out ${OUT_DIR}/policy_selection_high.json

step "Build final budget policy bundle" \
    "${OUT_DIR}/final_budget_policy_bundle.json" \
    "$PY" scripts/build_final_budget_policy_bundle.py \
    --policies \
        ${OUT_DIR}/policy_selection_low.json \
        ${OUT_DIR}/policy_selection_medium.json \
        ${OUT_DIR}/policy_selection_high.json \
    --out ${OUT_DIR}/final_budget_policy_bundle.json

step "Eval frozen bundle tren test set" \
    "${OUT_DIR}/final_budget_policy_test_eval.json" \
    "$PY" scripts/eval_final_budget_bundle.py \
    --corpus ${DATA_DIR}/corpus.json \
    --qa ${DATA_DIR}/splits/test.json \
    --dataset_split test \
    --dataset_name "$DATASET_NAME" \
    --policy_bundle ${OUT_DIR}/final_budget_policy_bundle.json \
    --router_model ${OUT_DIR}/router_hotpot_phase4.joblib \
    --llm_provider "$LLM_PROVIDER" \
    --llm_model "$LLM_MODEL" \
    --hybrid_min_confidence 0.55 \
    --hybrid_low_cost_confidence 0.55 \
    --hybrid_low_cost_workflows W1 \
    --budget_modes low medium high \
    --limit "$EVAL_LIMIT" \
    --seed "$EVAL_SEED" \
    --out ${OUT_DIR}/final_budget_policy_test_eval.json

step "Build final project report" \
    "${OUT_DIR}/final_project_report.json" \
    "$PY" scripts/build_final_project_report.py \
    --policy_bundle ${OUT_DIR}/final_budget_policy_bundle.json \
    --test_eval ${OUT_DIR}/final_budget_policy_test_eval.json \
    --out ${OUT_DIR}/final_project_report.json

# =============================================================================
# PHASE 5 - OFFLINE FULL-SYSTEM RL
# =============================================================================

step "Build full-system RL package" \
    "${OUT_DIR}/full_system_rl_package.json" \
    "$PY" scripts/build_full_system_rl_package.py \
    --policy_bundle ${OUT_DIR}/final_budget_policy_bundle.json \
    --final_eval ${OUT_DIR}/final_budget_policy_test_eval.json \
    --final_report ${OUT_DIR}/final_project_report.json \
    --macro_rollout \
        low=${OUT_DIR}/hotpotqa_large_train_rollouts.json \
        medium=${OUT_DIR}/hotpotqa_large_train_rollouts.json \
        high=${OUT_DIR}/hotpotqa_large_train_rollouts_high_w2w3_det.json \
    --critic_model \
        QR=${OUT_DIR}/process_critic_budget_qr_local.joblib \
        AG=${OUT_DIR}/process_critic_budget_ag.joblib \
    --critic_meta \
        QR=${OUT_DIR}/process_critic_budget_qr_local.joblib.meta.json \
        AG=${OUT_DIR}/process_critic_budget_ag.joblib.meta.json \
    --out ${OUT_DIR}/full_system_rl_package.json

step "Train offline full-system RL" \
    "${OUT_DIR}/offline_full_system_rl_guarded/offline_full_system_rl_training_report.json" \
    "$PY" scripts/train_offline_full_system_rl.py \
    --package ${OUT_DIR}/full_system_rl_package.json \
    --base_router_model ${OUT_DIR}/router_hotpot_phase4.joblib \
    --probe_corpus ${DATA_DIR}/corpus.json \
    --out_dir ${OUT_DIR}/offline_full_system_rl_guarded \
    --force

step "Eval RL candidate tren test set" \
    "${OUT_DIR}/offline_full_system_rl_guarded/final_budget_policy_test_eval_offline_rl_candidate.json" \
    "$PY" scripts/eval_final_budget_bundle.py \
    --corpus ${DATA_DIR}/corpus.json \
    --qa ${DATA_DIR}/splits/test.json \
    --dataset_split test \
    --dataset_name "$DATASET_NAME" \
    --policy_bundle ${OUT_DIR}/offline_full_system_rl_guarded/final_budget_policy_bundle_offline_rl_candidate.json \
    --router_model ${OUT_DIR}/router_hotpot_phase4.joblib \
    --llm_provider "$LLM_PROVIDER" \
    --llm_model "$LLM_MODEL" \
    --budget_modes low medium high \
    --limit "$EVAL_LIMIT" \
    --seed "$EVAL_SEED" \
    --out ${OUT_DIR}/offline_full_system_rl_guarded/final_budget_policy_test_eval_offline_rl_candidate.json

step "Kiem tra promotion gate" \
    "${OUT_DIR}/offline_full_system_rl_guarded/promotion_check.json" \
    "$PY" scripts/check_offline_rl_promotion.py \
    --candidate_eval ${OUT_DIR}/offline_full_system_rl_guarded/final_budget_policy_test_eval_offline_rl_candidate.json \
    --frozen_eval ${OUT_DIR}/final_budget_policy_test_eval.json \
    --out ${OUT_DIR}/offline_full_system_rl_guarded/promotion_check.json

step "Promote RL candidate thanh frozen bundle" \
    "${OUT_DIR}/final_budget_policy_bundle_rl_ready.json" \
    "$PY" scripts/promote_offline_rl_candidate.py \
    --candidate_bundle ${OUT_DIR}/offline_full_system_rl_guarded/final_budget_policy_bundle_offline_rl_candidate.json \
    --candidate_eval ${OUT_DIR}/offline_full_system_rl_guarded/final_budget_policy_test_eval_offline_rl_candidate.json \
    --promotion_check ${OUT_DIR}/offline_full_system_rl_guarded/promotion_check.json \
    --frozen_bundle ${OUT_DIR}/final_budget_policy_bundle_rl_ready.json \
    --frozen_eval ${OUT_DIR}/final_budget_policy_test_eval_rl_ready.json \
    --frozen_report ${OUT_DIR}/final_project_report_rl_ready.json \
    --rl_package ${OUT_DIR}/full_system_rl_package.json \
    --out ${OUT_DIR}/promotion_report.json \
    --skip_gate_check

step "Eval promoted bundle (final RL-ready)" \
    "${OUT_DIR}/final_budget_policy_test_eval_rl_ready.json" \
    "$PY" scripts/eval_final_budget_bundle.py \
    --corpus ${DATA_DIR}/corpus.json \
    --qa ${DATA_DIR}/splits/test.json \
    --dataset_split test \
    --dataset_name "$DATASET_NAME" \
    --policy_bundle ${OUT_DIR}/final_budget_policy_bundle_rl_ready.json \
    --router_model ${OUT_DIR}/router_hotpot_phase4.joblib \
    --llm_provider "$LLM_PROVIDER" \
    --llm_model "$LLM_MODEL" \
    --hybrid_min_confidence 0.55 \
    --hybrid_low_cost_confidence 0.55 \
    --hybrid_low_cost_workflows W1 \
    --budget_modes low medium high \
    --limit "$EVAL_LIMIT" \
    --seed "$EVAL_SEED" \
    --out ${OUT_DIR}/final_budget_policy_test_eval_rl_ready.json

# =============================================================================
# PHASE 6 - IMPROVE HIGH-BUDGET BANDIT (5-FOLD CV)
# =============================================================================

step "Train CV ensemble bandit 5-fold high budget" \
    "${OUT_DIR}/cv_ensemble_report.json" \
    "$PY" scripts/train_cv_ensemble_bandit.py \
    --input ${OUT_DIR}/hotpotqa_large_train_rollouts_high_w2w3_det.json \
    --output ${OUT_DIR}/cv_ensemble_high_bandit.joblib \
    --base_router_model ${OUT_DIR}/router_hotpot_phase4.joblib \
    --probe_corpus ${DATA_DIR}/corpus.json \
    --budget_mode high \
    --allowed_workflows W2 W3 \
    --k_folds "$CV_FOLDS" \
    --seed "$EVAL_SEED" \
    --out_report ${OUT_DIR}/cv_ensemble_report.json

# =============================================================================
# PHASE 7 - SELECTIVE CRITIC VERIFICATION
# eval_phase4_router.py dung --budget_mode (singular)
# =============================================================================

step "Eval high budget voi critic AG (val set)" \
    "${OUT_DIR}/router_eval_large_budget_high_w2w3_det_critic2_val.json" \
    "$PY" scripts/eval_phase4_router.py \
    --corpus ${DATA_DIR}/corpus.json \
    --qa ${DATA_DIR}/splits/val.json \
    --dataset_split val \
    --dataset_name "$DATASET_NAME" \
    --router_model ${OUT_DIR}/router_hotpot_phase4.joblib \
    --critic_model ${OUT_DIR}/process_critic_budget_ag.joblib \
    --llm_provider "$LLM_PROVIDER" \
    --llm_model "$LLM_MODEL" \
    --budget_mode high \
    --limit "$EVAL_LIMIT" \
    --seed "$EVAL_SEED" \
    --out ${OUT_DIR}/router_eval_large_budget_high_w2w3_det_critic2_val.json

step "Verify selective critic (gate sweep)" \
    "${OUT_DIR}/selective_critic_verification.json" \
    "$PY" scripts/verify_selective_critic.py \
    --critic_eval ${OUT_DIR}/router_eval_large_budget_high_w2w3_det_critic2_val.json \
    --budget_mode high \
    --out ${OUT_DIR}/selective_critic_verification.json \
    --out_csv ${OUT_DIR}/metrics/metrics_selective_critic_verification.csv

# =============================================================================
# PHASE 8 - RE-GATE & ONLINE RL
# =============================================================================

step "Re-gate online RL (4 deployment gates)" \
    "${OUT_DIR}/regate_report.json" \
    "$PY" scripts/regate_online_rl.py \
    --improved_bandit_report ${OUT_DIR}/cv_ensemble_report.json \
    --selective_critic_report ${OUT_DIR}/selective_critic_verification.json

# =============================================================================
# PHASE 9 - EXPORT METRICS CSV
# =============================================================================

step "Export metrics CSV" \
    "${OUT_DIR}/metrics/metrics_system_overview.csv" \
    "$PY" scripts/export_metrics_csv.py \
    --out_dir ${OUT_DIR}/metrics

# =============================================================================
ELAPSED_MIN=$(( (SECONDS - START_TOTAL) / 60 ))
log ""
log "======================================================"
log "  HOAN THANH trong ${ELAPSED_MIN} phut! Log: $LOG"
log "  KET QUA DE BAO CAO:"
log "  ${OUT_DIR}/metrics/metrics_macro_budget_summary.csv"
log "  ${OUT_DIR}/metrics/cv_ensemble_bandit_configs.csv"
log "  ${OUT_DIR}/metrics/metrics_micro_critic_summary.csv"
log "  ${OUT_DIR}/metrics/metrics_selective_critic_verification.csv"
log "  ${OUT_DIR}/metrics/metrics_system_overview.csv"
log "======================================================"
