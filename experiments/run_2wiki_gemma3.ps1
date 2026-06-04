# =============================================================================
# MAP-RAG-Gym - 2WikiMultiHopQA + Ollama gemma3 (Windows PowerShell)
# Su dung: .venv\Scripts\activate  roi  .\experiments\run_2wiki_gemma3.ps1
# Yeu cau: ollama pull gemma3
# =============================================================================

# ── CAU HINH ─────────────────────────────────────────────────────────────────
$LLM_PROVIDER = "ollama"
$LLM_MODEL    = "gemma3"
$TRAIN_LIMIT  = 420   # Doi thanh 20 de test nhanh
$EVAL_LIMIT   = 90    # Doi thanh 10 de test nhanh
$SEED         = 42
$EVAL_SEED    = 13
$CV_FOLDS     = 5
$DATASET_NAME = "2wikimultihopqa"
$DATA_DIR     = "data/$DATASET_NAME"
$OUT_DIR      = "outputs/${DATASET_NAME}_gemma3"
# --- HET CAU HINH ------------------------------------------------------------

$ErrorActionPreference = "Stop"
$StartTotal = Get-Date

$PY = ".venv\Scripts\python.exe"
if (-not (Test-Path $PY)) {
    Write-Host "ERROR: Khong tim thay $PY"
    exit 1
}
Write-Host "Using Python: $PY"

New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null
New-Item -ItemType Directory -Force -Path "$OUT_DIR\offline_full_system_rl_guarded" | Out-Null
New-Item -ItemType Directory -Force -Path "$OUT_DIR\metrics" | Out-Null

$LogFile = "$OUT_DIR\run_all.log"
$StepNum = 0

function Log($msg) {
    $ts = (Get-Date -Format "HH:mm:ss")
    $line = "[$ts] $msg"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

function Run-Step {
    param([string]$Name, [string]$OutFile, [string[]]$Args)
    $script:StepNum++
    $n = $script:StepNum
    if ($OutFile -and (Test-Path $OutFile)) {
        Log "[SKIP] Buoc $n - $Name (da co: $OutFile)"
        return
    }
    Log ""
    Log "======================================================"
    Log "  Buoc $n - $Name"
    Log "======================================================"
    $t0 = Get-Date
    & $script:PY $Args
    if ($LASTEXITCODE -ne 0) { Log "[LOI] Buoc $n that bai!"; exit 1 }
    Log "[OK] Buoc $n hoan thanh trong $([math]::Round(((Get-Date)-$t0).TotalSeconds,0))s"
}

Log "======================================================"
Log "   MAP-RAG-Gym | $LLM_PROVIDER/$LLM_MODEL"
Log "   Dataset: $DATASET_NAME | Train=$TRAIN_LIMIT Eval=$EVAL_LIMIT"
Log "======================================================"

# =============================================================================
# PHASE 1 - ROLLOUT
# =============================================================================

Run-Step "Rollout W1/W2/W3/W6 medium budget" `
    "$OUT_DIR\train_rollouts.json" `
    @("scripts\batch_rollout.py",
      "--corpus","$DATA_DIR/corpus.json",
      "--qa","$DATA_DIR/splits/train.json",
      "--dataset_name",$DATASET_NAME,"--dataset_split","train",
      "--llm_provider",$LLM_PROVIDER,"--llm_model",$LLM_MODEL,
      "--workflow_ids","W1","W2","W3","W6",
      "--budget_mode","medium","--limit",$TRAIN_LIMIT,"--seed",$SEED,
      "--out","$OUT_DIR/train_rollouts.json","--resume")

Run-Step "Rollout W2/W3 high budget det" `
    "$OUT_DIR\train_rollouts_high_w2w3_det.json" `
    @("scripts\batch_rollout.py",
      "--corpus","$DATA_DIR/corpus.json",
      "--qa","$DATA_DIR/splits/train.json",
      "--dataset_name",$DATASET_NAME,"--dataset_split","train",
      "--llm_provider",$LLM_PROVIDER,"--llm_model",$LLM_MODEL,
      "--workflow_ids","W2","W3",
      "--budget_mode","high","--limit",$TRAIN_LIMIT,"--seed",$SEED,
      "--out","$OUT_DIR/train_rollouts_high_w2w3_det.json","--resume")

Run-Step "Build process dataset" `
    "$OUT_DIR\process_train_budget.json" `
    @("scripts\build_process_dataset.py",
      "--input","$OUT_DIR/train_rollouts.json",
      "--out","$OUT_DIR/process_train_budget.json")

# =============================================================================
# PHASE 2 - TRAIN ROUTER
# =============================================================================

Run-Step "Train phase4 router (LearnedRouter)" `
    "$OUT_DIR\router_phase4.joblib" `
    @("scripts\train_phase4_router.py",
      "--input","$OUT_DIR/train_rollouts.json",
      "--output","$OUT_DIR/router_phase4.joblib",
      "--budget_modes","low","medium","high",
      "--allowed_workflows","W1","W2","W3","W6",
      "--seed",$EVAL_SEED)

Run-Step "Train meta router gate" `
    "$OUT_DIR\router_budget_calibrated.joblib" `
    @("scripts\train_meta_router.py",
      "--rollouts","$OUT_DIR/train_rollouts.json",
      "--router_model","$OUT_DIR/router_phase4.joblib",
      "--output","$OUT_DIR/router_budget_calibrated.joblib")

# =============================================================================
# PHASE 3 - TRAIN CRITIC
# =============================================================================

Run-Step "Train critic QR (local_reward)" `
    "$OUT_DIR\process_critic_qr_local.joblib" `
    @("scripts\train_process_critic.py",
      "--input","$OUT_DIR/process_train_budget.json",
      "--output","$OUT_DIR/process_critic_qr_local.joblib",
      "--target","local_reward",
      "--modules","QR")

Run-Step "Train critic AG (blended_reward)" `
    "$OUT_DIR\process_critic_ag.joblib" `
    @("scripts\train_process_critic.py",
      "--input","$OUT_DIR/process_train_budget.json",
      "--output","$OUT_DIR/process_critic_ag.joblib",
      "--target","blended_reward",
      "--modules","AG")

# =============================================================================
# PHASE 4 - EVALUATE & BUILD BUNDLE
# =============================================================================

Run-Step "Eval val set (low)" `
    "$OUT_DIR\router_eval_budget_low_val.json" `
    @("scripts\eval_phase4_router.py",
      "--corpus","$DATA_DIR/corpus.json",
      "--qa","$DATA_DIR/splits/val.json",
      "--dataset_split","val","--dataset_name",$DATASET_NAME,
      "--router_model","$OUT_DIR/router_phase4.joblib",
      "--llm_provider",$LLM_PROVIDER,"--llm_model",$LLM_MODEL,
      "--budget_mode","low","--limit",$EVAL_LIMIT,"--seed",$EVAL_SEED,
      "--out","$OUT_DIR/router_eval_budget_low_val.json")

Run-Step "Eval val set (medium)" `
    "$OUT_DIR\router_eval_budget_medium_val.json" `
    @("scripts\eval_phase4_router.py",
      "--corpus","$DATA_DIR/corpus.json",
      "--qa","$DATA_DIR/splits/val.json",
      "--dataset_split","val","--dataset_name",$DATASET_NAME,
      "--router_model","$OUT_DIR/router_phase4.joblib",
      "--llm_provider",$LLM_PROVIDER,"--llm_model",$LLM_MODEL,
      "--budget_mode","medium","--limit",$EVAL_LIMIT,"--seed",$EVAL_SEED,
      "--out","$OUT_DIR/router_eval_budget_medium_val.json")

Run-Step "Eval val set (high)" `
    "$OUT_DIR\router_eval_budget_high_val.json" `
    @("scripts\eval_phase4_router.py",
      "--corpus","$DATA_DIR/corpus.json",
      "--qa","$DATA_DIR/splits/val.json",
      "--dataset_split","val","--dataset_name",$DATASET_NAME,
      "--router_model","$OUT_DIR/router_phase4.joblib",
      "--llm_provider",$LLM_PROVIDER,"--llm_model",$LLM_MODEL,
      "--budget_mode","high","--limit",$EVAL_LIMIT,"--seed",$EVAL_SEED,
      "--out","$OUT_DIR/router_eval_budget_high_val.json")

Run-Step "Select best policy (low)" `
    "$OUT_DIR\policy_selection_low.json" `
    @("scripts\select_budget_policy.py",
      "--input","$OUT_DIR/router_eval_budget_low_val.json",
      "--out","$OUT_DIR/policy_selection_low.json")

Run-Step "Select best policy (medium)" `
    "$OUT_DIR\policy_selection_medium.json" `
    @("scripts\select_budget_policy.py",
      "--input","$OUT_DIR/router_eval_budget_medium_val.json",
      "--out","$OUT_DIR/policy_selection_medium.json")

Run-Step "Select best policy (high)" `
    "$OUT_DIR\policy_selection_high.json" `
    @("scripts\select_budget_policy.py",
      "--input","$OUT_DIR/router_eval_budget_high_val.json",
      "--out","$OUT_DIR/policy_selection_high.json")

Run-Step "Build final budget policy bundle" `
    "$OUT_DIR\final_budget_policy_bundle.json" `
    @("scripts\build_final_budget_policy_bundle.py",
      "--policies",
      "$OUT_DIR/policy_selection_low.json",
      "$OUT_DIR/policy_selection_medium.json",
      "$OUT_DIR/policy_selection_high.json",
      "--out","$OUT_DIR/final_budget_policy_bundle.json")

Run-Step "Eval frozen bundle tren test set" `
    "$OUT_DIR\final_budget_policy_test_eval.json" `
    @("scripts\eval_final_budget_bundle.py",
      "--corpus","$DATA_DIR/corpus.json",
      "--qa","$DATA_DIR/splits/test.json",
      "--dataset_split","test","--dataset_name",$DATASET_NAME,
      "--policy_bundle","$OUT_DIR/final_budget_policy_bundle.json",
      "--router_model","$OUT_DIR/router_phase4.joblib",
      "--llm_provider",$LLM_PROVIDER,"--llm_model",$LLM_MODEL,
      "--hybrid_min_confidence","0.55","--hybrid_low_cost_confidence","0.55",
      "--hybrid_low_cost_workflows","W1",
      "--budget_modes","low","medium","high",
      "--limit",$EVAL_LIMIT,"--seed",$EVAL_SEED,
      "--out","$OUT_DIR/final_budget_policy_test_eval.json")

Run-Step "Build final project report" `
    "$OUT_DIR\final_project_report.json" `
    @("scripts\build_final_project_report.py",
      "--policy_bundle","$OUT_DIR/final_budget_policy_bundle.json",
      "--test_eval","$OUT_DIR/final_budget_policy_test_eval.json",
      "--out","$OUT_DIR/final_project_report.json")

# =============================================================================
# PHASE 5 - OFFLINE FULL-SYSTEM RL
# =============================================================================

Run-Step "Build full-system RL package" `
    "$OUT_DIR\full_system_rl_package.json" `
    @("scripts\build_full_system_rl_package.py",
      "--policy_bundle","$OUT_DIR/final_budget_policy_bundle.json",
      "--final_eval","$OUT_DIR/final_budget_policy_test_eval.json",
      "--final_report","$OUT_DIR/final_project_report.json",
      "--macro_rollout",
      "low=$OUT_DIR/train_rollouts.json",
      "medium=$OUT_DIR/train_rollouts.json",
      "high=$OUT_DIR/train_rollouts_high_w2w3_det.json",
      "--critic_model",
      "QR=$OUT_DIR/process_critic_qr_local.joblib",
      "AG=$OUT_DIR/process_critic_ag.joblib",
      "--critic_meta",
      "QR=$OUT_DIR/process_critic_qr_local.joblib.meta.json",
      "AG=$OUT_DIR/process_critic_ag.joblib.meta.json",
      "--out","$OUT_DIR/full_system_rl_package.json")

Run-Step "Train offline full-system RL" `
    "$OUT_DIR\offline_full_system_rl_guarded\offline_full_system_rl_training_report.json" `
    @("scripts\train_offline_full_system_rl.py",
      "--package","$OUT_DIR/full_system_rl_package.json",
      "--base_router_model","$OUT_DIR/router_phase4.joblib",
      "--probe_corpus","$DATA_DIR/corpus.json",
      "--out_dir","$OUT_DIR/offline_full_system_rl_guarded",
      "--force")

Run-Step "Eval RL candidate tren test set" `
    "$OUT_DIR\offline_full_system_rl_guarded\final_budget_policy_test_eval_offline_rl_candidate.json" `
    @("scripts\eval_final_budget_bundle.py",
      "--corpus","$DATA_DIR/corpus.json",
      "--qa","$DATA_DIR/splits/test.json",
      "--dataset_split","test","--dataset_name",$DATASET_NAME,
      "--policy_bundle","$OUT_DIR/offline_full_system_rl_guarded/final_budget_policy_bundle_offline_rl_candidate.json",
      "--router_model","$OUT_DIR/router_phase4.joblib",
      "--llm_provider",$LLM_PROVIDER,"--llm_model",$LLM_MODEL,
      "--budget_modes","low","medium","high",
      "--limit",$EVAL_LIMIT,"--seed",$EVAL_SEED,
      "--out","$OUT_DIR/offline_full_system_rl_guarded/final_budget_policy_test_eval_offline_rl_candidate.json")

Run-Step "Kiem tra promotion gate" `
    "$OUT_DIR\offline_full_system_rl_guarded\promotion_check.json" `
    @("scripts\check_offline_rl_promotion.py",
      "--candidate_eval","$OUT_DIR/offline_full_system_rl_guarded/final_budget_policy_test_eval_offline_rl_candidate.json",
      "--frozen_eval","$OUT_DIR/final_budget_policy_test_eval.json",
      "--out","$OUT_DIR/offline_full_system_rl_guarded/promotion_check.json")

Run-Step "Promote RL candidate" `
    "$OUT_DIR\final_budget_policy_bundle_rl_ready.json" `
    @("scripts\promote_offline_rl_candidate.py",
      "--candidate_bundle","$OUT_DIR/offline_full_system_rl_guarded/final_budget_policy_bundle_offline_rl_candidate.json",
      "--candidate_eval","$OUT_DIR/offline_full_system_rl_guarded/final_budget_policy_test_eval_offline_rl_candidate.json",
      "--promotion_check","$OUT_DIR/offline_full_system_rl_guarded/promotion_check.json",
      "--frozen_bundle","$OUT_DIR/final_budget_policy_bundle_rl_ready.json",
      "--frozen_eval","$OUT_DIR/final_budget_policy_test_eval_rl_ready.json",
      "--frozen_report","$OUT_DIR/final_project_report_rl_ready.json",
      "--rl_package","$OUT_DIR/full_system_rl_package.json",
      "--out","$OUT_DIR/promotion_report.json",
      "--skip_gate_check")

Run-Step "Eval promoted bundle (RL-ready)" `
    "$OUT_DIR\final_budget_policy_test_eval_rl_ready.json" `
    @("scripts\eval_final_budget_bundle.py",
      "--corpus","$DATA_DIR/corpus.json",
      "--qa","$DATA_DIR/splits/test.json",
      "--dataset_split","test","--dataset_name",$DATASET_NAME,
      "--policy_bundle","$OUT_DIR/final_budget_policy_bundle_rl_ready.json",
      "--router_model","$OUT_DIR/router_phase4.joblib",
      "--llm_provider",$LLM_PROVIDER,"--llm_model",$LLM_MODEL,
      "--hybrid_min_confidence","0.55","--hybrid_low_cost_confidence","0.55",
      "--hybrid_low_cost_workflows","W1",
      "--budget_modes","low","medium","high",
      "--limit",$EVAL_LIMIT,"--seed",$EVAL_SEED,
      "--out","$OUT_DIR/final_budget_policy_test_eval_rl_ready.json")

# =============================================================================
# PHASE 6 - CV BANDIT
# =============================================================================

Run-Step "Train CV ensemble bandit 5-fold high budget" `
    "$OUT_DIR\cv_ensemble_report.json" `
    @("scripts\train_cv_ensemble_bandit.py",
      "--input","$OUT_DIR/train_rollouts_high_w2w3_det.json",
      "--output","$OUT_DIR/cv_ensemble_high_bandit.joblib",
      "--base_router_model","$OUT_DIR/router_phase4.joblib",
      "--probe_corpus","$DATA_DIR/corpus.json",
      "--budget_mode","high","--allowed_workflows","W2","W3",
      "--k_folds",$CV_FOLDS,"--seed",$EVAL_SEED,
      "--out_report","$OUT_DIR/cv_ensemble_report.json")

# =============================================================================
# PHASE 7 - SELECTIVE CRITIC
# =============================================================================

Run-Step "Eval high budget voi critic AG" `
    "$OUT_DIR\router_eval_budget_high_critic2_val.json" `
    @("scripts\eval_phase4_router.py",
      "--corpus","$DATA_DIR/corpus.json",
      "--qa","$DATA_DIR/splits/val.json",
      "--dataset_split","val","--dataset_name",$DATASET_NAME,
      "--router_model","$OUT_DIR/router_phase4.joblib",
      "--critic_model","$OUT_DIR/process_critic_ag.joblib",
      "--llm_provider",$LLM_PROVIDER,"--llm_model",$LLM_MODEL,
      "--budget_mode","high","--limit",$EVAL_LIMIT,"--seed",$EVAL_SEED,
      "--out","$OUT_DIR/router_eval_budget_high_critic2_val.json")

Run-Step "Verify selective critic" `
    "$OUT_DIR\selective_critic_verification.json" `
    @("scripts\verify_selective_critic.py",
      "--critic_eval","$OUT_DIR/router_eval_budget_high_critic2_val.json",
      "--budget_mode","high",
      "--out","$OUT_DIR/selective_critic_verification.json",
      "--out_csv","$OUT_DIR/metrics/metrics_selective_critic_verification.csv")

# =============================================================================
# PHASE 8 - RE-GATE
# =============================================================================

Run-Step "Re-gate online RL" `
    "$OUT_DIR\regate_report.json" `
    @("scripts\regate_online_rl.py",
      "--improved_bandit_report","$OUT_DIR/cv_ensemble_report.json",
      "--selective_critic_report","$OUT_DIR/selective_critic_verification.json")

# =============================================================================
# PHASE 9 - EXPORT METRICS
# =============================================================================

Run-Step "Export metrics CSV" `
    "$OUT_DIR\metrics\metrics_system_overview.csv" `
    @("scripts\export_metrics_csv.py",
      "--out_dir","$OUT_DIR/metrics")

# =============================================================================
$ElapsedMin = [math]::Round(((Get-Date)-$StartTotal).TotalMinutes,1)
Log ""
Log "======================================================"
Log "  HOAN THANH trong ${ElapsedMin} phut! Log: $LogFile"
Log "  Dataset: $DATASET_NAME | Model: $LLM_PROVIDER/$LLM_MODEL"
Log "  KET QUA: $OUT_DIR\metrics\"
Log "======================================================"
