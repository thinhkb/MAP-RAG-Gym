# =============================================================================
# MAP-RAG-Gym - Full Pipeline (Windows PowerShell)
# Su dung: .venv\Scripts\activate  roi  .\run_all.ps1
# =============================================================================

# ── CAU HINH ─────────────────────────────────────────────────────────────────
$LLM_PROVIDER = "ollama"
$LLM_MODEL    = "llama3.2"
$TRAIN_LIMIT  = 420   # Doi thanh 20 de test nhanh
$EVAL_LIMIT   = 90    # Doi thanh 10 de test nhanh
$SEED         = 42
$EVAL_SEED    = 13
$CV_FOLDS     = 5
# ── HET CAU HINH ─────────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"
$StartTotal = Get-Date

$PY = ".venv\Scripts\python.exe"
if (-not (Test-Path $PY)) {
    Write-Host "ERROR: Khong tim thay $PY"
    Write-Host "Chay: python -m venv .venv && .venv\Scripts\activate && pip install -e ."
    exit 1
}
Write-Host "Using Python: $PY"

$LogFile = "outputs\run_all.log"
$StepNum = 0

New-Item -ItemType Directory -Force -Path "outputs" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs\offline_full_system_rl_guarded" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs\metrics" | Out-Null

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
Log "   MAP-RAG-Gym | $LLM_PROVIDER/$LLM_MODEL | Train=$TRAIN_LIMIT Eval=$EVAL_LIMIT"
Log "======================================================"

# =============================================================================
# PHASE 1 - ROLLOUT
# =============================================================================

Run-Step "Rollout W1/W2/W3/W6 medium budget" `
    "outputs\hotpotqa_large_train_rollouts.json" `
    @("scripts\batch_rollout.py",
      "--corpus","data/hotpotqa_large/corpus.json",
      "--qa","data/hotpotqa_large/splits/train.json",
      "--dataset_name","hotpotqa_large","--dataset_split","train",
      "--llm_provider",$LLM_PROVIDER,"--llm_model",$LLM_MODEL,
      "--workflow_ids","W1","W2","W3","W6",
      "--budget_mode","medium","--limit",$TRAIN_LIMIT,"--seed",$SEED,
      "--out","outputs/hotpotqa_large_train_rollouts.json","--resume")

Run-Step "Rollout W2/W3 high budget det" `
    "outputs\hotpotqa_large_train_rollouts_high_w2w3_det.json" `
    @("scripts\batch_rollout.py",
      "--corpus","data/hotpotqa_large/corpus.json",
      "--qa","data/hotpotqa_large/splits/train.json",
      "--dataset_name","hotpotqa_large","--dataset_split","train",
      "--llm_provider",$LLM_PROVIDER,"--llm_model",$LLM_MODEL,
      "--workflow_ids","W2","W3",
      "--budget_mode","high","--limit",$TRAIN_LIMIT,"--seed",$SEED,
      "--out","outputs/hotpotqa_large_train_rollouts_high_w2w3_det.json","--resume")

Run-Step "Build process dataset" `
    "outputs\hotpotqa_large_process_train_budget.json" `
    @("scripts\build_process_dataset.py",
      "--input","outputs/hotpotqa_large_train_rollouts.json",
      "--out","outputs/hotpotqa_large_process_train_budget.json")

# =============================================================================
# PHASE 2 - TRAIN ROUTER
# Phase4 router TRUOC meta router (meta can phase4 lam router_model)
# =============================================================================

Run-Step "Train phase4 router (LearnedRouter)" `
    "outputs\router_hotpot_phase4.joblib" `
    @("scripts\train_phase4_router.py",
      "--input","outputs/hotpotqa_large_train_rollouts.json",
      "--output","outputs/router_hotpot_phase4.joblib",
      "--budget_modes","low","medium","high",
      "--allowed_workflows","W1","W2","W3","W6",
      "--seed",$EVAL_SEED)

Run-Step "Train meta router gate" `
    "outputs\router_hotpot_budget_calibrated.joblib" `
    @("scripts\train_meta_router.py",
      "--rollouts","outputs/hotpotqa_large_train_rollouts.json",
      "--router_model","outputs/router_hotpot_phase4.joblib",
      "--output","outputs/router_hotpot_budget_calibrated.joblib")

# =============================================================================
# PHASE 3 - TRAIN CRITIC
# Args dung: --input --output --target --modules (khong co --module hay --reward_type)
# =============================================================================

Run-Step "Train critic QR (local_reward)" `
    "outputs\process_critic_budget_qr_local.joblib" `
    @("scripts\train_process_critic.py",
      "--input","outputs/hotpotqa_large_process_train_budget.json",
      "--output","outputs/process_critic_budget_qr_local.joblib",
      "--target","local_reward",
      "--modules","QR")

Run-Step "Train critic AG (blended_reward)" `
    "outputs\process_critic_budget_ag.joblib" `
    @("scripts\train_process_critic.py",
      "--input","outputs/hotpotqa_large_process_train_budget.json",
      "--output","outputs/process_critic_budget_ag.joblib",
      "--target","blended_reward",
      "--modules","AG")

# =============================================================================
# PHASE 4 - EVALUATE & BUILD BUNDLE
# eval_phase4_router.py: --budget_mode (singular!), --router_model (required)
# =============================================================================

Run-Step "Eval val set (low)" `
    "outputs\router_eval_large_budget_low_val.json" `
    @("scripts\eval_phase4_router.py",
      "--corpus","data/hotpotqa_large/corpus.json",
      "--qa","data/hotpotqa_large/splits/val.json",
      "--dataset_split","val","--dataset_name","hotpotqa_large",
      "--router_model","outputs/router_hotpot_phase4.joblib",
      "--llm_provider",$LLM_PROVIDER,"--llm_model",$LLM_MODEL,
      "--budget_mode","low","--limit",$EVAL_LIMIT,"--seed",$EVAL_SEED,
      "--out","outputs/router_eval_large_budget_low_val.json")

Run-Step "Eval val set (medium)" `
    "outputs\router_eval_large_budget_medium_val.json" `
    @("scripts\eval_phase4_router.py",
      "--corpus","data/hotpotqa_large/corpus.json",
      "--qa","data/hotpotqa_large/splits/val.json",
      "--dataset_split","val","--dataset_name","hotpotqa_large",
      "--router_model","outputs/router_hotpot_phase4.joblib",
      "--llm_provider",$LLM_PROVIDER,"--llm_model",$LLM_MODEL,
      "--budget_mode","medium","--limit",$EVAL_LIMIT,"--seed",$EVAL_SEED,
      "--out","outputs/router_eval_large_budget_medium_val.json")

Run-Step "Eval val set (high)" `
    "outputs\router_eval_large_budget_high_val.json" `
    @("scripts\eval_phase4_router.py",
      "--corpus","data/hotpotqa_large/corpus.json",
      "--qa","data/hotpotqa_large/splits/val.json",
      "--dataset_split","val","--dataset_name","hotpotqa_large",
      "--router_model","outputs/router_hotpot_phase4.joblib",
      "--llm_provider",$LLM_PROVIDER,"--llm_model",$LLM_MODEL,
      "--budget_mode","high","--limit",$EVAL_LIMIT,"--seed",$EVAL_SEED,
      "--out","outputs/router_eval_large_budget_high_val.json")

Run-Step "Select best policy (low)" `
    "outputs\policy_selection_low.json" `
    @("scripts\select_budget_policy.py",
      "--input","outputs/router_eval_large_budget_low_val.json",
      "--out","outputs/policy_selection_low.json")

Run-Step "Select best policy (medium)" `
    "outputs\policy_selection_medium.json" `
    @("scripts\select_budget_policy.py",
      "--input","outputs/router_eval_large_budget_medium_val.json",
      "--out","outputs/policy_selection_medium.json")

Run-Step "Select best policy (high)" `
    "outputs\policy_selection_high.json" `
    @("scripts\select_budget_policy.py",
      "--input","outputs/router_eval_large_budget_high_val.json",
      "--out","outputs/policy_selection_high.json")

Run-Step "Build final budget policy bundle" `
    "outputs\final_budget_policy_bundle.json" `
    @("scripts\build_final_budget_policy_bundle.py",
      "--policies",
      "outputs/policy_selection_low.json",
      "outputs/policy_selection_medium.json",
      "outputs/policy_selection_high.json",
      "--out","outputs/final_budget_policy_bundle.json")

Run-Step "Eval frozen bundle tren test set" `
    "outputs\final_budget_policy_test_eval.json" `
    @("scripts\eval_final_budget_bundle.py",
      "--corpus","data/hotpotqa_large/corpus.json",
      "--qa","data/hotpotqa_large/splits/test.json",
      "--dataset_split","test","--dataset_name","hotpotqa_large",
      "--policy_bundle","outputs/final_budget_policy_bundle.json",
      "--router_model","outputs/router_hotpot_phase4.joblib",
      "--llm_provider",$LLM_PROVIDER,"--llm_model",$LLM_MODEL,
      "--hybrid_min_confidence","0.55","--hybrid_low_cost_confidence","0.55",
      "--hybrid_low_cost_workflows","W1",
      "--budget_modes","low","medium","high",
      "--limit",$EVAL_LIMIT,"--seed",$EVAL_SEED,
      "--out","outputs/final_budget_policy_test_eval.json")

Run-Step "Build final project report" `
    "outputs\final_project_report.json" `
    @("scripts\build_final_project_report.py",
      "--policy_bundle","outputs/final_budget_policy_bundle.json",
      "--test_eval","outputs/final_budget_policy_test_eval.json",
      "--out","outputs/final_project_report.json")

# =============================================================================
# PHASE 5 - OFFLINE FULL-SYSTEM RL
# =============================================================================

Run-Step "Build full-system RL package" `
    "outputs\full_system_rl_package.json" `
    @("scripts\build_full_system_rl_package.py",
      "--policy_bundle","outputs/final_budget_policy_bundle.json",
      "--final_eval","outputs/final_budget_policy_test_eval.json",
      "--final_report","outputs/final_project_report.json",
      "--macro_rollout",
      "low=outputs/hotpotqa_large_train_rollouts.json",
      "medium=outputs/hotpotqa_large_train_rollouts.json",
      "high=outputs/hotpotqa_large_train_rollouts_high_w2w3_det.json",
      "--critic_model",
      "QR=outputs/process_critic_budget_qr_local.joblib",
      "AG=outputs/process_critic_budget_ag.joblib",
      "--critic_meta",
      "QR=outputs/process_critic_budget_qr_local.joblib.meta.json",
      "AG=outputs/process_critic_budget_ag.joblib.meta.json",
      "--out","outputs/full_system_rl_package.json")

Run-Step "Train offline full-system RL" `
    "outputs\offline_full_system_rl_guarded\offline_full_system_rl_training_report.json" `
    @("scripts\train_offline_full_system_rl.py",
      "--package","outputs/full_system_rl_package.json",
      "--base_router_model","outputs/router_hotpot_phase4.joblib",
      "--probe_corpus","data/hotpotqa_large/corpus.json",
      "--out_dir","outputs/offline_full_system_rl_guarded",
      "--force")

Run-Step "Eval RL candidate tren test set" `
    "outputs\offline_full_system_rl_guarded\final_budget_policy_test_eval_offline_rl_candidate.json" `
    @("scripts\eval_final_budget_bundle.py",
      "--corpus","data/hotpotqa_large/corpus.json",
      "--qa","data/hotpotqa_large/splits/test.json",
      "--dataset_split","test","--dataset_name","hotpotqa_large",
      "--policy_bundle","outputs/offline_full_system_rl_guarded/final_budget_policy_bundle_offline_rl_candidate.json",
      "--router_model","outputs/router_hotpot_phase4.joblib",
      "--llm_provider",$LLM_PROVIDER,"--llm_model",$LLM_MODEL,
      "--budget_modes","low","medium","high",
      "--limit",$EVAL_LIMIT,"--seed",$EVAL_SEED,
      "--out","outputs/offline_full_system_rl_guarded/final_budget_policy_test_eval_offline_rl_candidate.json")

Run-Step "Kiem tra promotion gate" `
    "outputs\offline_full_system_rl_guarded\promotion_check.json" `
    @("scripts\check_offline_rl_promotion.py",
      "--candidate_eval","outputs/offline_full_system_rl_guarded/final_budget_policy_test_eval_offline_rl_candidate.json",
      "--frozen_eval","outputs/final_budget_policy_test_eval.json",
      "--out","outputs/offline_full_system_rl_guarded/promotion_check.json")

Run-Step "Promote RL candidate" `
    "outputs\final_budget_policy_bundle_rl_ready.json" `
    @("scripts\promote_offline_rl_candidate.py",
      "--candidate_bundle","outputs/offline_full_system_rl_guarded/final_budget_policy_bundle_offline_rl_candidate.json",
      "--candidate_eval","outputs/offline_full_system_rl_guarded/final_budget_policy_test_eval_offline_rl_candidate.json",
      "--promotion_check","outputs/offline_full_system_rl_guarded/promotion_check.json",
      "--skip_gate_check")

Run-Step "Eval promoted bundle (RL-ready)" `
    "outputs\final_budget_policy_test_eval_rl_ready.json" `
    @("scripts\eval_final_budget_bundle.py",
      "--corpus","data/hotpotqa_large/corpus.json",
      "--qa","data/hotpotqa_large/splits/test.json",
      "--dataset_split","test","--dataset_name","hotpotqa_large",
      "--policy_bundle","outputs/final_budget_policy_bundle_rl_ready.json",
      "--router_model","outputs/router_hotpot_phase4.joblib",
      "--llm_provider",$LLM_PROVIDER,"--llm_model",$LLM_MODEL,
      "--hybrid_min_confidence","0.55","--hybrid_low_cost_confidence","0.55",
      "--hybrid_low_cost_workflows","W1",
      "--budget_modes","low","medium","high",
      "--limit",$EVAL_LIMIT,"--seed",$EVAL_SEED,
      "--out","outputs/final_budget_policy_test_eval_rl_ready.json")

# =============================================================================
# PHASE 6 - CV BANDIT
# =============================================================================

Run-Step "Train CV ensemble bandit 5-fold high budget" `
    "outputs\cv_ensemble_report.json" `
    @("scripts\train_cv_ensemble_bandit.py",
      "--input","outputs/hotpotqa_large_train_rollouts_high_w2w3_det.json",
      "--output","outputs/cv_ensemble_high_bandit.joblib",
      "--base_router_model","outputs/router_hotpot_phase4.joblib",
      "--probe_corpus","data/hotpotqa_large/corpus.json",
      "--budget_mode","high","--allowed_workflows","W2","W3",
      "--k_folds",$CV_FOLDS,"--seed",$EVAL_SEED,
      "--out_report","outputs/cv_ensemble_report.json")

# =============================================================================
# PHASE 7 - SELECTIVE CRITIC
# =============================================================================

Run-Step "Eval high budget voi critic AG" `
    "outputs\router_eval_large_budget_high_w2w3_det_critic2_val.json" `
    @("scripts\eval_phase4_router.py",
      "--corpus","data/hotpotqa_large/corpus.json",
      "--qa","data/hotpotqa_large/splits/val.json",
      "--dataset_split","val","--dataset_name","hotpotqa_large",
      "--router_model","outputs/router_hotpot_phase4.joblib",
      "--critic_model","outputs/process_critic_budget_ag.joblib",
      "--llm_provider",$LLM_PROVIDER,"--llm_model",$LLM_MODEL,
      "--budget_mode","high","--limit",$EVAL_LIMIT,"--seed",$EVAL_SEED,
      "--out","outputs/router_eval_large_budget_high_w2w3_det_critic2_val.json")

Run-Step "Verify selective critic" `
    "outputs\selective_critic_verification.json" `
    @("scripts\verify_selective_critic.py",
      "--critic_eval","outputs/router_eval_large_budget_high_w2w3_det_critic2_val.json",
      "--budget_mode","high",
      "--out","outputs/selective_critic_verification.json",
      "--out_csv","outputs/metrics/metrics_selective_critic_verification.csv")

# =============================================================================
# PHASE 8 - RE-GATE
# =============================================================================

Run-Step "Re-gate online RL" `
    "outputs\regate_report.json" `
    @("scripts\regate_online_rl.py",
      "--improved_bandit_report","outputs/cv_ensemble_report.json",
      "--selective_critic_report","outputs/selective_critic_verification.json")

# =============================================================================
# PHASE 9 - EXPORT METRICS
# =============================================================================

Run-Step "Export metrics CSV" `
    "outputs\metrics\metrics_system_overview.csv" `
    @("scripts\export_metrics_csv.py",
      "--out_dir","outputs/metrics")

# =============================================================================
$ElapsedMin = [math]::Round(((Get-Date)-$StartTotal).TotalMinutes,1)
Log ""
Log "======================================================"
Log "  HOAN THANH trong ${ElapsedMin} phut! Log: $LogFile"
Log "  KET QUA: outputs\metrics\metrics_macro_budget_summary.csv"
Log "           outputs\metrics\cv_ensemble_bandit_configs.csv"
Log "           outputs\metrics\metrics_micro_critic_summary.csv"
Log "           outputs\metrics\metrics_selective_critic_verification.csv"
Log "           outputs\metrics\metrics_system_overview.csv"
Log "======================================================"
