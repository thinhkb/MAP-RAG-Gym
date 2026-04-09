# MAP-RAG-Gym

MVP project that combines two ideas:
- MAO-ARAG style routing over multiple workflows.
- RAG-Gym style process-level supervision, logging, and critic training.

The current repo is a practical free-stack baseline: local BM25 retrieval, lightweight workflow routing, experiment manifests, and offline process-supervision utilities.

## Goals
- Keep the full pipeline runnable with a free or local stack.
- Separate workflow selection, execution, and process evaluation so each part can be tested independently.
- Provide a base that can later absorb stronger retrieval, stronger LLMs, or learned critics without rewriting the project.

## Current Scope
- Phase 0 completed as baseline: data preparation, deterministic splits, config, and experiment manifest logging.
- Phase 1 completed as baseline: workflow library and module executors for `W1` to `W6`.
- Phase 2 completed as baseline: rule-based router.
- Phase 3 completed as baseline: heuristic process scoring, process dataset generation, and offline process critic training.
- Phase 4 completed as baseline: sklearn learned router and offline evaluation.
- Not completed yet: critic-guided inference, hybrid retrieval, calibrated routing, large-scale evaluation, and any online/RL training loop.

## Workflow Library

| Workflow | Steps | Intended use |
| --- | --- | --- |
| `W1` | `AG` | Direct answer for cheap/simple questions |
| `W2` | `QR -> RA -> AG` | Rewrite first, then retrieve and answer |
| `W3` | `RA -> DS -> AG` | Retrieve, select evidence, then answer |
| `W4` | `QDP -> RA -> AS` | Parallel decomposition for comparative multi-hop |
| `W5` | `QDS -> QR -> RA -> AS` | Serial decomposition for dependency-heavy questions |
| `W6` | `DRAFT -> REFLECT -> RA -> AG` | Reflective retrieval path |

## Installation
```bash
python -m venv .venv
```

Windows PowerShell:
```bash
.venv\Scripts\Activate.ps1
pip install -e .
```

macOS/Linux:
```bash
source .venv/bin/activate
pip install -e .
```

## Quick Demo
```bash
python scripts/run_phase1_demo.py
python scripts/run_phase2_router_demo.py
python scripts/run_phase3_evaluate.py
python scripts/train_phase4_router.py
```

These demo scripts regenerate their own small outputs when needed. The `outputs/` folder is now trimmed to keep only representative experiment artefacts and a small quick-demo snapshot.

## Main Experiment Commands

### Prepare HotpotQA subset and splits
```bash
python scripts/prepare_hotpotqa.py --split validation --limit 200 --out_dir data/hotpotqa --write_splits
python scripts/create_qa_splits.py --qa data/hotpotqa/qa.json --out_dir data/hotpotqa/splits --seed 13
```

### Build router training rollouts
```bash
python scripts/batch_rollout.py --corpus data/hotpotqa/corpus.json --qa data/hotpotqa/splits/train.json --dataset_split train --llm_provider ollama --llm_model llama3.2 --limit 140 --n_candidates 3 --seed 13 --out outputs/hotpotqa_train_rollouts.json
```

### Train and evaluate the learned router
```bash
python scripts/train_phase4_router.py --input outputs/hotpotqa_train_rollouts.json --output outputs/router_hotpot_phase1.joblib --seed 13
python scripts/eval_phase4_router.py --corpus data/hotpotqa/corpus.json --qa data/hotpotqa/splits/val.json --dataset_split val --router_model outputs/router_hotpot_phase1.joblib --llm_provider ollama --llm_model llama3.2 --limit 30 --seed 13 --out outputs/router_eval_phase1.json
```

### Build process-supervision data and train the critic
```bash
python scripts/build_process_dataset.py --input outputs/hotpotqa_train_rollouts.json --out outputs/hotpotqa_process_train.json
python scripts/train_process_critic.py --input outputs/hotpotqa_process_train.json --output outputs/process_critic_hotpot_phase2.joblib --seed 13 --holdout_ratio 0.15
```

All major experiment files include a `manifest` with split, seed, model, prompt version, and git commit so runs are easier to compare and reproduce.

## Retained Outputs
- `outputs/hotpotqa_train_rollouts.json`: main rollout set used to train the router and derive process data.
- `outputs/router_hotpot_phase1.joblib` and `outputs/router_hotpot_phase1.joblib.meta.json`: learned router baseline and training metadata.
- `outputs/router_eval_phase1.json`: latest retained router evaluation on HotpotQA validation.
- `outputs/hotpotqa_process_train.json`: process-supervision dataset derived from the retained rollout set.
- `outputs/process_critic_hotpot_phase2.joblib` and `outputs/process_critic_hotpot_phase2.joblib.meta.json`: baseline process critic and evaluation metadata.
- `outputs/phase1_demo.json` and `outputs/phase2_router.json`: small quick-demo snapshots.

Sample, smoke, tmp, and stale result files were removed because they were either redundant, generated from tiny toy runs, or no longer matched the current retained experiment path.

## Latest Results

### Router evaluation snapshot
Source artefacts:
- `outputs/router_eval_phase1.json`
- `outputs/router_hotpot_phase1.joblib.meta.json`

Setup:
- Train split: 140 HotpotQA questions, 132 usable router labels after filtering.
- Validation split: 30 HotpotQA questions.
- LLM provider: Ollama `llama3.2`.
- Candidate count during rollout generation: `n_candidates=3`.

Results:

| Method | Avg utility | Avg tokens | Avg latency ms | Notes |
| --- | ---: | ---: | ---: | --- |
| Fixed `W3` | 0.4911 | 119.6 | 931.9 | Best overall retained utility |
| Fixed `W2` | 0.4858 | 233.0 | 2355.5 | Strong utility, higher cost than `W3` |
| Rule-based router | 0.3558 | 221.6 | 2347.8 | Best adaptive baseline so far |
| Fixed `W6` | 0.3527 | 292.5 | 3729.2 | Reflective path helps some cases but is expensive |
| Learned router | 0.2758 | 120.3 | 1701.3 | Efficient, but clearly behind the rule router |
| Fixed `W1` | 0.1703 | 40.2 | 989.5 | Cheapest but weakest quality |

Interpretation:
- Multi-step retrieval workflows are already useful; `W2` and `W3` beat direct answering by a wide margin.
- The current rule router is the strongest adaptive option in the retained experiments.
- The learned router is cheaper than the rule router, but its quality drops too much to claim a win yet.
- The learned router is biased toward `W1` on validation, which suggests label imbalance and under-retrieval on harder questions.

### Process critic snapshot
Source artefact:
- `outputs/process_critic_hotpot_phase2.joblib.meta.json`

Setup:
- Training source: `outputs/hotpotqa_train_rollouts.json`
- Process examples: 6,278 total
- Holdout evaluation examples: 948
- Target: `blended_reward = 0.7 * local_reward + 0.3 * outcome_utility`

Results:
- Overall MAE: `0.1781`
- Overall RMSE: `0.2529`
- Pearson: `0.3989`
- Spearman: `0.5092`

Interpretation:
- The critic does learn a meaningful signal from process traces.
- Rank correlation is decent for a lightweight baseline, so the critic is promising as a reranker or filter.
- Module quality is uneven: some modules like `QDS` are ranked reasonably well, while `RA` and `DS` remain weak and likely need better features or more balanced data.

## Assessment

### What is already working
- The repo now has an end-to-end baseline from dataset prep to rollout logging, router training, and critic training.
- Experiment manifests make retained results auditable instead of relying on unnamed JSON dumps.
- `W2` and `W3` show that the workflow library is not just decorative; the better retrieval-heavy paths materially improve utility.
- The rule router is strong enough to act as a useful baseline for later learning-based work.
- Process-supervision data generation is in place, and the critic baseline shows non-trivial ranking signal.

### Current limitations
- The latest learned router does not beat the rule router on the retained validation run.
- Router labels are imbalanced toward cheap workflows, which likely pushes the model to over-predict `W1`.
- Evaluation is still small-scale: 30 validation questions is enough for direction, not for strong claims.
- The critic is trained and evaluated offline only; it is not yet plugged back into inference-time action selection.
- Retrieval remains local BM25 over the prepared corpus, so evidence coverage is still a bottleneck.
- There is no calibrated confidence, abstention policy, or budget-aware fallback yet.

### Feasible solutions
- Rebalance router labels or use cost-aware weighting so the learned router does not collapse toward `W1`.
- Add a confidence threshold: if the learned router is uncertain, fall back to the rule router or to `W3`.
- Use the process critic to rerank query rewrites, retrieved evidence, and answer candidates before final selection.
- Expand the experiment matrix to a larger train/val/test split and keep one untouched final test set.
- Upgrade retrieval from plain BM25 to a hybrid BM25 plus embedding retriever once the baseline protocol is stable.
- Add module-specific critic features for retrieval and document selection, because those stages are currently the weakest.

## Next Phases
- Phase 5: critic-guided inference. Use the trained process critic to rerank candidates inside `QR`, `RA`, `DS`, and `AG`, then measure whether utility improves without exploding cost.
- Phase 6: stronger router and evaluation. Rebuild router labels on a larger dataset, add confidence calibration and fallback logic, and compare learned vs rule routing on a fixed validation/test protocol.
- Phase 7: retrieval upgrade. Replace pure local BM25 with a stronger corpus and hybrid retrieval so the best workflows are limited less by missing evidence.
- Phase 8: closed-loop improvement. Use retained traces, critic scores, and harder benchmarks to move from offline analysis toward iterative policy improvement.

## Model Provider Notes
- `dummy` provider is still useful for smoke tests and pipeline debugging.
- Ollama is the main local path for the retained experiments. The kept artefacts were generated with `llama3.2`.
- Gemini support remains available for future runs through `scripts/run_gemini_eval.py`.

## Mapping to the Original Repos
RAG-Gym contributes the process-supervision framing: trajectory logging, step-level rewards, critic training, and best-of-N style action selection.

MAO-ARAG contributes the planner idea: choose different workflows based on the question instead of forcing one static chain.

This repo intentionally keeps the implementation lightweight. The current priority is reproducible analysis and better routing decisions, not jumping straight to heavy RL or GPU-dependent training.
