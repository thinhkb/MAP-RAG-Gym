# MAP-RAG-Gym

MVP project combining concepts from:
- MAO-ARAG: planner/router that selects dynamic workflows based on queries.
- RAG-Gym: evaluates each step of retrieval/reasoning/query at the process level.

## Objectives
- 100% runnable with free stack.
- Default uses local BM25 retrieval + rule router + heuristic evaluator.
- Ready hooks to integrate Gemini free tier or local open-source models later.

## Phase Structure
- Phase 0: data, config, corpus, and logging schema
- Phase 1: workflow + executors
- Phase 2: rule-based router
- Phase 3: process evaluator + critic heuristics
- Phase 4: learned router (sklearn baseline)

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Demo Run
```bash
python scripts/run_phase1_demo.py
python scripts/run_phase2_router_demo.py
python scripts/run_phase3_evaluate.py
python scripts/train_phase4_router.py
```

## LLM Free Suggestions
1. Ollama local:
   - qwen2.5:3b-instruct
   - llama3.1:8b-instruct if machine is strong enough
2. Gemini API free tier:
   - set environment variable `GEMINI_API_KEY`
3. Or set `provider=dummy` to run pipeline logic without LLM.

## Mapping to Original Repos
RAG-Gym provides process supervision ideas, trajectory logging, reward/critic, and best-of-N inference. The repo README describes training/inference via trajectory sampling, SFT, DPO, and PRM. Agent inference generates multiple actions and can score to select the best action.
MAO-ARAG provides planner ideas that select dynamic workflows and optimize quality/cost/latency; repo README describes planner + executor and PPO training, but the original setup uses heavy GPU, so this MVP uses fixed workflow library + rule/learned router suitable for basic projects.

## Important Notes
- This version prioritizes simulation and error analysis first, not end-to-end RL immediately.
- If you want, the next logical step is:
  - replace dummy/generic executor with Gemini or Ollama,
  - add retrieval from Wikipedia dump,
  - train critic or router using collected logs.

## Training Results and Conclusions

### Evaluation Results
Based on evaluation of 30 queries from HotpotQA validation set using Ollama (llama3.2 model) with learned router trained on 100 rollout samples:

- **Fixed Workflows**:
  - W1 (Direct Answer): avg_utility = 0.0856, avg_tokens = 122.7, avg_latency = 22824 ms
  - W2 (Single Retrieval + Answer): avg_utility = 0.254, avg_tokens = 766.4, avg_latency = 6465 ms
  - W3 (Multi-step Reasoning): avg_utility = 0.2503, avg_tokens = 366.4, avg_latency = 2601 ms
  - W6 (Reflective Retrieval): avg_utility = 0.1459, avg_tokens = 746.6, avg_latency = 5304 ms

- **Rule-based Router**: avg_utility = 0.1523, avg_tokens = 610.3, avg_retrieval_calls = 1.0667, avg_latency = 4644 ms
  - Workflow distribution: W6 (50%), W3 (30%), W2 (17%), W4 (3%)

- **Learned Router**: avg_utility = 0.2303, avg_tokens = 378.0, avg_retrieval_calls = 0.5333, avg_latency = 3422 ms
  - Workflow distribution: W1 (47%), W3 (23%), W6 (17%), W2 (13%)

### Conclusions
- The learned router achieves higher utility (0.2303) compared to rule-based (0.1523), approaching the performance of the best fixed workflow W2 (0.254).
- Learned router is more resource-efficient: uses fewer tokens (378 vs 610 for rule-based, 766 for W2) and lower latency (3422 ms vs 4644 ms for rule-based).
- The router learns to prefer simpler workflows (W1 dominant) while maintaining good performance, suggesting effective adaptation to query complexity.
- Results demonstrate the potential of learning-based routing for balancing accuracy and efficiency in RAG systems, with room for improvement through larger training datasets and more sophisticated models.

## New project utilities

### Prepare a paper-style free dataset (HotpotQA)
```bash
pip install -e .
python scripts/prepare_hotpotqa.py --split validation --limit 200 --out_dir data/hotpotqa
```

### Batch rollout to create router training data
```bash
python scripts/batch_rollout.py --corpus data/hotpotqa/corpus.json --qa data/hotpotqa/qa.json --llm_provider dummy --limit 50 --out outputs/hotpotqa_rollouts.json
python scripts/train_phase4_router.py --input outputs/hotpotqa_rollouts.json --output outputs/router_hotpot.joblib
```

### Run with Gemini
```bash
export GEMINI_API_KEY=YOUR_KEY
python scripts/run_gemini_eval.py --corpus data/hotpotqa/corpus.json --qa data/hotpotqa/qa.json --limit 20 --model gemini-2.0-flash
```

### Optional free local model with Ollama
```bash
ollama serve
ollama pull llama3.1:8b-instruct
export OLLAMA_TIMEOUT=300
export OLLAMA_MAX_RETRIES=3
python scripts/batch_rollout.py --corpus data/hotpotqa/corpus.json --qa data/hotpotqa/qa.json --llm_provider ollama --llm_model llama3.1:8b-instruct --limit 20 --resume
```

If machine is weak or prompts are long, try smaller model like `qwen2.5:3b-instruct` or increase `OLLAMA_TIMEOUT` to `600`.
