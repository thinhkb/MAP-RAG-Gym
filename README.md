# MAP-RAG-Gym

MVP do an co so theo huong lai giua:
- MAO-ARAG: planner/router chon workflow dong theo query.
- RAG-Gym: danh gia tung buoc retrieval/reasoning/query o muc process.

## Muc tieu
- 100% co the chay voi stack free.
- Mac dinh dung local BM25 retrieval + rule router + heuristic evaluator.
- Co san hook de gan Gemini free tier hoac model open-source local sau.

## Cau truc phase
- Phase 0: du lieu, config, corpus va logging schema
- Phase 1: workflow + executors
- Phase 2: rule-based router
- Phase 3: process evaluator + critic heuristics
- Phase 4: learned router (sklearn baseline)

## Cai dat
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Chay demo nhanh
```bash
python scripts/run_phase1_demo.py
python scripts/run_phase2_router_demo.py
python scripts/run_phase3_evaluate.py
python scripts/train_phase4_router.py
```

## LLM free goi y
1. Ollama local:
   - qwen2.5:3b-instruct
   - llama3.1:8b-instruct neu may du manh
2. Gemini API free tier:
   - dat bien moi truong `GEMINI_API_KEY`
3. Hoac de `provider=dummy` de chay pipeline logic khong can LLM that.

## Mapping voi hai repo goc
RAG-Gym cung cap y tuong process supervision, trajectory logging, reward/critic va best-of-N inference. Repo README mo ta training/inference theo trajectory sampling, SFT, DPO va PRM. Agent inference sinh nhieu action va co the score de chon action tot nhat. 
MAO-ARAG cung cap y tuong planner chon workflow dong va toi uu chat luong/chi phi/latency; repo README mo ta planner + executor va training PPO, nhung setup goc dung GPU rat nang, nen MVP nay dung workflow library co dinh + rule/learned router de phu hop do an co so. 

## Ghi chu quan trong
- Ban nay uu tien mo phong va phan tich loi truoc, khong RL end-to-end ngay.
- Neu ban muon, buoc tiep theo hop ly la:
  - thay dummy/generic executor bang Gemini hoac Ollama,
  - bo sung retrieval tu Wikipedia dump,
  - train critic hoac router bang log da thu duoc.


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

Neu may yeu hoac prompt dai, hay thu model nho hon nhu `qwen2.5:3b-instruct` hoac tang `OLLAMA_TIMEOUT` len `600`.
