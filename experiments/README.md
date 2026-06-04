# Experiments — Cấu hình multi-model / multi-dataset

Thư mục này chứa các script thực nghiệm cho các cặp **Dataset × LLM** khác nhau.  
Mỗi experiment có **hai phiên bản**: `.sh` (Bash) và `.ps1` (Windows PowerShell).

## Danh sách script

| Bash script | PowerShell script | Dataset | Model | Output dir |
|-------------|-------------------|---------|-------|------------|
| `run_2wiki_llama32.sh` | `run_2wiki_llama32.ps1` | 2WikiMultiHopQA | llama3.2 (Ollama) | `outputs/2wikimultihopqa_llama32/` |
| `run_2wiki_gemini_flash.sh` | `run_2wiki_gemini_flash.ps1` | 2WikiMultiHopQA | gemini-2.0-flash (API) | `outputs/2wikimultihopqa_gemini_flash/` |
| `run_2wiki_gemma3.sh` | `run_2wiki_gemma3.ps1` | 2WikiMultiHopQA | gemma3 (Ollama) | `outputs/2wikimultihopqa_gemma3/` |
| `run_hotpotqa_gemini_flash.sh` | `run_hotpotqa_gemini_flash.ps1` | HotpotQA Large | gemini-2.0-flash (API) | `outputs/hotpotqa_large_gemini_flash/` |
| `run_hotpotqa_gemma3.sh` | `run_hotpotqa_gemma3.ps1` | HotpotQA Large | gemma3 (Ollama) | `outputs/hotpotqa_large_gemma3/` |

> Script gốc `run_all.sh` / `run_all.ps1` (HotpotQA Large + llama3.2) vẫn là baseline chính.

---

## Cách chạy

### Bash (Linux / Mac / Git Bash trên Windows)

```bash
# 1. Kích hoạt venv
source .venv/bin/activate           # Linux/Mac
# hoặc trên Git Bash Windows:
source .venv/Scripts/activate

# 2. Cấp quyền thực thi (chỉ cần 1 lần)
chmod +x experiments/*.sh

# 3. Chạy từ thư mục gốc của project
./experiments/run_2wiki_llama32.sh
./experiments/run_2wiki_gemini_flash.sh
./experiments/run_2wiki_gemma3.sh
./experiments/run_hotpotqa_gemini_flash.sh
./experiments/run_hotpotqa_gemma3.sh
```

### PowerShell (Windows)

```powershell
.venv\Scripts\activate
.\experiments\run_2wiki_gemini_flash.ps1
```

---

## Gemini API Key — Bắt buộc khi dùng model Gemini

Các script `*_gemini_flash.*` yêu cầu **GEMINI_API_KEY**. Key đã được lưu trong file `.env` ở thư mục gốc:

```env
# File .env (ở thư mục gốc MAP-RAG-Gym/)
GEMINI_API_KEY=<your-api-key-here>
OLLAMA_BASE_URL=http://localhost:11434
```

**Script Bash tự động đọc `.env`** và kiểm tra key trước khi chạy. Nếu thiếu:

```
ERROR: GEMINI_API_KEY chua duoc set!
Them vao file .env: GEMINI_API_KEY=<your-key>
```

**Lấy key miễn phí tại:** https://aistudio.google.com/app/apikey

---

## Test nhanh (khuyến nghị)

Sửa 2 dòng đầu trong bất kỳ script nào để chạy thử:

```bash
TRAIN_LIMIT=20   # thay vì 420
EVAL_LIMIT=10    # thay vì 90
```

Ước tính: **~10-15 phút** với Gemini Flash | **~30-45 phút** với Ollama local

---

## Tùy chỉnh thêm model mới

Sao chép script, đổi 3 dòng cấu hình:

```bash
LLM_PROVIDER="ollama"                        # hoặc "gemini"
LLM_MODEL="qwen2.5"                          # tên model Ollama / Gemini
OUT_DIR="outputs/${DATASET_NAME}_qwen25"     # output dir riêng
```

---

## So sánh kết quả

File metric chính: `outputs/<experiment>/metrics/metrics_macro_budget_summary.csv`
