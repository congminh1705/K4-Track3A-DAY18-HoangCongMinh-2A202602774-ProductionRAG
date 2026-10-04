# Lab 18: Production RAG Pipeline

**K4-Track3A · Ngày 18 · Production RAG**  
**Thời gian:** 2h implement + 30 phút reflection

---

## Tổng quan

Bài tập **cá nhân** — implement toàn bộ 5 modules:

```
M1 Chunking → M5 Enrichment → M2 Hybrid Search → M3 Reranking → LLM Answer → M4 RAGAS Eval
```

Xem **ASSIGNMENT.md** để biết chi tiết từng module và timeline.

## Prerequisites

| Dependency | Bắt buộc? | Dùng cho |
|-----------|-----------|----------|
| Docker (Qdrant) | ✅ Có | M2 Dense Search |
| Python 3.11+ | ✅ Có | Tất cả modules (RAGAS cần 3.11+ cho asyncio) |
| `OPENAI_API_KEY` | ⚠️ M4+M5 | RAGAS eval (M4), Enrichment LLM (M5) |

**Pre-download models** (tránh timeout trong lab):
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"
```

## Quick Start

### 1. Clone repository & tạo môi trường ảo

**Linux / macOS / Git Bash:**
```bash
git clone <repo-url>
cd K4-Track3A-Production-RAG
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
git clone <repo-url>
cd K4-Track3A-Production-RAG
python -m venv .venv
.venv\Scripts\Activate.ps1
```
*(Nếu dùng Windows CMD: chạy `.venv\Scripts\activate.bat`)*

### 2. Cài đặt dependencies & Khởi động dịch vụ

**Linux / macOS / Git Bash:**
```bash
docker compose up -d                    # Khởi động Qdrant vector database
pip install -r requirements.txt
cp .env.example .env                    # Tạo file .env và điền OPENAI_API_KEY
python naive_baseline.py                # Khởi tạo baseline
```

**Windows (PowerShell):**
```powershell
docker compose up -d                    # Khởi động Qdrant vector database
pip install -r requirements.txt
Copy-Item .env.example .env             # Tạo file .env và điền OPENAI_API_KEY
python naive_baseline.py                # Khởi tạo baseline
```
*(Nếu dùng Windows CMD: dùng `copy .env.example .env` thay cho `Copy-Item`)*

## Chạy toàn bộ & Kiểm tra

```bash
python main.py                          # Chạy Naive + Production + In bảng so sánh
python check_lab.py                     # Script kiểm tra hợp lệ trước khi nộp (chạy được trên mọi OS)
```

## Cấu trúc repo

```
K4-Track3A-Production-RAG/
├── README.md                   # File này
├── ASSIGNMENT.md               # ★ Đề bài + timeline + reflection
├── RUBRIC.md                   # Hệ thống chấm điểm
│
├── main.py                     # Entry point: chạy toàn bộ pipeline
├── check_lab.py                # Kiểm tra định dạng trước khi nộp
├── naive_baseline.py           # Baseline (chạy trước)
├── config.py                   # Shared config
├── requirements.txt            # Dependencies
├── docker-compose.yml          # Qdrant local
├── .env.example                # API keys template
│
├── data/                       # Corpus tiếng Việt — 25 .md files + 3 PDFs (28 files total)
│   ├── nghi_phep_nam_v2023.md  # Nghỉ phép 12 ngày (v2023, superseded)
│   ├── nghi_phep_nam_v2024.md  # Nghỉ phép 15 ngày (v2024, hiện hành)
│   ├── mat_khau_v1.md          # Password policy 90 ngày (OLD)
│   ├── mat_khau_v2.md          # Password policy 120 ngày + MFA (NEW)
│   ├── ... (28 files total)    # 8 categories: leave, salary, IT, workflow, training, admin, safety, compliance
│   ├── so_tay_an_toan.pdf      # An toàn PCCC + sơ cứu (PDF text)
│   ├── BCTC.pdf                # Báo cáo tài chính (scan, cần OCR)
│   └── Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.pdf # Nghị định BVDL (scan, cần OCR)
├── test_set.json               # 20 Q&A pairs (6 types: lookup, version, negation, multi-hop, numeric, ambiguous)
│
├── src/                        # ★ Scaffold code (có TODO markers)
│   ├── m1_chunking.py          # Module 1: Chunking
│   ├── m2_search.py            # Module 2: Hybrid Search
│   ├── m3_rerank.py            # Module 3: Reranking
│   ├── m4_eval.py              # Module 4: Evaluation
│   ├── m5_enrichment.py        # Module 5: Enrichment Pipeline
│   └── pipeline.py             # Ghép toàn bộ pipeline
│
├── tests/                      # Auto-grading
│   ├── test_m1.py
│   ├── test_m2.py
│   ├── test_m3.py
│   ├── test_m4.py
│   └── test_m5.py
│
├── analysis/                   # ★ Deliverable
│   ├── failure_analysis.md     # Phân tích failures (cá nhân)
│   └── reflections/            # Reflection cá nhân
│       └── reflection_TEMPLATE.md
│
├── reports/                    # ★ Auto-generated (bắt buộc: reports/ragas_report.json)
│   ├── ragas_report.json
│   └── naive_baseline_report.json
│
└── templates/                  # Templates gốc (backup)
    └── failure_analysis.md
```

## Timeline (Thời lượng ước tính)

| Thời lượng | Hoạt động |
|------------|-----------|
| 10 phút | Setup môi trường + chạy `naive_baseline.py` |
| 90 phút | Implement M1 → M2 → M3 → M4 → M5 |
| 20 phút | Chạy pipeline + RAGAS + failure analysis |
| 30 phút | Reflection: lecture mapping + project plan |

## Quy chuẩn đặt tên Repository & Nộp bài

- **Cấu trúc đặt tên repo:**  
  `K4-Track3A-DAY18-<HoVaTen>-<MSSV>-ProductionRAG`  
  *(Ví dụ: `K4-Track3A-DAY18-NguyenVanAn-AI20K001-ProductionRAG`)*
- **Hạn chót nộp bài:** **23h59 ngày diễn ra bài lab (GMT+7)** trên cổng VLearn LMS / Codelab.
- **Chi tiết yêu cầu:** Xem tại [ASSIGNMENT.md](ASSIGNMENT.md) và [RUBRIC.md](RUBRIC.md).


## Kết quả triển khai và chạy trên Windows

Bài làm: **Hoàng Công Minh — 2A202602774**. Tên repository theo đề:
`K4-Track3A-DAY18-HoangCongMinh-2A202602774-ProductionRAG`.
Thư mục local được giữ nguyên để tránh làm hỏng đường dẫn đang sử dụng.

- M1: cosine similarity, parent/child có ID theo nguồn, section Markdown giữ code block.
- M2: Vietnamese BM25 + bge-m3 + Qdrant `query_points()` + RRF.
- M3: CrossEncoder bge-reranker-v2-m3; có thêm Flashrank.
- M4: RAGAS 4 metrics, lưu từng câu, bottom-5 và trạng thái đo.
- M5: combined 1 call/chunk, có fallback extractive; summary và HyQA được đưa vào văn bản index.
- Pipeline: retrieve/rerank child rồi mở rộng parent; lưu latency từng bước.

Windows uses Python 3.11 in `.venv`; the obsolete Linux environment backup has been removed. Installed dependencies are pinned in `requirements-lock.txt`. `transformers<5` và `datasets<3` giữ tương thích với stack RAGAS 0.1.

```powershell
# Cài trên máy mới có Python 3.11:
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
Copy-Item .env.example .env
# Điền key và endpoint của đúng nhà cung cấp trong .env.
# OpenRouter: OPENAI_BASE_URL=https://openrouter.ai/api/v1
#            LLM_MODEL=openai/gpt-4o-mini

# Chạy khi đã đồng ý gửi dữ liệu lab đến nhà cung cấp API:
.venv\Scripts\python.exe main.py
.venv\Scripts\python.exe check_lab.py

# Chạy cục bộ, không gọi LLM/RAGAS qua API:
$env:LAB_OFFLINE="1"
$env:HF_HUB_OFFLINE="1" # Chỉ sau khi đã tải model về .model-cache
.venv\Scripts\python.exe main.py
.venv\Scripts\python.exe -m pytest tests/ -q
# Bỏ chế độ offline khi cần đánh giá thật:
Remove-Item Env:LAB_OFFLINE -ErrorAction SilentlyContinue
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
```

Nếu Docker có sẵn, dùng `docker compose up -d`; nếu không kết nối được Qdrant,
client dùng in-memory cho lần chạy hiện tại. Model mặc định cần khoảng vài GB
đĩa trống và chạy CPU có thể chậm. Hai PDF scan cần OCR trước khi truy vấn;
loader hiện chỉ đọc text layer theo yêu cầu scaffold.

RAGAS dùng LLM và embedding `text-embedding-3-small` qua endpoint đã cấu hình
(OpenRouter dùng tên `openai/text-embedding-3-small`). Câu hỏi tái tạo để đo
Answer Relevancy dùng cùng ngôn ngữ với câu trả lời. Giữ cùng cấu hình evaluator
khi so sánh baseline và production; cấu hình được ghi trong JSON report.
Khi offline hoặc API lỗi, báo cáo có `evaluation.status=unavailable`, điểm 0 là
placeholder chưa đo, không được diễn giải thành điểm RAGAS thật.
`check_lab.py` sẽ báo chưa sẵn sàng nộp nếu chưa có evaluation đo thành công.

Model BGE mặc định dùng `MODEL_DTYPE=float32` sau khi đã xác minh trên CPU hiện tại.
Có thể đặt `MODEL_DTYPE=bfloat16` khi cần giảm RAM; batch 4 và sequence tối đa
512 token. Nội dung vượt giới hạn sẽ bị model truncate nên cần đánh giá recall
khi dùng corpus dài hơn.

API tham khảo: [Qdrant Python client](https://github.com/qdrant/qdrant-client) và
[RAGAS evaluate](https://docs.ragas.io/en/v0.1.21/references/evaluate/).

Trên máy hạn chế pagefile, evaluation lấy candidate cho toàn bộ test set trước,
giải phóng dense model rồi mới nạp reranker. Query đơn cũng giải phóng model
trước khi chuyển giai đoạn. Latency bao gồm chi phí nạp lại khi chạy query đơn.

`main.py` chỉ tạo `analysis/failure_analysis.md` nếu file chưa tồn tại; bài phân tích
đã viết được giữ nguyên. Có thể chạy riêng `python render_lab_report.py`.
Khi report thay đổi, cần đối chiếu và cập nhật phân tích theo kết quả thực tế.


Endpoint embedding OpenRouter đã được kiểm tra theo
[tài liệu chính thức](https://openrouter.ai/docs/api/api-reference/embeddings/create-embeddings).
Sau khi được chấp thuận gửi corpus lab, đã chạy evaluation online thay cho report chưa đo.
Historical successful-run metrics and exit status are retained in `reports/verification.json`. The recovered production report contains aggregate metrics only; original per-question scores are unavailable. Failed evaluations cannot overwrite a successful report, and existing analysis is preserved.

Có thể dùng `python main.py --resume` khi phiên chạy bị ngắt. Baseline được dùng
lại khi evaluator và bộ Q&A khớp; enrichment/câu trả lời lưu theo fingerprint.
Combined enrichment chạy tối đa 4 request song song, vẫn 1 request cho mỗi chunk.
Đặt `ENRICHMENT_WORKERS=1` nếu cần chạy tuần tự. Cache chỉ dùng lại trong chế độ
online và nằm trong reports, không chứa API key.

### Tiếp tục đánh giá khi API bị ngắt

Chạy `python main.py --eval-only` để đánh giá các câu trả lời đã lưu, không cần
tải lại mô hình hoặc dựng lại index. Mỗi câu đo thành công có checkpoint riêng
trong `reports/.evaluation-cache/` (Git bỏ qua). Checkpoint chỉ được dùng lại khi
câu hỏi, đáp án, context, ground truth và cấu hình evaluator khớp.
RAGAS chạy tuần tự, giới hạn output 1.024 tokens; lỗi một câu không được ghi đè
report thành công và không làm mất các câu đã đo. HTTP 402 cần kiểm tra số dư
tài khoản API, không phải chỉ hạn mức của key.
