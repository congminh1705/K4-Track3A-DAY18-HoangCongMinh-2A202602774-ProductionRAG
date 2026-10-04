# Reflection — Hoàng Công Minh

MSSV: 2A202602774 · K4 Track 3A · Ngày kiểm tra: 04/10/2026.
Bản ghi kỹ thuật dựa trên mã nguồn và các lần kiểm tra thực tế trong workspace.

## 1. Mapping bài giảng vào mã nguồn

| Concept | Module / hàm | Quan sát |
|---|---|---|
| Semantic chunking | M1 `chunk_semantic()` | Encode câu, chuẩn hóa vector và so cosine giữa hai câu kề nhau; threshold 0.85. Model được cache để tránh load mỗi tài liệu. |
| Parent/child chunking | M1 `chunk_hierarchical()`; pipeline `run_query()` | 26 tài liệu có text tạo 101 child; parent 2048, child 256 ký tự. ID dựa trên nguồn và nội dung tránh trùng giữa tài liệu; rerank child rồi mở rộng parent để giữ ngữ cảnh. |
| BM25 + Dense fusion | M2 `segment_vietnamese()`, `reciprocal_rank_fusion()` | Chuẩn hóa chữ thường/NFC, bỏ dấu nối `_`; RRF cộng 1/(60+rank+1), giữ riêng kết quả có cùng text nhưng khác nguồn. Dense sử dụng bge-m3 và Qdrant query_points. |
| Cross-encoder | M3 `CrossEncoderReranker.rerank()` | Chấm cặp query/text rồi xếp giảm dần và chọn top-3. Cache model giữa các instance; batch 4, max length 512 để giảm bộ nhớ CPU. |
| RAGAS 4 metrics | M4 `evaluate_ragas()`, `failure_analysis()` | Chỉ tạo bottom-5 theo điểm khi có evaluation thật; trạng thái unavailable không được xem là metric thấp hay hallucination. Lưu từng câu và context để kiểm chứng. |
| Contextual embeddings | M5 `_enrich_single_call()`, `enrich_chunks()` | Combined 1 request/chunk tạo summary, questions, context, metadata; enrichment đưa summary/questions vào text index. Metadata nguồn/parent không bị LLM ghi đè. Offline dùng extractive fallback. |

## 2. Khó khăn và giải quyết

- Lỗi môi trường: `did not find executable at '/usr/bin\python.exe': The system cannot find the path specified.`
  Kiểm tra `.venv` nhận thấy môi trường tạo trên Linux. Giữ bản cũ ở
  `.venv-linux-backup`, tạo `.venv` mới bằng CPython 3.11.15 cho Windows.
- Lỗi cài package: `An attempt was made to access a socket in a way forbidden by its access permissions. (os error 10013)`.
  Dùng quyền mạng được duyệt để cài dependency và tải model công khai, cache trong workspace.
- Lỗi API: `401 AuthenticationError` / `invalid_api_key`.
  Key hiện có thuộc OpenRouter nhưng endpoint mặc định là OpenAI; sửa `.env` sang endpoint
  OpenRouter và model `openai/gpt-4o-mini`. Yêu cầu kết nối nhỏ đã trả về thành công.
  Không ghi key vào tài liệu hay Git.
- Production dừng khi embedding batch 16. Giảm batch còn 4, giới hạn model sequence 512
  và cache chung dense encoder cho baseline/production. Lần chạy lại tiến triển qua indexing.
- Khi load hai model lớn, Windows báo `OSError: The paging file is too small for this operation to complete`.
  Đã thử bfloat16 để giảm bộ nhớ; cấu hình cuối dùng float32 nhanh hơn khi nạp model tuần tự. Evaluation retrieve cả batch,
  giải phóng dense model rồi mới nạp reranker; query đơn cũng chuyển model tuần tự.
- Docker chưa có trên máy; Qdrant fallback in-memory giữ pipeline chạy trong phiên hiện tại.
- Hai PDF scan không có text layer; ghi rõ cần OCR. Không tạo nội dung thay thế cho chúng.
- Điểm cần tìm hiểu thêm: phân biệt lỗi generation/retrieval với lỗi evaluator; tác động của
  model embedding và ngôn ngữ prompt chấm RAGAS; kiểm tra yêu cầu phiên bản/historical query.

## 3. Action plan cho project

Project đề xuất: trợ lý tra cứu chính sách nhân sự nội bộ từ corpus lab.
Đây là kế hoạch kỹ thuật đề xuất; chưa phải mô tả một sản phẩm cá nhân đã triển khai.

Hiện trạng: baseline paragraph + dense-only; production hybrid + rerank + enrichment.
Rủi ro: chính sách cũ/mới trộn nhau, câu hỏi mơ hồ, multi-hop cần nhiều nguồn và PDF scan chưa OCR.

1. Chunk theo section khi tài liệu có cấu trúc; dùng parent/child cho tài liệu dài.
2. Giữ BM25 + dense + RRF; thêm bộ lọc effective_date/version, cho phép query lịch sử.
3. Rerank top-20 xuống top-3 bằng bge-reranker-v2-m3; đo latency CPU trước khi triển khai.
4. Chạy RAGAS cùng cấu hình trên baseline và production; mở rộng test set bằng câu phủ định,
   numeric, version, multi-hop, ambiguous; lưu context và câu trả lời để review thủ công.
5. Dùng combined enrichment và cache theo hash nội dung; chỉ gửi dữ liệu tới provider được chấp thuận.

Timeline: tuần 1 bổ sung version metadata và OCR với kiểm tra chất lượng; tuần 2 đánh giá
retrieval, sửa bottom-5 và đo latency; tuần 3 xây giao diện tra cứu, citations và kiểm thử truy cập.

## Final verification

All 20 Production questions have measured scores. Tests: 50/50; pip check and ruff passed.
Per the authorized scope, baseline and 11 measured OpenRouter questions were retained;
only the remaining 9 questions were sent to Gemini (gemini-3.1-flash-lite, gemini-embedding-001).
The report records each question's evaluator. Delta is descriptive, not a controlled
same-evaluator comparison; it cannot establish improvement caused only by the pipeline.

| Metric | Baseline | Production | Delta |
|---|---:|---:|---:|
| faithfulness | 0.8405 | 0.9200 | +0.0795 |
| answer_relevancy | 0.6737 | 0.8485 | +0.1748 |
| context_precision | 0.9250 | 0.9000 | -0.0250 |
| context_recall | 0.9250 | 0.9250 | +0.0000 |

Measured bottom-5: Senior leave/salary, tuition repayment, advance fee, MFA, password length.
The Senior answer omits salary. The advance fee is 5,000 instead of 50,000 VND:
15,000,000 * 0.02 * 5/30 = 50,000. These are observed limitations, not claimed fixes.
Tuition repayment matches ground truth despite faithfulness 0; judge behavior is a hypothesis
until its trace is reviewed. See failure_analysis.md for evidence, Error Trees and proposed fixes.

API debugging: OpenRouter HTTP 402 was caused by insufficient account credit, not the key limit.
Gemini required its own endpoint and embedding model. Multiple candidates were not enabled,
so the evaluator generates them sequentially. Rate-limit errors required spacing requests;
the Windows event-loop warning was addressed with explicit HTTP client lifecycle management.
Successful per-question checkpoints survive interruptions and are matched to input/configuration.

The four current aggregates exceed the rubric thresholds, but do not guarantee every answer
is correct or a grade. The personal-project plan above remains proposed work.
