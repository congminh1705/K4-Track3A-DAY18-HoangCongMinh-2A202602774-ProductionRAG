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

## Trạng thái xác minh

Trạng thái sau khi khôi phục báo cáo: code đạt 49/49 tests; bảng bên dưới là kết quả
lịch sử của lần chạy thành công. JSON gốc có điểm từng câu đã bị ghi đè. Đã đo lại
và lưu checkpoint cho 11/20 câu; 9 câu còn lại bị chặn bởi OpenRouter HTTP 402
(tài khoản chưa mua credit). Report hiện là `measured_summary_recovered`, chưa đủ
để xác nhận bài sẵn sàng nộp. Cần hoàn tất đánh giá và cập nhật bottom-5 trước khi nộp.

Xem `reports/verification.json` và `reports/ragas_report.json` cho kết quả cập nhật.
Nếu evaluation.status là unavailable, bảng điểm RAGAS và bottom-5 vẫn chưa đủ điều kiện nộp.


Kết quả cuối sau khi được đồng ý gửi dữ liệu tới OpenRouter: **47/47 tests pass**,
`pip check` không có dependency lỗi, ruff sạch. `python main.py --resume` và
`python check_lab.py` đều có Python exit code 0; baseline tương thích đã được dùng lại.

| Metric | Baseline | Production | Delta |
|---|---:|---:|---:|
| faithfulness | 0.7958 | 0.9200 | +0.1242 |
| answer_relevancy | 0.6571 | 0.7943 | +0.1372 |
| context_precision | 0.9250 | 0.9500 | +0.0250 |
| context_recall | 0.9250 | 0.9500 | +0.0250 |

Lần chạy online tiếp tục production mất 537.0 giây;
reranking trung bình 12.87 giây/câu. Cả bốn metric vượt 0.75, nhưng review thủ công
vẫn phát hiện thiếu lương trong câu multi-hop và lỗi số học tạm ứng: LLM trả 5.000,
trong khi phép tính pro-rata theo tháng 30 ngày là 50.000 VNĐ. Bài học là cần kiểm tra
kết quả numeric độc lập và đa dạng parent context; không dùng aggregate như cam kết
mọi đáp án đều đúng.

Evaluator ban đầu dùng MiniLM cục bộ và tái tạo câu hỏi tiếng Anh, khiến câu thử đúng
có relevancy khoảng 0.084. Đổi sang text-embedding-3-small qua OpenRouter và yêu cầu
câu hỏi tái tạo cùng ngôn ngữ đưa câu thử gần 1.0. Đây là sanity check evaluator;
điểm production trong bảng lấy từ câu trả lời pipeline trên toàn bộ 20 câu, không lấy
câu trả lời ground truth để generation.

Thêm cache enrichment theo hash và checkpoint câu trả lời sau khi một phiên bị ngắt.
Cache lưu atomic, không chứa API key; kiểm tra hồi quy xác nhận không gọi API lần hai
cho cùng chunk và không dùng lại câu trả lời khi fingerprint khác. Môi trường pytest
được cấu hình thư mục tạm trong workspace để tránh WinError 5 trên Temp của Windows.
