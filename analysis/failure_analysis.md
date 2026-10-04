# Failure Analysis — Lab 18

Hoàng Công Minh · 2A202602774 · K4 Track 3A

## RAGAS comparison

| Metric | Baseline | Production | Delta |
|---|---:|---:|---:|
| faithfulness | 0.8405 | 0.9200 | +0.0795 |
| answer_relevancy | 0.6737 | 0.8485 | +0.1748 |
| context_precision | 0.9250 | 0.9000 | -0.0250 |
| context_recall | 0.9250 | 0.9250 | +0.0000 |

Production giữ 11 câu đã đo bằng OpenRouter và chỉ gửi 9 câu còn lại tới Gemini.
Baseline giữ evaluator OpenRouter. Delta không phải so sánh có kiểm soát bằng cùng evaluator;
không kết luận chênh lệch chỉ do pipeline. JSON ghi cấu hình evaluator cho từng câu.

## Bottom-5 theo trung bình bốn metric RAGAS

### Ca 1

- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Worst metric:** context_precision
- **Context count:** 2

**Got (thực tế):**

> Nhân viên Senior có 9 năm thâm niên sẽ được nghỉ **18 ngày phép năm** (15 ngày cơ bản + 3 ngày cộng thêm cho 9 năm thâm niên).
>
> Về lương, thông tin không được cung cấp trong chính sách nghỉ phép năm, vì vậy không tìm thấy thông tin cụ thể về lương của nhân viên này.

- **Worst score:** 0.0000; **Mean four metrics:** 0.3750
- **Error Tree:** Output incorrect → inspect evidence → context_precision → Too many irrelevant chunks
- **Diagnostic hypothesis:** Too many irrelevant chunks; cần kiểm tra context và đáp án trước khi kết luận.
- **Suggested fix:** Rerank candidates and filter obsolete versions

- **Đối chiếu thủ công:** Đáp án đúng 18 ngày phép nhưng thiếu lương Senior 20–35 triệu. Context chỉ có chính sách phép, chưa đủ bảng lương. Context precision 0 phản ánh cách judge chấm câu hỏi hai phần; không kết luận toàn bộ context vô ích.
- **Error Tree sau đối chiếu:** Câu hỏi hai phần → kiểm tra đủ nguồn → thiếu bảng lương → retrieval/parent expansion → đáp án thiếu phần lương.
- **Hướng sửa cụ thể:** Tách truy vấn phép và lương; giữ parent từ cả hai nguồn, kiểm tra đủ từng phần trước khi sinh đáp án.

### Ca 2

- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Nhân viên phải cam kết làm việc ít nhất 1 năm sau khi hoàn thành khóa học. Nghỉ sau 8 tháng là trước hạn cam kết, phải hoàn trả 100% chi phí tức 25.000.000 VNĐ.
- **Worst metric:** faithfulness
- **Context count:** 1

**Got (thực tế):**

> Nhân viên phải hoàn trả **100% chi phí** đào tạo đã được tài trợ, tức là **25 triệu VNĐ**.

- **Worst score:** 0.0000; **Mean four metrics:** 0.7038
- **Error Tree:** Output incorrect → inspect evidence → faithfulness → LLM hallucinating
- **Diagnostic hypothesis:** LLM hallucinating; cần kiểm tra context và đáp án trước khi kết luận.
- **Suggested fix:** Ground the prompt in retrieved evidence; lower temperature

- **Đối chiếu thủ công:** Đáp án 100%, tức 25 triệu, khớp ground truth và chính sách hoàn trả trong một năm. Faithfulness 0 là số đo thực tế. Có thể judge không coi số tiền trong câu hỏi là bằng chứng context; đây là giả thuyết cần trace, không phải bằng chứng hallucination.
- **Error Tree sau đối chiếu:** Đáp án khớp ground truth → kiểm tra điều kiện và dữ kiện trong câu hỏi → kiểm tra judge trace → phân biệt lỗi evaluator với lỗi generation.
- **Hướng sửa cụ thể:** Trình bày điều kiện 8 tháng < 1 năm và công thức 25 triệu × 100%; kiểm tra judge trace trước khi thay đổi pipeline. Không tự sửa điểm.

### Ca 3

- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày).
- **Worst metric:** faithfulness
- **Context count:** 1

**Got (thực tế):**

> Nhân viên tạm ứng 15 triệu VNĐ và thanh toán sau 20 ngày. Theo chính sách, khoản tạm ứng chưa thanh toán sau 15 ngày sẽ bị tính phí 2%/tháng trên số tiền chưa hoàn ứng.
>
> Số tiền phạt sẽ được tính như sau:
> - Số tiền tạm ứng: 15.000.000 VNĐ
> - Thời gian quá hạn: 5 ngày (20 ngày - 15 ngày)
> - Phí phạt: 2%/tháng tương đương với 0,0667%/ngày (2% chia cho 30 ngày)
>
> Tính phí phạt cho 5 ngày:
> - Phí phạt = 15.000.000 VNĐ * 0,0667% * 5 ngày = 5.000 VNĐ
>
> Vậy, nhân viên sẽ bị phạt 5.000 VNĐ.

- **Worst score:** 0.4000; **Mean four metrics:** 0.7132
- **Error Tree:** Output incorrect → inspect evidence → faithfulness → LLM hallucinating
- **Diagnostic hypothesis:** LLM hallucinating; cần kiểm tra context và đáp án trước khi kết luận.
- **Suggested fix:** Ground the prompt in retrieved evidence; lower temperature

- **Đối chiếu thủ công:** Đáp án ghi 5.000 đồng, trong khi 15.000.000 × 0,02 × (20−15)/30 = 50.000 đồng. Đây là lỗi số học xác minh được; các điều kiện phí trong câu trả lời vẫn đúng.
- **Error Tree sau đối chiếu:** Đủ điều kiện và số liệu → kiểm tra phép tính → kết quả lệch 10 lần → generation/arithmetic.
- **Hướng sửa cụ thể:** Tính bằng calculator hoặc công thức xác định; kiểm tra số tiền cuối trước khi LLM giải thích.

### Ca 4

- **Question:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Expected:** Có, theo chính sách mật khẩu v2.0 hiện hành, tất cả nhân viên bắt buộc kích hoạt MFA cho email, VPN và hệ thống nội bộ. Chính sách cũ v1.0 không yêu cầu MFA.
- **Worst metric:** context_recall
- **Context count:** 2

**Got (thực tế):**

> Có, tất cả nhân viên **bắt buộc** phải kích hoạt xác thực đa yếu tố (MFA) cho tài khoản email, VPN và các hệ thống nội bộ.

- **Worst score:** 0.5000; **Mean four metrics:** 0.8094
- **Error Tree:** Output incorrect → inspect evidence → context_recall → Missing relevant chunks
- **Diagnostic hypothesis:** Missing relevant chunks; cần kiểm tra context và đáp án trước khi kết luận.
- **Suggested fix:** Improve chunking, hybrid search and parent expansion

- **Đối chiếu thủ công:** Đáp án yêu cầu MFA đúng bản hiện hành. Ground truth có thêm lịch sử v1 không yêu cầu MFA, nhưng context thiếu phần lịch sử và có nguồn mua sắm không liên quan. Recall 0,5 không chứng minh đáp án hiện hành sai.
- **Error Tree sau đối chiếu:** Đáp án hiện hành đúng → đối chiếu ground truth → thiếu thông tin lịch sử → context recall/retrieval.
- **Hướng sửa cụ thể:** Gắn metadata phiên bản; chỉ lấy nguồn lịch sử khi cần đối chiếu, loại context mua sắm không liên quan.

### Ca 5

- **Question:** Mật khẩu phải có tối thiểu bao nhiêu ký tự?
- **Expected:** Theo chính sách hiện hành (v2.0), mật khẩu phải có tối thiểu 12 ký tự. Chính sách cũ (v1.0) yêu cầu 8 ký tự nhưng đã bị thay thế.
- **Worst metric:** context_precision
- **Context count:** 2

**Got (thực tế):**

> Mật khẩu phải có tối thiểu **12 ký tự** theo Chính sách mật khẩu phiên bản hiện hành (v2.0).

- **Worst score:** 0.5000; **Mean four metrics:** 0.8410
- **Error Tree:** Output incorrect → inspect evidence → context_precision → Too many irrelevant chunks
- **Diagnostic hypothesis:** Too many irrelevant chunks; cần kiểm tra context và đáp án trước khi kết luận.
- **Suggested fix:** Rerank candidates and filter obsolete versions

- **Đối chiếu thủ công:** Đáp án 12 ký tự đúng bản v2.0. Context đầu là bản v1.0 yêu cầu 8 ký tự, sau đó mới đến v2.0; precision 0,5 phù hợp việc còn một bản đã thay thế.
- **Error Tree sau đối chiếu:** Đáp án đúng → kiểm tra thứ tự nguồn → bản v1.0 đã thay thế đứng trước → context precision/version filtering.
- **Hướng sửa cụ thể:** Lọc trạng thái đã thay thế và ưu tiên ngày hiệu lực mới nhất nếu câu hỏi không yêu cầu lịch sử.

## Latency thực tế (CPU + API)

| Stage | Seconds |
|---|---:|
| chunking_seconds | 0.525 |
| enrichment_seconds | 0.086 |
| indexing_seconds | 69.010 |
| reranker_loading_seconds | 0.000 |
| Mean retrieval_seconds | 0.213 |
| Mean reranking_seconds | 12.874 |
| Mean generation_seconds | 1.798 |

Generation gọi LLM qua OpenRouter; xem JSON để đối chiếu latency từng câu.

## Case study và bước tiếp theo

Ca Senior 9 năm thâm niên yêu cầu cả chính sách phép 2024 và bảng lương. Kiểm tra
context có đủ hai nguồn trước khi sửa prompt. Nếu thiếu nguồn thì sửa retrieval/reranking;
nếu đủ nguồn nhưng đáp án sai thì sửa generation và phép tính. Không dùng ground truth
để tạo câu trả lời cho pipeline.

Trong một giờ tiếp theo: bổ sung metadata phiên bản, kiểm tra context của 5 ca,
rồi chạy lại RAGAS cùng cấu hình để so sánh trước/sau khi sửa. Chạy lại script này
để cập nhật bottom-5 theo report mới.
