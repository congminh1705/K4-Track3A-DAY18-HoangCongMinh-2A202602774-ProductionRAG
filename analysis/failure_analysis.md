# Failure Analysis — Lab 18

Hoàng Công Minh · 2A202602774

## Kết quả lần chạy thành công


| Metric | Baseline | Production | Delta |
|---|---:|---:|---:|
| faithfulness | 0.7958 | 0.9200 | +0.1242 |
| answer_relevancy | 0.6571 | 0.7943 | +0.1372 |
| context_precision | 0.9250 | 0.9500 | +0.0250 |
| context_recall | 0.9250 | 0.9500 | +0.0250 |

## Năm ca đã phân tích khi hoàn thành lab

Không thể xác minh lại thứ hạng bottom-5 hoặc toàn bộ điểm từng câu vì JSON gốc đã mất.

### Senior 9 năm: nghỉ phép và lương

- Quan sát: Đáp án nêu 18 ngày phép nhưng thiếu khoảng lương 20–35 triệu; context thiếu bảng lương.
- Error Tree: Câu hỏi nhiều phần → thiếu nguồn → retrieval/parent expansion.
- Hướng sửa: Tách truy vấn phép và lương; giữ context từ cả hai nguồn.

### Kích hoạt MFA

- Quan sát: Đáp án đúng chính sách hiện hành; ground truth còn nhắc phiên bản lịch sử không có trong context.
- Error Tree: Thiếu lịch sử → kiểm tra phiên bản → context recall.
- Hướng sửa: Bổ sung metadata phiên bản và truy hồi bản lịch sử khi cần.

### Hoàn trả khóa học 25 triệu sau 8 tháng

- Quan sát: Đáp án hoàn trả 100%, tức 25 triệu, phù hợp ground truth. Faithfulness thấp có thể do số tiền nằm trong câu hỏi thay vì context; đây là giả thuyết.
- Error Tree: Đáp án phù hợp → kiểm tra bằng chứng định lượng → kiểm tra judge trace.
- Hướng sửa: Trình bày công thức và điều kiện; không tự sửa điểm đo.

### Số ngày phép năm

- Quan sát: Đáp án 15 ngày đúng chính sách 2024; context vẫn chứa bản 2023 ở vị trí đầu.
- Error Tree: Context có bản cũ → kiểm tra ngày hiệu lực → context precision.
- Hướng sửa: Gắn effective_date/version và ưu tiên bản hiện hành.

### Phạt tạm ứng 15 triệu, thanh toán sau 20 ngày

- Quan sát: Đáp án 5.000 đồng sai; kết quả mong đợi khoảng 50.000 đồng theo 15.000.000 × 0,02 × 5/30.
- Error Tree: Đủ số liệu → kiểm tra phép tính → generation/arithmetic.
- Hướng sửa: Dùng calculator hoặc phép tính xác định trước khi giải thích.

## Giới hạn

Metrics lịch sử đáp ứng ngưỡng bonus. Cần đo lại thành công để có đầy đủ điểm từng câu
và xếp hạng bottom-5. Các hướng sửa trên là đề xuất, chưa được đo kiểm chứng cải thiện.
