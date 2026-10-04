"""Render the lab analysis from measured reports or clearly labeled offline evidence."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def main(update=False):
    production = json.loads((ROOT / "reports/ragas_report.json").read_text(encoding="utf-8"))
    baseline = json.loads((ROOT / "reports/naive_baseline_report.json").read_text(encoding="utf-8"))
    analysis_path = ROOT / "analysis/failure_analysis.md"
    measured = production.get("evaluation", {}).get("status") == "measured"
    if update and not measured:
        raise SystemExit("Cannot replace analysis with an unmeasured report.")
    if analysis_path.exists() and not update:
        print("Keeping existing failure_analysis.md; review new reports before updating analysis.")
        return
    both_measured = measured and baseline.get("evaluation", {}).get("status") == "measured"
    lines = [
        "# Failure Analysis — Lab 18",
        "",
        "Hoàng Công Minh · 2A202602774 · K4 Track 3A",
        "",
        "## RAGAS comparison",
        "",
        "| Metric | Baseline | Production | Delta |",
        "|---|---:|---:|---:|",
    ]
    for metric in METRICS:
        if both_measured:
            a, b = baseline["aggregate"][metric], production["aggregate"][metric]
            lines.append(f"| {metric} | {a:.4f} | {b:.4f} | {b - a:+.4f} |")
        else:
            lines.append(f"| {metric} | Chưa đo | Chưa đo | N/A |")
    configuration = production.get("evaluation", {}).get("configuration", {})
    if configuration.get("evaluation_mode") == "mixed_evaluators":
        lines += [
            "",
            "Production giữ 11 câu đã đo bằng OpenRouter và chỉ gửi 9 câu còn lại tới Gemini.",
            "Baseline giữ evaluator OpenRouter. Delta không phải so sánh có kiểm soát bằng cùng evaluator;",
            "không kết luận chênh lệch chỉ do pipeline. JSON ghi cấu hình evaluator cho từng câu.",
        ]
    if measured:
        cases = production["failures"][:5]
        lines += ["", "## Bottom-5 theo trung bình bốn metric RAGAS", ""]
    else:
        samples = production.get("evaluation", {}).get("samples", [])
        cases = [samples[i] for i in (3, 11, 12, 13, 16) if i < len(samples)]
        lines += [
            "",
            "## Năm ca review cục bộ; chưa phải bottom-5 RAGAS",
            "",
            "Evaluation đang unavailable vì chạy LAB_OFFLINE=1. Điểm 0 trong JSON là placeholder,",
            "không phải số đo. Chưa có căn cứ xếp hạng bottom-5. Các ca dưới được chọn để review",
            "version, multi-hop, workflow và numeric từ câu trả lời/context thực tế của lần chạy cục bộ.",
            "",
        ]
    fixes = [
        "Thêm effective_date/version; ưu tiên policy 2024 nếu query không hỏi lịch sử.",
        "Giữ cả nguồn nghỉ phép và bảng lương khi truy vấn multi-hop; tổng hợp và dẫn từng nguồn.",
        "Kết hợp quy trình mua sắm với yêu cầu CNTT; kiểm tra ngưỡng phê duyệt 30 triệu.",
        "Đọc điều kiện hoàn trả và thời gian cam kết; thực hiện phép tính từ bằng chứng, tránh suy đoán.",
        "Đối chiếu hạn thanh toán tạm ứng, số ngày trễ và tỷ lệ phạt; tính rồi nêu công thức.",
    ]
    for index, case in enumerate(cases):
        got = case.get("answer", "")
        excerpt = got[:600] + (" ... [rút gọn; xem JSON để đọc đầy đủ]" if len(got) > 600 else "")
        lines += [
            f"### Ca {index + 1}",
            "",
            f"- **Question:** {case['question']}",
            f"- **Expected:** {case.get('ground_truth', '')}",
            f"- **Worst metric:** {case.get('worst_metric', 'Chưa đo')}",
            f"- **Context count:** {len(case.get('contexts', []))}",
            "",
            "**Got (thực tế):**",
            "",
            "\n".join(("> " + line.rstrip() if line.strip() else ">") for line in excerpt.splitlines()),
            "",
        ]
        if measured:
            lines += [
                f"- **Worst score:** {case['score']:.4f}; **Mean four metrics:** {case['average_score']:.4f}",
                f"- **Error Tree:** {case['error_tree']}",
                f"- **Diagnostic hypothesis:** {case['diagnosis']}; cần kiểm tra context và đáp án trước khi kết luận.",
                f"- **Suggested fix:** {case['suggested_fix']}",
                "",
            ]
            question = case["question"]
            if "Senior" in question:
                review = "Đáp án đúng 18 ngày phép nhưng thiếu lương Senior 20–35 triệu. Context chỉ có chính sách phép, chưa đủ bảng lương. Context precision 0 phản ánh cách judge chấm câu hỏi hai phần; không kết luận toàn bộ context vô ích."
                fix = "Tách truy vấn phép và lương; giữ parent từ cả hai nguồn, kiểm tra đủ từng phần trước khi sinh đáp án."
                tree = "Câu hỏi hai phần → kiểm tra đủ nguồn → thiếu bảng lương → retrieval/parent expansion → đáp án thiếu phần lương."
            elif "25 triệu" in question:
                review = "Đáp án 100%, tức 25 triệu, khớp ground truth và chính sách hoàn trả trong một năm. Faithfulness 0 là số đo thực tế. Có thể judge không coi số tiền trong câu hỏi là bằng chứng context; đây là giả thuyết cần trace, không phải bằng chứng hallucination."
                fix = "Trình bày điều kiện 8 tháng < 1 năm và công thức 25 triệu × 100%; kiểm tra judge trace trước khi thay đổi pipeline. Không tự sửa điểm."
                tree = "Đáp án khớp ground truth → kiểm tra điều kiện và dữ kiện trong câu hỏi → kiểm tra judge trace → phân biệt lỗi evaluator với lỗi generation."
            elif "tạm ứng" in question:
                review = "Đáp án ghi 5.000 đồng, trong khi 15.000.000 × 0,02 × (20−15)/30 = 50.000 đồng. Đây là lỗi số học xác minh được; các điều kiện phí trong câu trả lời vẫn đúng."
                fix = "Tính bằng calculator hoặc công thức xác định; kiểm tra số tiền cuối trước khi LLM giải thích."
                tree = "Đủ điều kiện và số liệu → kiểm tra phép tính → kết quả lệch 10 lần → generation/arithmetic."
            elif "MFA" in question:
                review = "Đáp án yêu cầu MFA đúng bản hiện hành. Ground truth có thêm lịch sử v1 không yêu cầu MFA, nhưng context thiếu phần lịch sử và có nguồn mua sắm không liên quan. Recall 0,5 không chứng minh đáp án hiện hành sai."
                fix = "Gắn metadata phiên bản; chỉ lấy nguồn lịch sử khi cần đối chiếu, loại context mua sắm không liên quan."
                tree = "Đáp án hiện hành đúng → đối chiếu ground truth → thiếu thông tin lịch sử → context recall/retrieval."
            elif "Mật khẩu" in question:
                review = "Đáp án 12 ký tự đúng bản v2.0. Context đầu là bản v1.0 yêu cầu 8 ký tự, sau đó mới đến v2.0; precision 0,5 phù hợp việc còn một bản đã thay thế."
                fix = "Lọc trạng thái đã thay thế và ưu tiên ngày hiệu lực mới nhất nếu câu hỏi không yêu cầu lịch sử."
                tree = "Đáp án đúng → kiểm tra thứ tự nguồn → bản v1.0 đã thay thế đứng trước → context precision/version filtering."
            else:
                review = "Đối chiếu đáp án với ground truth và context trước khi coi diagnostic tự động là nguyên nhân đã xác minh."
                fix = case["suggested_fix"]
                tree = case["error_tree"]
            lines += [
                f"- **Đối chiếu thủ công:** {review}",
                f"- **Error Tree sau đối chiếu:** {tree}",
                f"- **Hướng sửa cụ thể:** {fix}",
                "",
            ]
        else:
            lines += [
                "- **Error Tree:** Output là trích đoạn → chưa có LLM generation → kiểm tra parent contexts trong JSON → kiểm tra đủ nguồn/phiên bản → chạy generation và evaluator được chấp thuận.",
                "- **Root cause xác minh được:** chế độ offline trả parent đầu tiên, không tổng hợp câu trả lời hoặc tính số. Chất lượng retrieval cần đối chiếu context; chưa kết luận metric thấp.",
                f"- **Suggested fix:** {fixes[index]}",
                "",
            ]
    latency = production.get("evaluation", {}).get("latency", {})
    lines += [
        ("## Latency thực tế (CPU + API)" if measured else "## Latency đo cục bộ"),
        "",
        "| Stage | Seconds |",
        "|---|---:|",
    ]
    for key, value in latency.get("build", {}).items():
        lines.append(f"| {key} | {value:.3f} |")
    queries = latency.get("queries", [])
    if queries:
        for key in ("retrieval_seconds", "reranking_seconds", "generation_seconds"):
            lines.append(f"| Mean {key} | {sum(row[key] for row in queries) / len(queries):.3f} |")
    lines += [
        "",
        (
            "Generation gọi LLM qua OpenRouter; xem JSON để đối chiếu latency từng câu."
            if measured
            else "Offline generation là trả trích đoạn, không đại diện cho latency LLM API."
        ),
        "",
        "## Case study và bước tiếp theo",
        "",
        "Ca Senior 9 năm thâm niên yêu cầu cả chính sách phép 2024 và bảng lương. Kiểm tra",
        "context có đủ hai nguồn trước khi sửa prompt. Nếu thiếu nguồn thì sửa retrieval/reranking;",
        "nếu đủ nguồn nhưng đáp án sai thì sửa generation và phép tính. Không dùng ground truth",
        "để tạo câu trả lời cho pipeline.",
        "",
        "Trong một giờ tiếp theo: bổ sung metadata phiên bản, kiểm tra context của 5 ca,",
        (
            "rồi chạy lại RAGAS cùng cấu hình để so sánh trước/sau khi sửa. Chạy lại script này"
            if measured
            else "rồi chạy RAGAS thật sau khi được đồng ý gửi dữ liệu tới provider. Chạy lại script này"
        ),
        ("để cập nhật bottom-5 theo report mới." if measured else "để thay phần review cục bộ bằng bottom-5 đã đo."),
        "",
    ]
    (ROOT / "analysis/failure_analysis.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="Explicitly replace analysis from a measured report")
    main(update=parser.parse_args().update)
