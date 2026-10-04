"""Render the lab analysis from measured reports or clearly labeled offline evidence."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def main():
    production = json.loads((ROOT / "reports/ragas_report.json").read_text(encoding="utf-8"))
    baseline = json.loads((ROOT / "reports/naive_baseline_report.json").read_text(encoding="utf-8"))
    analysis_path = ROOT / "analysis/failure_analysis.md"
    if analysis_path.exists():
        print("Keeping existing failure_analysis.md; review new reports before updating analysis.")
        return
    measured = production.get("evaluation", {}).get("status") == "measured"
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
                f"- **Error Tree:** {case['error_tree']}",
                f"- **Root cause:** {case['diagnosis']}",
                f"- **Suggested fix:** {case['suggested_fix']}",
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
    main()
