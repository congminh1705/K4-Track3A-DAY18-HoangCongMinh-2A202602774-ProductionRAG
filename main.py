"""
Lab 18: Production RAG Pipeline — Main Entry Point
===================================================
Chạy toàn bộ pipeline: naive baseline → production → so sánh → report.

Usage:
    python main.py
"""

import json
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def main(resume=False, eval_only=False):
    print("=" * 60)
    print("LAB 18: PRODUCTION RAG PIPELINE")
    print("=" * 60)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    start = time.time()

    os.makedirs("reports", exist_ok=True)

    if eval_only:
        from src.m4_eval import evaluate_ragas, failure_analysis, save_report

        for filename in ("naive_baseline_report.json", "ragas_report.json"):
            path = os.path.join("reports", filename)
            with open(path, encoding="utf-8") as handle:
                report = json.load(handle)
            if report.get("evaluation", {}).get("status") == "measured":
                print(f"Keeping measured report: {filename}")
                continue
            samples = report.get("evaluation", {}).get("samples", [])
            if not samples:
                raise SystemExit(f"No saved answers in {filename}; run the pipeline first.")
            result = evaluate_ragas(
                *[[sample[key] for sample in samples] for key in ("question", "answer", "contexts", "ground_truth")]
            )
            if result["status"] != "measured":
                raise SystemExit(result["error"])
            if "latency" in report.get("evaluation", {}):
                result["latency"] = report["evaluation"]["latency"]
            save_report(result, failure_analysis(result["per_question"], bottom_n=5), path)
        return

    # Step 1: Basic Baseline
    print("\n📌 STEP 1: Running Basic RAG Baseline...")
    print("-" * 40)
    from naive_baseline import main as run_baseline

    baseline_path = os.path.join("reports", "naive_baseline_report.json")
    reuse = False
    if resume and os.path.exists(baseline_path):
        from config import EVAL_EMBEDDING_MODEL, LLM_MODEL
        from src.m4_eval import load_test_set

        with open(baseline_path, encoding="utf-8") as handle:
            existing = json.load(handle)
        config = existing.get("evaluation", {}).get("configuration", {})
        expected = [(q["question"], q["ground_truth"]) for q in load_test_set()]
        actual = [(q["question"], q["ground_truth"]) for q in existing.get("per_question", [])]
        reuse = (
            existing.get("evaluation", {}).get("status") == "measured"
            and config.get("llm_model") == LLM_MODEL
            and config.get("embedding_model") == EVAL_EMBEDDING_MODEL
            and expected == actual
        )
    if reuse:
        print("Reusing compatible measured baseline report.")
    else:
        run_baseline()

    # Step 2: Production Pipeline
    print("\n📌 STEP 2: Running Production Pipeline...")
    print("-" * 40)
    from src.pipeline import build_pipeline, evaluate_pipeline

    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)

    # Ensure reports are located in reports/
    for f in ["ragas_report.json", "naive_baseline_report.json"]:
        if os.path.exists(f):
            os.replace(f, f"reports/{f}")

    # Step 3: Comparison
    print("\n📌 STEP 3: Comparison")
    print("-" * 40)
    naive_path = "reports/naive_baseline_report.json"
    prod_path = "reports/ragas_report.json"

    if os.path.exists(naive_path) and os.path.exists(prod_path):
        with open(naive_path, encoding="utf-8") as f:
            naive = json.load(f)
        with open(prod_path, encoding="utf-8") as f:
            prod = json.load(f)

        print(f"\n{'Metric':<25} {'Basic':>8} {'Production':>12} {'Δ':>8}")
        print("-" * 55)
        for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if (
                naive.get("evaluation", {}).get("status") != "measured"
                or prod.get("evaluation", {}).get("status") != "measured"
            ):
                print(f"  {m:<23} Chưa đo; không có delta hợp lệ")
                continue
            n = naive.get("aggregate", {}).get(m, 0)
            p = prod.get("aggregate", {}).get(m, 0)
            d = p - n
            status = "✓" if p >= 0.75 else " "
            print(f"{status} {m:<23} {n:>8.4f} {p:>12.4f} {d:>+8.4f}")

    from render_lab_report import main as render_analysis

    render_analysis()

    elapsed = time.time() - start
    print(f"\n⏱️  Total time: {elapsed:.1f}s")
    print("\n📋 Next steps:")
    print("  1. Điền analysis/failure_analysis.md")
    print("  2. Viết analysis/reflections/reflection_[HọTên].md")
    print("  3. Chạy: python check_lab.py")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Reuse a compatible measured baseline report")
    parser.add_argument(
        "--eval-only", action="store_true", help="Evaluate saved answers without rebuilding the pipeline"
    )
    args = parser.parse_args()
    main(resume=args.resume, eval_only=args.eval_only)
