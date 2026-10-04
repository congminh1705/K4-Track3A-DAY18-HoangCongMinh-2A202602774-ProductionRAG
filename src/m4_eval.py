from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(
    questions: list[str], answers: list[str], contexts: list[list[str]], ground_truths: list[str]
) -> dict:
    """Run RAGAS evaluation."""
    import math

    from config import EVAL_EMBEDDING_MODEL, LLM_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL

    if len({len(questions), len(answers), len(contexts), len(ground_truths)}) != 1:
        raise ValueError("Evaluation inputs must have equal lengths")
    keys = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

    def unavailable(reason):
        return {
            **dict.fromkeys(keys, 0.0),
            "per_question": [],
            "status": "unavailable",
            "error": reason,
            "attempted_questions": len(questions),
            "samples": [
                dict(question=q, answer=a, contexts=c, ground_truth=g)
                for q, a, c, g in zip(questions, answers, contexts, ground_truths)
            ],
        }

    if not questions:
        return unavailable("Empty evaluation dataset")
    if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-...":
        return unavailable("OPENAI_API_KEY is missing or a placeholder; scores are not measured")
    if len(questions) > 1:
        import hashlib
        from dataclasses import asdict
        from pathlib import Path

        from config import REPORTS_DIR

        cache_dir = Path(REPORTS_DIR) / ".evaluation-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        completed = []
        configuration = None
        for index, (q, a, c, g) in enumerate(zip(questions, answers, contexts, ground_truths), 1):
            signature = json.dumps(
                [q, a, c, g, LLM_MODEL, EVAL_EMBEDDING_MODEL, OPENAI_BASE_URL, "same-language-v2-2048"],
                ensure_ascii=False,
            )
            cache_path = cache_dir / (hashlib.sha256(signature.encode()).hexdigest() + ".json")
            if cache_path.exists():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                completed.append(EvalResult(**cached["result"]))
                configuration = cached["configuration"]
                print(f"RAGAS {index}/{len(questions)}: reused measured checkpoint")
                continue
            result = evaluate_ragas([q], [a], [c], [g])
            if result["status"] != "measured":
                failed = unavailable(result["error"])
                failed["completed_questions"] = len(completed)
                return failed
            row = result["per_question"][0]
            configuration = result["configuration"]
            temporary = cache_path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"result": asdict(row), "configuration": configuration}), encoding="utf-8")
            os.replace(temporary, cache_path)
            completed.append(row)
            print(f"RAGAS {index}/{len(questions)}: measured and saved checkpoint")
        return {
            **{key: sum(getattr(row, key) for row in completed) / len(completed) for key in keys},
            "per_question": completed,
            "status": "measured",
            "configuration": configuration,
        }
    try:
        from copy import deepcopy

        from datasets import Dataset
        from langchain_core.callbacks import BaseCallbackHandler
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
        from ragas.run_config import RunConfig

        failures = []

        class CaptureErrors(BaseCallbackHandler):
            def on_llm_error(self, error, **kwargs):
                failures.append(error)

        dataset = Dataset.from_dict(
            {"question": questions, "answer": answers, "contexts": contexts, "ground_truth": ground_truths}
        )
        metrics = [deepcopy(m) for m in (faithfulness, answer_relevancy, context_precision, context_recall)]
        metrics[1].question_generation.instruction += (
            " Generate the reconstructed question in the same language as the answer; "
            "use Vietnamese for Vietnamese answers. Keep the noncommittal criteria unchanged."
        )
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=ChatOpenAI(
                model=LLM_MODEL,
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                temperature=0,
                max_tokens=1024,
            ),
            embeddings=OpenAIEmbeddings(
                model=EVAL_EMBEDDING_MODEL,
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                check_embedding_ctx_length=False,
                request_timeout=30,
                max_retries=1,
            ),
            run_config=RunConfig(timeout=60, max_retries=1, max_workers=1),
            raise_exceptions=False,
            callbacks=[CaptureErrors()],
        )
        rows = result.to_pandas().to_dict(orient="records")
        per_question = []
        for row in rows:
            scores = {key: float(row[key]) for key in keys}
            if not all(math.isfinite(v) for v in scores.values()):
                if failures:
                    detail = str(failures[-1]).replace(OPENAI_API_KEY, "[REDACTED]")[:1500]
                    return unavailable(f"RAGAS API failed: {detail}")
                return unavailable("RAGAS returned non-finite metrics")
            per_question.append(
                EvalResult(row["question"], row["answer"], list(row["contexts"]), row["ground_truth"], **scores)
            )
        return {
            **{key: sum(getattr(r, key) for r in per_question) / len(per_question) for key in keys},
            "per_question": per_question,
            "status": "measured",
            "configuration": {
                "llm_model": LLM_MODEL,
                "embedding_model": EVAL_EMBEDDING_MODEL,
                "reconstructed_question_language": "answer_language",
            },
        }
    except Exception as error:
        detail = str(error).replace(OPENAI_API_KEY, "[REDACTED]")[:1500]
        status = getattr(error, "status_code", None)
        reason = f"RAGAS failed: {type(error).__name__}"
        if status is not None:
            reason += f" (HTTP {status})"
        if detail:
            reason += f": {detail}"
        print(f"RAGAS evaluation unavailable: {reason}")
        return unavailable(reason)


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    tree = {
        "faithfulness": ("LLM hallucinating", "Ground the prompt in retrieved evidence; lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking, hybrid search and parent expansion"),
        "context_precision": ("Too many irrelevant chunks", "Rerank candidates and filter obsolete versions"),
        "answer_relevancy": ("Answer does not match question", "Improve answer instructions and handle ambiguity"),
    }
    failures = []
    for result in eval_results:
        values = {key: getattr(result, key) for key in tree}
        worst = min(values, key=values.get)
        diagnosis, fix = tree[worst]
        failures.append(
            {
                "question": result.question,
                "answer": result.answer,
                "ground_truth": result.ground_truth,
                "contexts": result.contexts,
                "worst_metric": worst,
                "score": values[worst],
                "average_score": sum(values.values()) / 4,
                "diagnosis": diagnosis,
                "suggested_fix": fix,
                "error_tree": f"Output incorrect → inspect evidence → {worst} → {diagnosis}",
            }
        )
    return sorted(failures, key=lambda f: f["average_score"])[: max(0, bottom_n)]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    from dataclasses import asdict

    from config import REPORTS_DIR

    if not os.path.isabs(path):
        path = os.path.join(REPORTS_DIR, os.path.basename(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    protected_statuses = {"measured", "measured_summary_recovered"}
    if results.get("status") not in protected_statuses and os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous.get("evaluation", {}).get("status") in protected_statuses:
            path = os.path.splitext(path)[0] + ".last_failed.json"
            print("Keeping successful report; saving failed attempt separately.")
    report = {
        "aggregate": {
            k: results[k] for k in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
        },
        "num_questions": len(results.get("per_question", [])),
        "evaluation": {
            k: v
            for k, v in results.items()
            if k not in ("per_question", "faithfulness", "answer_relevancy", "context_precision", "context_recall")
        },
        "failures": failures,
        "per_question": [asdict(r) for r in results.get("per_question", [])],
    }
    if results.get("status") == "measured" and os.path.exists(path):
        import shutil

        with open(path, encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous.get("evaluation", {}).get("status") in protected_statuses:
            shutil.copy2(path, os.path.splitext(path)[0] + ".previous.json")
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    os.replace(temporary, path)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
