"""Checks for integration defects beyond the original scaffold tests."""

import numpy as np
import pytest
from qdrant_client import QdrantClient

from src.m1_chunking import chunk_hierarchical, chunk_structure_aware
from src.m2_search import DenseSearch, SearchResult, reciprocal_rank_fusion
from src.m4_eval import evaluate_ragas


def test_failed_evaluation_preserves_successful_report(tmp_path):
    import json

    from src.m4_eval import save_report

    path = tmp_path / "ragas_report.json"
    original = {"evaluation": {"status": "measured"}, "aggregate": {"faithfulness": 0.92}}
    path.write_text(json.dumps(original), encoding="utf-8")
    scores = dict.fromkeys(("faithfulness", "answer_relevancy", "context_precision", "context_recall"), 0.0)
    save_report({**scores, "status": "unavailable", "error": "HTTP 402"}, [], str(path))
    assert json.loads(path.read_text(encoding="utf-8")) == original
    failed = json.loads((tmp_path / "ragas_report.last_failed.json").read_text(encoding="utf-8"))
    assert failed["evaluation"]["status"] == "unavailable"


def test_evaluation_resumes_only_matching_measured_questions(monkeypatch, tmp_path):
    import ragas

    import config

    calls = []
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-api-key")
    monkeypatch.setattr(config, "REPORTS_DIR", str(tmp_path))

    def fake_evaluate(dataset, **kwargs):
        calls.append(dataset["question"][0])

        class Result:
            def to_pandas(self):
                frame = dataset.to_pandas()
                for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
                    frame[metric] = 1.0
                return frame

        return Result()

    monkeypatch.setattr(ragas, "evaluate", fake_evaluate)
    arguments = (["first", "second"], ["a", "b"], [["c"], ["d"]], ["g", "h"])
    assert evaluate_ragas(*arguments)["status"] == "measured"
    assert evaluate_ragas(*arguments)["status"] == "measured"
    assert calls == ["first", "second"]
    arguments[1][1] = "updated answer"
    assert evaluate_ragas(*arguments)["status"] == "measured"
    assert calls == ["first", "second", "second"]


def test_parent_ids_are_unique_and_chunks_bounded():
    a, children = chunk_hierarchical("word " * 100, parent_size=80, child_size=20, metadata={"source": "a"})
    b, _ = chunk_hierarchical("word " * 100, parent_size=80, child_size=20, metadata={"source": "b"})
    assert not {p.metadata["parent_id"] for p in a} & {p.metadata["parent_id"] for p in b}
    assert all(len(p.text) <= 80 for p in a)
    assert all(len(c.text) <= 20 for c in children)


def test_structure_does_not_split_header_inside_code():
    chunks = chunk_structure_aware("# Actual\n```python\n# code comment\n```\n## Next\nbody")
    assert len(chunks) == 2
    assert "# code comment" in chunks[0].text


def test_rrf_exact_score_and_source_identity():
    a = SearchResult("same", 0.8, {"source": "a"}, "dense")
    b = SearchResult("same", 0.7, {"source": "b"}, "bm25")
    result = reciprocal_rank_fusion([[a, a], [a, b]])
    assert len(result) == 2
    assert result[0].score == pytest.approx(2 / 61)


def test_dense_qdrant_roundtrip():
    class Encoder:
        def encode(self, texts, **kwargs):
            if isinstance(texts, str):
                return np.array([1.0, 0.0])
            return np.array([[1.0, 0.0], [0.0, 1.0]])

    dense = DenseSearch.__new__(DenseSearch)
    dense.client = QdrantClient(":memory:")
    dense._encoder = Encoder()
    dense.index([{"text": "leave", "metadata": {"parent_id": "p1"}}, {"text": "vpn"}])
    result = dense.search("leave", top_k=1)
    assert result[0].text == "leave"
    assert result[0].metadata["parent_id"] == "p1"


def test_eval_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        evaluate_ragas(["q"], [], [["c"]], ["gt"])


def test_pipeline_expands_parent_from_reranked_child(monkeypatch):
    from types import SimpleNamespace

    import config
    from src.pipeline import run_query

    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    child = SearchResult("child", 1.0, {"parent_id": "p1"}, "hybrid")
    search = SimpleNamespace(precomputed_results={"q": [child]}, parent_lookup={"p1": "complete parent"})
    reranker = SimpleNamespace(rerank=lambda *args, **kwargs: [child])
    answer, contexts = run_query("q", search, reranker)
    assert answer == "complete parent"
    assert contexts == ["complete parent"]


def test_evaluation_retrieves_batch_before_loading_reranker(monkeypatch):
    from types import SimpleNamespace

    import config
    import src.pipeline as pipeline

    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    events = []

    class Search:
        dense = SimpleNamespace(_encoder=None)
        timings = {}
        parent_lookup = {}

        def search(self, query):
            events.append("search:" + query)
            return [SearchResult("evidence", 1.0, {}, "hybrid")]

    class Reranker:
        def _load_model(self):
            events.append("load")

        def rerank(self, query, docs, **kwargs):
            events.append("rerank:" + query)
            return []

    monkeypatch.setattr(
        pipeline,
        "load_test_set",
        lambda: [{"question": "a", "ground_truth": "gt"}, {"question": "b", "ground_truth": "gt"}],
    )
    monkeypatch.setattr(
        pipeline,
        "evaluate_ragas",
        lambda *args: {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": [],
            "status": "unavailable",
        },
    )
    monkeypatch.setattr(pipeline, "save_report", lambda *args: None)
    pipeline.evaluate_pipeline(Search(), Reranker())
    assert events == ["search:a", "search:b", "load", "rerank:a", "rerank:b"]


def test_evaluator_preserves_shared_metrics_and_matches_answer_language(monkeypatch):
    import ragas
    from ragas.metrics import answer_relevancy

    import config

    original = answer_relevancy.question_generation.instruction
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-api-key")
    monkeypatch.setattr(config, "EVAL_EMBEDDING_MODEL", "openai/text-embedding-3-small")

    def fake_evaluate(dataset, metrics, embeddings, **kwargs):
        assert "same language as the answer" in metrics[1].question_generation.instruction
        assert embeddings.model == "openai/text-embedding-3-small"

        class Result:
            def to_pandas(self):
                frame = dataset.to_pandas()
                for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
                    frame[name] = 1.0
                return frame

        return Result()

    monkeypatch.setattr(ragas, "evaluate", fake_evaluate)
    result = evaluate_ragas(["q"], ["a"], [["c"]], ["gt"])
    assert result["status"] == "measured"
    assert len(result["per_question"]) == 1
    assert answer_relevancy.question_generation.instruction == original


def test_combined_enrichment_reuses_cached_api_response(monkeypatch, tmp_path):
    import json

    import config
    import src.m5_enrichment as enrichment

    monkeypatch.setattr(config, "REPORTS_DIR", str(tmp_path))
    monkeypatch.setattr(enrichment, "OPENAI_API_KEY", "test-api-key")
    calls = []

    def request(*args, **kwargs):
        calls.append(1)
        return json.dumps(
            {"summary": "summary", "questions": ["question?"], "context": "context", "metadata": {"topic": "topic"}}
        )

    monkeypatch.setattr(enrichment, "_request", request)
    first = enrichment._enrich_single_call("content", "source")
    second = enrichment._enrich_single_call("content", "source")
    assert first == second
    assert len(calls) == 1
    assert "test-api-key" not in (tmp_path / "enrichment_cache.json").read_text()


def test_answer_checkpoint_rejects_different_fingerprint(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import config
    from src.pipeline import _load_answers, _save_answers

    monkeypatch.setattr(config, "REPORTS_DIR", str(tmp_path))
    search = SimpleNamespace(fingerprint="v1")
    answers = {"q": {"answer": "a", "contexts": ["c"], "timings": {}}}
    _save_answers(search, answers)
    assert _load_answers(search) == answers
    assert _load_answers(SimpleNamespace(fingerprint="v2")) == {}
