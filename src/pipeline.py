from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LLM_MODEL, OPENAI_BASE_URL, RERANK_MODEL, RERANK_TOP_K
from src.m1_chunking import chunk_hierarchical, load_documents
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import evaluate_ragas, failure_analysis, load_test_set, save_report
from src.m5_enrichment import enrich_chunks


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    parent_lookup = {}
    timings = {}
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        parent_lookup.update({p.metadata["parent_id"]: p.text for p in parents})
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time() - t0:.1f}s)", flush=True)

    timings["chunking_seconds"] = time.time() - t0

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time() - t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    timings["enrichment_seconds"] = time.time() - t0

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    print(f"  ✓ Indexed ({time.time() - t0:.1f}s)", flush=True)

    timings["indexing_seconds"] = time.time() - t0
    search.parent_lookup = parent_lookup
    search.timings = timings
    import hashlib
    import json

    from config import EMBEDDING_MODEL, OFFLINE

    search.fingerprint = hashlib.sha256(
        json.dumps(
            [all_chunks, EMBEDDING_MODEL, RERANK_MODEL, LLM_MODEL, OPENAI_BASE_URL, "pipeline-v1", OFFLINE],
            ensure_ascii=False,
            sort_keys=True,
        ).encode()
    ).hexdigest()

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker(RERANK_MODEL)
    timings["reranker_loading_seconds"] = time.time() - t0
    print(f"  ✓ Reranker configured (lazy loading) ({time.time() - t0:.1f}s)", flush=True)

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    t0 = time.perf_counter()
    precomputed = getattr(search, "precomputed_results", {})
    if query in precomputed:
        results = precomputed[query]
    else:
        # Keep one large model resident on machines with limited Windows paging space.
        from src.m3_rerank import _cached_cross_encoder

        reranker._model = None
        _cached_cross_encoder.cache_clear()
        import gc

        gc.collect()
        results = search.search(query)
        _release_dense(search)
    retrieval_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    rerank_seconds = time.perf_counter() - t0
    contexts = []
    for result in reranked or results[:RERANK_TOP_K]:
        text = getattr(search, "parent_lookup", {}).get(result.metadata.get("parent_id"), result.text)
        if text not in contexts:
            contexts.append(text)
    search.last_query_timings = {"retrieval_seconds": retrieval_seconds, "reranking_seconds": rerank_seconds}
    t0 = time.perf_counter()

    from config import OPENAI_API_KEY

    if OPENAI_API_KEY and contexts:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, timeout=30, max_retries=0)
            context_str = "\n\n".join(contexts)
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": "Trả lời bằng tiếng Việt CHỈ dựa trên context; dẫn tên nguồn khi có. Ưu tiên phiên bản hiện hành trừ khi hỏi lịch sử. Nếu thiếu bằng chứng nói Không tìm thấy; nếu câu hỏi mơ hồ hãy yêu cầu làm rõ.",
                    },
                    {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
                ],
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {type(e).__name__}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    search.last_query_timings["generation_seconds"] = time.perf_counter() - t0
    return answer, contexts


def _release_dense(search):
    import gc

    from src.m2_search import _cached_dense_encoder

    search.dense._encoder = None
    _cached_dense_encoder.cache_clear()
    gc.collect()


def _load_answers(search):
    import json
    from pathlib import Path

    from config import REPORTS_DIR

    if not getattr(search, "fingerprint", None):
        return {}
    try:
        data = json.loads((Path(REPORTS_DIR) / "production_answers.json").read_text(encoding="utf-8"))
        if data.get("fingerprint") == search.fingerprint:
            return data.get("items", {})
    except (OSError, ValueError):
        pass
    return {}


def _save_answers(search, items):
    import json
    from pathlib import Path

    from config import REPORTS_DIR

    if not getattr(search, "fingerprint", None):
        return
    path = Path(REPORTS_DIR) / "production_answers.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"fingerprint": search.fingerprint, "items": items}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    saved_answers = _load_answers(search)
    pending = [item for item in test_set if item["question"] not in saved_answers]
    query_timings = []
    search.precomputed_results = {}
    retrieval_timings = {}
    # Retrieve the full batch before loading the large cross-encoder.
    for item in pending:
        started = time.perf_counter()
        search.precomputed_results[item["question"]] = search.search(item["question"])
        retrieval_timings[item["question"]] = time.perf_counter() - started
    _release_dense(search)
    started = time.perf_counter()
    if pending:
        reranker._load_model()
    search.timings["reranker_loading_seconds"] = time.perf_counter() - started
    for i, item in enumerate(test_set):
        if item["question"] in saved_answers:
            cached = saved_answers[item["question"]]
            answer, contexts = cached["answer"], cached["contexts"]
            timing = {**cached["timings"], "reused": True}
        else:
            answer, contexts = run_query(item["question"], search, reranker)
            search.last_query_timings["retrieval_seconds"] = retrieval_timings[item["question"]]
            timing = search.last_query_timings
            saved_answers[item["question"]] = {"answer": answer, "contexts": contexts, "timings": timing}
            _save_answers(search, saved_answers)
        query_timings.append({"question": item["question"], **timing})
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i + 1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    import gc

    from src.m3_rerank import _cached_cross_encoder

    reranker._model = None
    _cached_cross_encoder.cache_clear()
    gc.collect()
    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print(f"  ✓ RAGAS done ({time.time() - t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print(f"Evaluation status: {results.get('status')}")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if results.get("status") != "measured":
            print(f"  {m}: chưa đo")
            continue
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    from config import EMBEDDING_MODEL, MODEL_DTYPE, OFFLINE

    results.setdefault("configuration", {}).update(
        {
            "retrieval_embedding_model": EMBEDDING_MODEL,
            "rerank_model": RERANK_MODEL,
            "model_dtype": MODEL_DTYPE,
            "offline": OFFLINE,
        }
    )
    results["latency"] = {"build": search.timings, "queries": query_timings, "evaluation_seconds": time.time() - t0}
    failures = failure_analysis(results.get("per_question", []), bottom_n=5)
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
