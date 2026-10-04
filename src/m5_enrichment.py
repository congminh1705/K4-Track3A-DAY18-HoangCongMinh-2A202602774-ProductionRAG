from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""

    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    result = _request("Tóm tắt đoạn văn trong 2 câu ngắn bằng tiếng Việt.", text)
    return result or _fallback(text, "")["summary"]


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if n_questions <= 0:
        return []
    result = _request(f"Tạo {n_questions} câu hỏi mà đoạn văn trả lời được. Mỗi câu trên một dòng.", text)
    if result:
        import re

        return [re.sub(r"^[\d. )-]+", "", q.strip()) for q in result.splitlines() if q.strip()][:n_questions]
    return _fallback(text, "")["questions"][:n_questions]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    context = _request(
        "Viết 1 câu mô tả vị trí và chủ đề đoạn văn trong tài liệu.", f"Tài liệu: {document_title}\n{text}"
    )
    context = context or _fallback(text, document_title)["context"]
    return f"{context}\n\n{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    result = _request("Trả về JSON metadata gồm topic, entities (list), category, language.", text, json_mode=True)
    if result:
        import json

        try:
            data = json.loads(result)
            if isinstance(data, dict):
                return data
        except (ValueError, TypeError):
            pass
    return _fallback(text, "")["metadata"]


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    import json
    from hashlib import sha256

    from config import LLM_MODEL, OPENAI_BASE_URL

    cache_key = sha256(
        json.dumps([text, source, LLM_MODEL, OPENAI_BASE_URL, "combined-v1"], ensure_ascii=False).encode()
    ).hexdigest()
    cached = _cache_get(cache_key) if OPENAI_API_KEY else None
    if cached is not None:
        return cached
    result = _request(
        "Phân tích đoạn văn, chỉ dựa trên nội dung được cung cấp. Trả về JSON gồm "
        "summary (string), questions (list 3 câu hỏi), context (string), "
        "metadata (object gồm topic, entities, category, language).",
        f"Tài liệu: {source}\n\n{text}",
        json_mode=True,
    )
    fallback = _fallback(text, source)
    if result:
        try:
            data = json.loads(result)
            if not isinstance(data, dict):
                return fallback
            for key, expected in (("summary", str), ("context", str), ("questions", list), ("metadata", dict)):
                if not isinstance(data.get(key), expected):
                    data[key] = fallback[key]
            data["questions"] = [q for q in data["questions"] if isinstance(q, str)]
            _cache_put(cache_key, data)
            return data
        except (ValueError, TypeError):
            pass
    return fallback


from threading import Lock

_CACHE_LOCK = Lock()


def _cache_get(key):
    import json
    from pathlib import Path

    from config import REPORTS_DIR

    with _CACHE_LOCK:
        try:
            cache = json.loads((Path(REPORTS_DIR) / "enrichment_cache.json").read_text(encoding="utf-8"))
            value = cache.get(key) if isinstance(cache, dict) else None
            if isinstance(value, dict) and all(
                isinstance(value.get(k), t)
                for k, t in (("summary", str), ("questions", list), ("context", str), ("metadata", dict))
            ):
                if all(isinstance(question, str) for question in value["questions"]):
                    return value
        except (OSError, ValueError):
            pass
    return None


def _cache_put(key, value):
    import json
    from pathlib import Path

    from config import REPORTS_DIR

    with _CACHE_LOCK:
        path = Path(REPORTS_DIR) / "enrichment_cache.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(cache, dict):
                cache = {}
        except (OSError, ValueError):
            cache = {}
        cache[key] = value
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)


def _request(instruction, text, json_mode=False):
    if not OPENAI_API_KEY or OPENAI_API_KEY in ("sk-...", "your-api-key"):
        return ""
    from config import LLM_MODEL, OPENAI_BASE_URL

    try:
        from openai import OpenAI

        options = {"response_format": {"type": "json_object"}} if json_mode else {}
        response = OpenAI(
            api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, timeout=30, max_retries=0
        ).chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": instruction}, {"role": "user", "content": text}],
            temperature=0,
            max_tokens=450,
            **options,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as error:
        print(f"Enrichment fallback: {type(error).__name__}")
        return ""


def _fallback(text, source):
    import re

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", text) if s.strip()]
    topic = next((s.lstrip("# ") for s in sentences if s.startswith("#")), "Chính sách nội bộ")
    return {
        "summary": " ".join(sentences[:2]),
        "questions": [s.rstrip(".!?") + "?" for s in sentences[:3]],
        "context": f"Trích từ tài liệu {source or 'nội bộ'}; chủ đề: {topic}.",
        "metadata": {"topic": topic, "entities": [], "category": "policy", "language": "vi"},
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    combined_results = []
    if use_combined:
        from concurrent.futures import ThreadPoolExecutor

        workers = max(1, min(8, int(os.getenv("ENRICHMENT_WORKERS", "4")))) if OPENAI_API_KEY else 1
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = pool.map(
                lambda c: _enrich_single_call(c["text"], c.get("metadata", {}).get("source", "")), chunks
            )
            for i, result in enumerate(results, 1):
                combined_results.append(result)
                if i % 10 == 0 or i == len(chunks):
                    print(f"  Generated enrichment {i}/{len(chunks)}...", flush=True)
    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = combined_results[i]
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        additions = ([f"Tóm tắt: {summary}"] if summary else []) + questions
        if additions:
            enriched_text += "\n\n" + "\n".join(additions)
        enriched.append(
            EnrichedChunk(
                original_text=text,
                enriched_text=enriched_text,
                summary=summary,
                hypothesis_questions=questions,
                auto_metadata={**auto_meta, **chunk.get("metadata", {})},
                method="+".join(methods),
            )
        )

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
