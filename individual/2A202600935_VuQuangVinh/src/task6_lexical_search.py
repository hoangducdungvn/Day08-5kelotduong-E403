"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)

Cài đặt:
    pip install rank-bm25
"""

import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

# Load corpus từ chunk_index.json do Task 4 tạo ra
CHUNK_INDEX_PATH = Path(__file__).parent.parent / "data" / "chunk_index.json"

# Cache để không build lại index mỗi lần gọi
CORPUS: list[dict] = []
_bm25: BM25Okapi | None = None


def _load_corpus() -> list[dict]:
    """Load corpus từ chunk_index.json (Task 4 output)."""
    with open(CHUNK_INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}

    Returns:
        BM25Okapi index
    """
    # Tokenize — dùng split() cho tiếng Việt (đã được tách sẵn bằng dấu cách)
    # Lowercase để khớp query không phân biệt hoa thường
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def _get_index() -> tuple[list[dict], BM25Okapi]:
    """Lazy load corpus + build index, cache lại để tái dùng."""
    global CORPUS, _bm25
    if _bm25 is None:
        CORPUS = _load_corpus()
        _bm25 = build_bm25_index(CORPUS)
    return CORPUS, _bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    corpus, bm25 = _get_index()

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Lấy top_k indices theo score descending
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # bỏ qua chunk không có từ nào khớp
            results.append({
                "content": corpus[idx]["content"],
                "score": float(scores[idx]),
                "metadata": corpus[idx]["metadata"],
            })

    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")