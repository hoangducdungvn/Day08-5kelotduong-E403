"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import sys
import pickle
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# TODO: Load corpus từ data/standardized/ hoặc từ vector store
def load_corpus() -> list[dict]:
    # Thử load từ vectorstore.pkl trước
    vectorstore_path = Path(__file__).parent.parent / "data" / "vectorstore.pkl"
    if vectorstore_path.exists():
        try:
            with open(vectorstore_path, "rb") as f:
                chunks = pickle.load(f)
                if chunks:
                    return [{"content": c["content"], "metadata": c["metadata"]} for c in chunks]
        except Exception:
            pass
            
    # Fallback: load và chunk dynamically
    try:
        from src.task4_chunking_indexing import load_documents, chunk_documents
        docs = load_documents()
        chunks = chunk_documents(docs)
        return [{"content": c["content"], "metadata": c["metadata"]} for c in chunks]
    except Exception:
        return []

CORPUS = load_corpus()


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi
    
    # Tokenize đơn giản bằng regex tách chữ
    def tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenize

BM25_INDEX, TOKENIZE_FN = build_bm25_index(CORPUS) if CORPUS else (None, None)


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
    global CORPUS, BM25_INDEX, TOKENIZE_FN
    if not CORPUS:
        CORPUS = load_corpus()
        if CORPUS:
            BM25_INDEX, TOKENIZE_FN = build_bm25_index(CORPUS)

    if not CORPUS or not BM25_INDEX:
        return []

    tokenized_query = TOKENIZE_FN(query)
    scores = BM25_INDEX.get_scores(tokenized_query)

    import numpy as np
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        # Giữ lại các kết quả có điểm số >= 0
        if scores[idx] >= 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx]["metadata"]
            })
    return results


if __name__ == "__main__":
    # Set console output encoding to UTF-8
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
