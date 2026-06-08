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

import pickle
from pathlib import Path

# Load corpus từ data/vector_store/metadatas.pkl đã tạo ở Task 4
db_path = Path(__file__).parent.parent / "data" / "vector_store"
meta_file = db_path / "metadatas.pkl"

if meta_file.exists():
    with open(meta_file, "rb") as f:
        CORPUS = pickle.load(f)
else:
    CORPUS = []
    
# Build index
def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi
    import re
    
    # Tokenize đơn giản: lowercase và split
    # Xóa dấu câu cơ bản để search chính xác hơn
    tokenized_corpus = []
    for doc in corpus:
        text = doc["content"].lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokenized_corpus.append(text.split())
        
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


bm25_index = build_bm25_index(CORPUS) if CORPUS else None

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
    import numpy as np
    import re
    
    global CORPUS, bm25_index
    if not CORPUS:
        return []
        
    query_clean = re.sub(r'[^\w\s]', ' ', query.lower())
    tokenized_query = query_clean.split()
    
    scores = bm25_index.get_scores(tokenized_query)
    
    # Get top_k indices
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx]["metadata"]
            })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        content_safe = r['content'][:100].encode('cp1258', 'replace').decode('cp1258')
        print(f"[{r['score']:.3f}] {content_safe}...")
