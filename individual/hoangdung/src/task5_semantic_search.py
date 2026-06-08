"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    import faiss
    import numpy as np
    import pickle
    from pathlib import Path
    from sentence_transformers import SentenceTransformer
    
    # Load model (cùng model với Task 4)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = model.encode([query]).astype("float32")
    
    # Đường dẫn file
    db_path = Path(__file__).parent.parent / "data" / "vector_store"
    index_file = db_path / "faiss_index.bin"
    meta_file = db_path / "metadatas.pkl"
    
    if not index_file.exists() or not meta_file.exists():
        return []
        
    # Load FAISS index và metadata
    index = faiss.read_index(str(index_file))
    with open(meta_file, "rb") as f:
        metadatas = pickle.load(f)
        
    # Search
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1: # Không tìm thấy
            continue
            
        # Trong L2 distance, khoảng cách nhỏ là giống nhau. 
        # Convert sang score giảm dần: 1 / (1 + distance)
        score = 1.0 / (1.0 + float(dist))
        
        meta = metadatas[idx]
        results.append({
            "content": meta["content"],
            "score": score,
            "metadata": meta["metadata"]
        })
        
    # Sắp xếp theo score giảm dần
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        content_safe = r['content'][:100].encode('cp1258', 'replace').decode('cp1258')
        print(f"[{r['score']:.3f}] {content_safe}...")
