"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


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
    # Thử kết nối Weaviate trước nếu có cấu hình
    try:
        import os
        import weaviate
        from dotenv import load_dotenv
        from sentence_transformers import SentenceTransformer
        from src.task4_chunking_indexing import EMBEDDING_MODEL
        
        load_dotenv()
        weaviate_url = os.getenv("WEAVIATE_URL")
        weaviate_key = os.getenv("WEAVIATE_API_KEY")
        
        if weaviate_url and "xxx" not in weaviate_url:
            model = SentenceTransformer(EMBEDDING_MODEL)
            query_vector = model.encode(query).tolist()
            
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=weaviate_url,
                auth_credentials=weaviate.auth.AuthApiKey(weaviate_key)
            )
            collection = client.collections.get("DrugLawDocs")
            
            from weaviate.classes.query import MetadataQuery
            results = collection.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                return_metadata=MetadataQuery(distance=True)
            )
            
            output = []
            for obj in results.objects:
                # distance to similarity score
                score = 1.0 - obj.metadata.distance if obj.metadata.distance is not None else 0.0
                output.append({
                    "content": obj.properties["content"],
                    "score": score,
                    "metadata": {
                        "source": obj.properties.get("source"),
                        "type": obj.properties.get("doc_type"),
                        "chunk_index": obj.properties.get("chunk_index", 0)
                    }
                })
            client.close()
            output.sort(key=lambda x: x["score"], reverse=True)
            return output[:top_k]
    except Exception as e:
        # Fallback hoàn toàn xuống local pickle
        pass

    # Fallback: Chạy tìm kiếm cục bộ (Local Vector Store)
    import pickle
    import numpy as np
    from pathlib import Path
    from sentence_transformers import SentenceTransformer
    from src.task4_chunking_indexing import EMBEDDING_MODEL

    vectorstore_path = Path(__file__).parent.parent / "data" / "vectorstore.pkl"
    if not vectorstore_path.exists():
        print(f"Warning: Vectorstore file not found at {vectorstore_path}")
        return []

    with open(vectorstore_path, "rb") as f:
        chunks = pickle.load(f)

    if not chunks:
        return []

    # Embed query
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_vector = model.encode(query)

    results = []
    for chunk in chunks:
        if "embedding" not in chunk:
            continue
        chunk_vector = np.array(chunk["embedding"])
        
        # Tính cosine similarity
        dot_product = np.dot(query_vector, chunk_vector)
        norm_q = np.linalg.norm(query_vector)
        norm_c = np.linalg.norm(chunk_vector)
        similarity = float(dot_product / (norm_q * norm_c)) if norm_q > 0 and norm_c > 0 else 0.0

        results.append({
            "content": chunk["content"],
            "score": similarity,
            "metadata": chunk["metadata"]
        })

    # Sắp xếp giảm dần theo score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
