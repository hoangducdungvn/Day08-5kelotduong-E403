"""Task 5 - Semantic search over the local chunk index."""

import json
from pathlib import Path

try:
    from .rag_utils import cosine_similarity, default_chunks, hashed_embedding
    from .task4_chunking_indexing import EMBEDDING_DIM, INDEX_PATH, embed_chunks
except ImportError:
    from rag_utils import cosine_similarity, default_chunks, hashed_embedding
    from task4_chunking_indexing import EMBEDDING_DIM, INDEX_PATH, embed_chunks


def _load_index() -> list[dict]:
    if Path(INDEX_PATH).exists():
        return json.loads(Path(INDEX_PATH).read_text(encoding="utf-8"))
    return embed_chunks(default_chunks())


def semantic_search(query: str, top_k: int = 5) -> list[dict]:
    """Search chunks by cosine similarity against local hashed embeddings."""
    if top_k <= 0:
        return []

    query_embedding = hashed_embedding(query, EMBEDDING_DIM)
    results = []
    for chunk in _load_index():
        embedding = chunk.get("embedding") or hashed_embedding(chunk["content"], EMBEDDING_DIM)
        score = cosine_similarity(query_embedding, embedding)
        if score > 0:
            results.append(
                {
                    "content": chunk["content"],
                    "score": float(score),
                    "metadata": chunk.get("metadata", {}),
                }
            )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for result in semantic_search("hinh phat ma tuy", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
