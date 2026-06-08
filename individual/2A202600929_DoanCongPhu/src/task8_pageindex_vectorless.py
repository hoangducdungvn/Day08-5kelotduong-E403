"""Task 8 - PageIndex-compatible vectorless fallback search."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

try:
    from .rag_utils import default_chunks, tokenize
except ImportError:
    from rag_utils import default_chunks, tokenize

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
PAGEINDEX_LOCAL_PATH = PROJECT_DIR / "data" / "pageindex_local.json"


def upload_documents() -> Path:
    """Store documents in a local PageIndex-like JSON file for offline demos."""
    chunks = default_chunks()
    PAGEINDEX_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAGEINDEX_LOCAL_PATH.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    print(f"Uploaded {len(chunks)} local PageIndex chunks to {PAGEINDEX_LOCAL_PATH}")
    return PAGEINDEX_LOCAL_PATH


def _load_pageindex_docs() -> list[dict]:
    if PAGEINDEX_LOCAL_PATH.exists():
        return json.loads(PAGEINDEX_LOCAL_PATH.read_text(encoding="utf-8"))
    return default_chunks()


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Vectorless retrieval using token coverage and document structure."""
    if top_k <= 0:
        return []

    query_tokens = set(tokenize(query))
    results = []
    for doc in _load_pageindex_docs():
        content_tokens = tokenize(doc["content"])
        if not content_tokens:
            continue
        token_set = set(content_tokens)
        overlap = len(query_tokens & token_set)
        if query_tokens:
            score = overlap / len(query_tokens)
        else:
            score = 0.0
        score += min(overlap, 5) * 0.01
        if score <= 0:
            continue
        results.append(
            {
                "content": doc["content"],
                "score": float(score),
                "metadata": doc.get("metadata", {}),
                "source": "pageindex",
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    upload_documents()
    for result in pageindex_search("ma tuy", top_k=3):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
