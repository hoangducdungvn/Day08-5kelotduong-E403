"""Task 4 - Load, chunk, embed and index standardized Markdown documents."""

import json
from pathlib import Path

try:
    from .rag_utils import chunk_documents_offline, hashed_embedding, load_markdown_documents
except ImportError:
    from rag_utils import chunk_documents_offline, hashed_embedding, load_markdown_documents

PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
INDEX_PATH = PROJECT_DIR / "data" / "vector_index.json"

# Legal documents are split by Chương/Điều/Khoản metadata; news still uses the
# paragraph-aware fallback. This reduces retrieval noise for statute lookups.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "legal_aware_article_clause"

# A dependency-free hashed embedding keeps the demo runnable offline. It can be
# replaced by BAAI/bge-m3 without changing the downstream result structure.
EMBEDDING_MODEL = "local-hashed-token-embedding"
EMBEDDING_DIM = 384
VECTOR_STORE = "local_json"


def load_documents() -> list[dict]:
    """Read every Markdown document from data/standardized."""
    return load_markdown_documents(STANDARDIZED_DIR)


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split legal docs by provisions and non-legal docs by paragraphs."""
    return chunk_documents_offline(documents, CHUNK_SIZE, CHUNK_OVERLAP)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Attach a normalized local embedding to each chunk."""
    return [
        {**chunk, "embedding": hashed_embedding(chunk["content"], EMBEDDING_DIM)}
        for chunk in chunks
    ]


def index_to_vectorstore(chunks: list[dict]) -> Path:
    """Persist the local vector index as JSON and return its path."""
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    return INDEX_PATH


def run_pipeline() -> Path:
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    docs = load_documents()
    chunks = embed_chunks(chunk_documents(docs))
    index_path = index_to_vectorstore(chunks)
    print(f"Loaded {len(docs)} documents; indexed {len(chunks)} chunks to {index_path}")
    return index_path


if __name__ == "__main__":
    run_pipeline()
