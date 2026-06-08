"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb
"""

import json
import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from sentence_transformers import SentenceTransformer

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
CHUNK_INDEX_PATH = Path(__file__).parent.parent / "data" / "chunk_index.json"
COLLECTION_NAME = "drug_law_docs"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn trong comment
# =============================================================================

# Chunking: MarkdownHeaderTextSplitter → RecursiveCharacterTextSplitter (2-pass)
# Lý do: data là markdown có heading rõ (Điều/Khoản trong luật, tiêu đề bài báo)
# → Pass 1 split theo header để giữ ngữ cảnh pháp lý theo từng điều khoản
# → Pass 2 recursive để đảm bảo chunk không vượt quá chunk_size
CHUNK_SIZE = 512        # 512 chars: đủ context cho 1 điều khoản, không quá dài cho embedding
CHUNK_OVERLAP = 64      # ~12% overlap: tránh mất context ở ranh giới chunk
CHUNKING_METHOD = "markdown_header+recursive"

# Embedding: all-MiniLM-L6-v2
# Lý do thực tế: bge-m3 (2.3GB) vượt quá dung lượng disk môi trường lab.
# all-MiniLM-L6-v2 (90MB) nhẹ hơn 25x, đủ cho semantic search,
# dễ swap sang bge-m3 khi deploy production chỉ bằng 1 dòng config.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Vector store: ChromaDB
# Lý do: local, zero config, không cần Docker → phù hợp môi trường lab
# Task 6 (BM25) handle lexical search riêng nên không cần hybrid built-in của Weaviate
VECTOR_STORE = "chromadb"

HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "source_path": str(md_file.relative_to(STANDARDIZED_DIR)),
                "type": doc_type,
            }
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn: MarkdownHeader → Recursive (2-pass).

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,  # giữ header trong chunk để LLM có context khi generate
    )
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )

    chunks = []
    for doc in documents:
        # Pass 1: split theo markdown header
        header_chunks = header_splitter.split_text(doc["content"])

        # Pass 2: nếu chunk vẫn quá dài → recursive split tiếp
        chunk_index = 0
        for hc in header_chunks:
            text = hc.page_content
            merged_meta = {**doc["metadata"], **hc.metadata}

            if len(text) > CHUNK_SIZE:
                sub_texts = recursive_splitter.split_text(text)
                for sub in sub_texts:
                    chunks.append({
                        "content": sub,
                        "metadata": {**merged_meta, "chunk_index": chunk_index},
                    })
                    chunk_index += 1
            else:
                chunks.append({
                    "content": text,
                    "metadata": {**merged_meta, "chunk_index": chunk_index},
                })
                chunk_index += 1

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng BAAI/bge-m3.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    print(f"  Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["content"] for c in chunks]

    # normalize_embeddings=True để dùng cosine similarity
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    )

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()

    return chunks


def index_to_vectorstore(chunks: list[dict]) -> None:
    """
    Lưu chunks vào ChromaDB (local persistent).
    Đồng thời lưu chunk_index.json để Task 6 (BM25) dùng lại.
    """
    # ── ChromaDB ──────────────────────────────────────────────────────────────
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    # Xoá collection cũ nếu re-index
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Insert theo batch
    BATCH_SIZE = 64
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        collection.add(
            ids=[f"chunk_{i + j}" for j in range(len(batch))],
            documents=[c["content"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
            metadatas=[
                # ChromaDB chỉ nhận metadata value là str/int/float/bool
                {k: str(v) for k, v in c["metadata"].items()}
                for c in batch
            ],
        )

    print(f"  ChromaDB: {collection.count()} vectors @ {CHROMA_DIR}")

    # ── Lưu chunk_index.json cho Task 6 & 7 ──────────────────────────────────
    os.makedirs(CHUNK_INDEX_PATH.parent, exist_ok=True)
    chunks_to_save = [
        {"content": c["content"], "metadata": c["metadata"]}
        for c in chunks
    ]
    with open(CHUNK_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks_to_save, f, ensure_ascii=False, indent=2)

    print(f"  chunk_index.json: {len(chunks_to_save)} chunks @ {CHUNK_INDEX_PATH}")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()