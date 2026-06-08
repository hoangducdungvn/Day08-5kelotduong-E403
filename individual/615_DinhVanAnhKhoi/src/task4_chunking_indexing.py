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
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# TODO: Chọn chunking strategy và giải thích vì sao
CHUNK_SIZE = 500        # Chọn 500 ký tự để giữ các câu văn bản pháp lý/tin tức có ý nghĩa trọn vẹn, không quá dài làm loãng vector.
CHUNK_OVERLAP = 50      # Chọn 50 ký tự gối đầu để đảm bảo thông tin ngữ nghĩa tại biên của các chunk không bị đứt đoạn.
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# TODO: Chọn embedding model và giải thích
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Chọn model nhẹ (all-MiniLM-L6-v2) để tránh lỗi tràn bộ nhớ (Out of Memory/Memory Allocation Failed) trên môi trường thử nghiệm, nhưng vẫn đảm bảo khả năng tìm kiếm ngữ nghĩa tốt.
EMBEDDING_DIM = 384

# TODO: Chọn vector store
VECTOR_STORE = "weaviate"  # "weaviate" | "chromadb" | "faiss" (Sử dụng lưu trữ pickle local làm fallback để chạy độc lập không phụ thuộc service)


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
    if not STANDARDIZED_DIR.exists():
        print(f"Warning: STANDARDIZED_DIR {STANDARDIZED_DIR} does not exist.")
        return documents

    for filepath in STANDARDIZED_DIR.rglob("*.md"):
        if filepath.is_file():
            try:
                content = filepath.read_text(encoding="utf-8")
                doc_type = "legal" if "legal" in filepath.parts else "news"
                documents.append({
                    "content": content,
                    "metadata": {
                        "source": filepath.name,
                        "type": doc_type
                    }
                })
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    chunks = []
    
    # Thử dùng langchain_text_splitters trước
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        for doc in documents:
            splits = splitter.split_text(doc["content"])
            for i, chunk_text in enumerate(splits):
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": i
                    }
                })
        return chunks
    except ImportError:
        # Custom fallback recursive character splitter (không phụ thuộc langchain)
        print("  [langchain info] langchain-text-splitters not installed, using custom fallback recursive splitter.")
        
        def split_text_recursive(text: str, chunk_size: int, chunk_overlap: int, separators: list[str]) -> list[str]:
            if not separators:
                result_chunks = []
                i = 0
                while i < len(text):
                    result_chunks.append(text[i:i+chunk_size])
                    i += chunk_size - chunk_overlap
                return result_chunks
                
            separator = separators[0]
            next_separators = separators[1:]
            
            parts = text.split(separator) if separator != "" else list(text)
            
            result_chunks = []
            current_chunk = []
            current_len = 0
            
            for part in parts:
                part_len = len(part)
                if part_len > chunk_size:
                    if current_chunk:
                        result_chunks.append(separator.join(current_chunk))
                        current_chunk = []
                        current_len = 0
                    sub_splits = split_text_recursive(part, chunk_size, chunk_overlap, next_separators)
                    result_chunks.extend(sub_splits)
                else:
                    join_len = len(separator) if current_chunk else 0
                    if current_len + join_len + part_len > chunk_size:
                        result_chunks.append(separator.join(current_chunk))
                        
                        overlap_chunk = []
                        overlap_len = 0
                        for p in reversed(current_chunk):
                            p_join_len = len(separator) if overlap_chunk else 0
                            if overlap_len + p_join_len + len(p) <= chunk_overlap:
                                overlap_chunk.insert(0, p)
                                overlap_len += p_join_len + len(p)
                            else:
                                break
                        current_chunk = overlap_chunk
                        current_len = overlap_len
                        join_len = len(separator) if current_chunk else 0
                        
                    current_chunk.append(part)
                    current_len += join_len + part_len
                    
            if current_chunk:
                result_chunks.append(separator.join(current_chunk))
                
            return [c for c in result_chunks if c.strip()]

        separators = ["\n\n", "\n", ". ", " ", ""]
        for doc in documents:
            splits = split_text_recursive(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP, separators)
            for i, chunk_text in enumerate(splits):
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": i
                    }
                })
        return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["content"] for c in chunks]
    print(f"Encoding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
        
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store.
    """
    import pickle
    
    # Lưu vào local pickle làm fallback để đảm bảo chạy độc lập hoàn hảo
    vectorstore_path = Path(__file__).parent.parent / "data" / "vectorstore.pkl"
    vectorstore_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vectorstore_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"  [Local Pickle Store] Indexed {len(chunks)} chunks to: {vectorstore_path}")

    # Đồng thời thử kết nối Weaviate nếu Weaviate có sẵn
    try:
        import weaviate
        from weaviate.classes.config import Configure, Property, DataType
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        weaviate_url = os.getenv("WEAVIATE_URL")
        weaviate_key = os.getenv("WEAVIATE_API_KEY")
        
        if weaviate_url and "xxx" not in weaviate_url:
            print(f"Connecting to Weaviate Cloud: {weaviate_url}")
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=weaviate_url,
                auth_credentials=weaviate.auth.AuthApiKey(weaviate_key)
            )
        else:
            print("Connecting to local Weaviate...")
            client = weaviate.connect_to_local()
            
        # Kiểm tra và tạo collection
        try:
            client.collections.delete("DrugLawDocs")
        except Exception:
            pass
            
        collection = client.collections.create(
            name="DrugLawDocs",
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="doc_type", data_type=DataType.TEXT),
            ]
        )
        
        # Insert
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                batch.add_object(
                    properties={
                        "content": chunk["content"],
                        "source": chunk["metadata"]["source"],
                        "doc_type": chunk["metadata"]["type"]
                    },
                    vector=chunk["embedding"]
                )
        print("[OK] Successfully indexed to Weaviate")
        client.close()
    except Exception as e:
        print(f"  [Weaviate Info] Weaviate indexing skipped or failed (falling back entirely to local pickle): {e}")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n[OK] Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"[OK] Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"[OK] Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("[OK] Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
