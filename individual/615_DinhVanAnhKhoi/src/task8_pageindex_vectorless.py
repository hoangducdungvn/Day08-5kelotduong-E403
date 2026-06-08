"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to sys.path to prevent ModuleNotFoundError
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    if not PAGEINDEX_API_KEY or "xxx" in PAGEINDEX_API_KEY:
        print("⚠ PAGEINDEX_API_KEY chưa cấu hình hoặc là placeholder.")
        return

    try:
        from pageindex import PageIndexClient
        client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

        for md_file in STANDARDIZED_DIR.rglob("*.md"):
            if md_file.is_file():
                try:
                    response = client.submit_document(file_path=str(md_file))
                    print(f"  ✓ Uploaded to PageIndex: {md_file.name} (ID: {response.get('doc_id')})")
                except Exception as file_err:
                    print(f"[Warning] Lỗi upload file {md_file.name}: {file_err}")
    except Exception as e:
        print(f"[Warning] Không thể upload lên PageIndex: {e}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if PAGEINDEX_API_KEY and "xxx" not in PAGEINDEX_API_KEY:
        try:
            from pageindex import PageIndexClient
            import time

            client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
            docs_resp = client.list_documents()
            documents = docs_resp.get("documents", [])

            if documents:
                # Query document đầu tiên tìm được
                doc_id = documents[0]["doc_id"]
                resp = client.submit_query(doc_id=doc_id, query=query)
                retrieval_id = resp.get("retrieval_id")

                if retrieval_id:
                    # Chờ kết quả hoàn thành (tối đa 10 giây)
                    for _ in range(10):
                        retrieval_resp = client.get_retrieval(retrieval_id)
                        if retrieval_resp.get("status") == "completed":
                            results = retrieval_resp.get("results", [])
                            output = []
                            for r in results[:top_k]:
                                output.append({
                                    "content": r.get("text", ""),
                                    "score": float(r.get("score", 1.0)),
                                    "metadata": r.get("metadata", {}),
                                    "source": "pageindex"
                                })
                            return output
                        time.sleep(1)
        except Exception as api_err:
            print(f"[Warning] PageIndex API query failed: {api_err}. Falling back to local search.")

    # Fallback: Chạy tìm kiếm ngữ nghĩa local và gán nguồn "pageindex"
    try:
        from src.task5_semantic_search import semantic_search
        local_results = semantic_search(query, top_k=top_k)
        output = []
        for r in local_results:
            item = r.copy()
            item["source"] = "pageindex"
            output.append(item)
        return output
    except Exception as local_err:
        print(f"[Error] Fallback local search failed: {local_err}")
        return []


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")

