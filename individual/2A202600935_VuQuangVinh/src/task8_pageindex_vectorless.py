"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
import time
import json
import re
from pathlib import Path

import requests
from dotenv import load_dotenv
from pageindex import PageIndexClient
from markdown_pdf import MarkdownPdf, Section

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
DOC_IDS_PATH = Path(__file__).parent.parent / "data" / "pageindex_doc_ids.json"
BASE_URL = "https://api.pageindex.ai"

# Khởi tạo SDK Client của PageIndex
if PAGEINDEX_API_KEY:
    pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)


# =============================================================================
# Upload helpers
# =============================================================================

def upload_documents() -> dict[str, str]:
    """
    Tự động convert markdown sang PDF và upload lên PageIndex bằng SDK.
    """
    doc_ids: dict[str, str] = {}

    if DOC_IDS_PATH.exists():
        with open(DOC_IDS_PATH) as f:
            doc_ids = json.load(f)
        print(f"  Loaded {len(doc_ids)} existing entries từ {DOC_IDS_PATH}")

    # Vẫn quét các file .md trong thư mục
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        filename = md_file.name
        pdf_filename = filename.replace(".md", ".pdf")
        pdf_path = md_file.parent / pdf_filename

        if filename in doc_ids:
            print(f"  ⏭ Skip (already uploaded): {filename}")
            continue

        print(f"  🔄 Đang chuyển đổi {filename} sang PDF...")
        try:
            # Đọc nội dung Markdown
            content = md_file.read_text(encoding="utf-8")

            content = re.sub(r'^\[\^[^\]]*\]:.*$', '', content, flags=re.MULTILINE)

            content = re.sub(r'\[\^[^\]]*\]', '', content)
            
            # Tạo file PDF
            pdf = MarkdownPdf(toc_level=0)
            pdf.add_section(Section(content))
            pdf.save(str(pdf_path))

            print(f"  ⬆️ Uploading: {pdf_filename}...")
            
            # Upload file PDF vừa tạo
            doc_info = pi_client.submit_document(str(pdf_path))
            doc_id = doc_info.get("doc_id")
            
            if doc_id:
                print(f"  ✓ Uploaded success: {pdf_filename} → {doc_id}")
                doc_ids[filename] = doc_id # Lưu tên gốc .md đi kèm ID để dễ track
            else:
                print(f"  ⚠ Failed to get doc_id for {pdf_filename}: {doc_info}")
                
        except Exception as e:
            print(f"  ⚠ Lỗi khi xử lý/upload {filename}: {e}")

    # Lưu lại danh sách IDs
    os.makedirs(DOC_IDS_PATH.parent, exist_ok=True)
    with open(DOC_IDS_PATH, "w") as f:
        json.dump(doc_ids, f, indent=2)

    return doc_ids


# =============================================================================
# Retrieval helpers
# =============================================================================

def _retrieve_from_doc(doc_id: str, query: str, top_k: int) -> str | None:
    """Submit câu hỏi thông qua Chat API mới của PageIndex."""
    print(f"      ➔ Đang gọi Chat AI phân tích tài liệu (có thể mất 10-20s)...")
    
    # Đổi endpoint sang chat/completions
    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={
            "api_key": PAGEINDEX_API_KEY,                  # Header cũ
            "Authorization": f"Bearer {PAGEINDEX_API_KEY}" # Chuẩn OpenAI
        },
        json={
            "messages": [{"role": "user", "content": query}],
            "doc_id": doc_id
            # Bỏ top_k đi vì Chat API của họ tự động lo việc trích xuất
        },
    )
    
    if not resp.ok:
        print(f"      [Lỗi Gọi API] {resp.status_code}: {resp.text[:200]}")
        return None
        
    data = resp.json()
    
    try:
        # Lấy thẳng câu trả lời theo chuẩn cấu trúc Chat Completion của OpenAI
        ans = data["choices"][0]["message"]["content"]
        return str(ans)
    except (KeyError, IndexError):
        print(f"      [LỖI CẤU TRÚC DỮ LIỆU] {data}")
        return None

def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    if not DOC_IDS_PATH.exists():
        raise RuntimeError("Chưa upload documents. Chạy upload_documents() trước.")

    with open(DOC_IDS_PATH) as f:
        doc_ids: dict[str, str] = json.load(f)

    if not doc_ids:
        return []

    results = []
    for i, (filename, doc_id) in enumerate(list(doc_ids.items())[:top_k]):
        print(f"  🔎 Đang tìm trong: {filename}...")
        try:
            content = _retrieve_from_doc(doc_id, query, top_k=3)
            if content:
                results.append({
                    "content": content,
                    "score": round(1.0 - i * 0.05, 4),
                    "metadata": {"filename": filename, "doc_id": doc_id},
                    "source": "pageindex",
                })
            else:
                print(f"     ➔ Không có kết quả hoặc AI không tìm thấy đáp án trong file này.")
        except Exception as e:
            print(f"  ⚠ Error querying {filename}: {e}")
            continue

    return results

if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
        print("  Lấy API key tại: https://dash.pageindex.ai")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")