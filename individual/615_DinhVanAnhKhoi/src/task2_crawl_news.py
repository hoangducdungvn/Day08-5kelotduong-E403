"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# TODO: Điền danh sách URL bài báo cần crawl
ARTICLE_URLS = [
    "https://vietnamnet.vn/loat-ca-si-dinh-chat-cam-ma-tuy-pha-huy-nao-bo-nguoi-tre-ra-sao-2518285.html",
    "https://baovanhoa.vn/giai-tri/ma-tuy-va-nhung-cu-nga-ngua-cua-showbiz-viet-230477.html",
    "https://baolaocai.vn/bao-dong-tinh-trang-nghe-si-dung-ma-tuy-va-nhung-he-luy-voi-xa-hoi-post900028.html",
    "https://tuoitre.vn/bat-ca-si-long-nhat-va-ca-si-son-ngoc-minh-vi-lien-quan-ma-tuy-20260520082138943.html",
    "https://vnexpress.net/ma-tuy-trong-loi-song-showbiz-5074606.html"
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    # Thử dùng crawl4ai trước
    try:
        from crawl4ai import AsyncWebCrawler
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            if result and result.markdown:
                title = result.metadata.get("title") or "Unknown"
                return {
                    "url": url,
                    "title": title,
                    "date_crawled": datetime.now().isoformat(),
                    "content_markdown": result.markdown,
                }
    except Exception as e:
        print(f"  [crawl4ai info] Crawl4AI failed or not installed, switching to fallback. Error: {e}")

    # Fallback: requests + BeautifulSoup
    import requests
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, lambda: requests.get(url, headers=headers, timeout=15, verify=False))
    response.raise_for_status()
    response.encoding = response.apparent_encoding

    soup = BeautifulSoup(response.text, "html.parser")

    title = ""
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text().strip()

    if not title:
        meta_title = soup.find("meta", property="og:title")
        if meta_title:
            title = meta_title.get("content", "").strip()

    if not title:
        title = soup.title.string.strip() if soup.title else "Unknown Title"

    # Xóa script, style, iframe, header, footer để giữ nội dung sạch
    for tag in soup(["script", "style", "iframe", "header", "footer", "nav"]):
        tag.decompose()

    # Tìm main content
    content_div = None
    for selector in ["article", ".fck_detail", ".content-detail-text", ".detail-content", ".post-content", ".detail__content", ".entry-content"]:
        found = soup.select_one(selector)
        if found:
            content_div = found
            break

    if not content_div:
        content_div = soup.body if soup.body else soup

    markdown_text = ""
    try:
        import markdownify
        markdown_text = markdownify.markdownify(str(content_div), heading_style="ATX")
    except ImportError:
        paragraphs = []
        for p in content_div.find_all(["p", "h1", "h2", "h3", "h4"]):
            text = p.get_text().strip()
            if text:
                if p.name.startswith("h"):
                    level = p.name[1]
                    paragraphs.append(f"{'#' * int(level)} {text}\n")
                else:
                    paragraphs.append(f"{text}\n")
        markdown_text = "\n".join(paragraphs)

    if len(markdown_text.strip()) < 100:
        markdown_text = content_div.get_text(separator="\n").strip()

    if len(markdown_text.strip()) < 600:
        paragraphs = []
        for p in soup.find_all("p"):
            text = p.get_text().strip()
            if len(text) > 30:
                paragraphs.append(text)
        if paragraphs:
            markdown_text = "\n\n".join(paragraphs)

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": markdown_text,
    }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [OK] Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("[WARNING] Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
