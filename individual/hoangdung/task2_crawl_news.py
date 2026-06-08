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


ARTICLE_URLS = [
    "https://tuoitre.vn/ca-si-chi-dan-nguoi-mau-an-tay-tiktoker-co-dong-bi-dieu-tra-vi-lien-quan-ma-tuy-20241109191334927.htm",
    "https://thanhnien.vn/ca-si-chi-dan-nguoi-mau-an-tay-bi-dieu-tra-vi-lien-quan-ma-tuy-185241109210925916.htm",
    "https://vnexpress.net/ca-si-chau-viet-cuong-bi-tuyen-phat-13-nam-tu-3893339.html",
    "https://tuoitre.vn/toa-giam-an-cho-chau-viet-cuong-con-11-nam-tu-20190808162237976.htm",
    "https://thanhnien.vn/dien-vien-huu-tin-bi-tuyen-phat-7-nam-6-thang-tu-vi-to-chuc-su-dung-ma-tuy-185230428131558238.htm",
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
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        
        # Extract title safely
        title = "Unknown"
        if hasattr(result, 'metadata') and isinstance(result.metadata, dict):
            title = result.metadata.get("title", "Unknown")
        elif hasattr(result, 'title') and result.title:
            title = result.title

        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": getattr(result, "markdown", ""),
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
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2))
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
