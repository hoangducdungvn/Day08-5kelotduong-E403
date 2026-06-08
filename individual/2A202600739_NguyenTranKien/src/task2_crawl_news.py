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
    "https://tuoitre.vn/nghi-an-ca-si-chi-dan-va-nguoi-mau-an-tay-bi-bat-vi-ma-tuy-20241110091809051.htm",
    "https://thanhnien.vn/ca-si-chi-dan-nguoi-mau-an-tay-bi-dieu-tra-nghi-lien-quan-ma-tuy-185241110093856182.htm",
    "https://dantri.com.vn/phap-luat/ca-si-chi-dan-nguoi-mau-an-tay-bi-dieu-tra-lien-quan-ma-tuy-20241110095810245.htm",
    "https://vietnamnet.vn/dien-vien-huu-tin-bi-bat-vi-ma-tuy-doi-dien-muc-an-nao-2029707.html",
    "https://vietnamnet.vn/nhung-nghe-si-viet-danh-mat-su-nghiep-vi-ma-tuy-2342080.html",
    "https://tuoitre.vn/ca-si-chau-viet-cuong-bi-tuyen-an-13-nam-tu-giam-20190307125300676.htm",
    "https://thanhnien.vn/nam-dien-vien-huu-tin-bi-tuyen-phat-7-nam-6-thang-tu-vi-su-dung-ma-tuy-185230428135805175.htm",
    "https://vnexpress.net/dien-vien-le-hang-bi-bat-vi-mua-ban-ma-tuy-4597034.html",
    "https://dantri.com.vn/phap-luat/nhung-nghe-si-viet-dinh-dang-den-ma-tuy-20230425121200001.htm",
    "https://tuoitre.vn/khoi-to-bat-tam-giam-nguoi-mau-an-tay-ca-si-chi-dan-vi-ma-tuy-20241114135503253.htm"
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
    import trafilatura
    
    # Dùng crawl4ai để bypass anti-bot
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        
        title = "Unknown Title"
        if hasattr(result, "metadata") and isinstance(result.metadata, dict):
            title = result.metadata.get("title", title)
        elif hasattr(result, "title") and result.title:
            title = result.title
            
        content_markdown = ""
        if hasattr(result, "html") and result.html:
            extracted = trafilatura.extract(result.html, include_comments=False, include_tables=False, output_format="markdown")
            if extracted:
                content_markdown = extracted
        
        # Nếu trafilatura thất bại, dùng markdown mặc định của crawl4ai
        if not content_markdown and hasattr(result, "markdown"):
            content_markdown = result.markdown
            
        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": content_markdown or "",
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
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
