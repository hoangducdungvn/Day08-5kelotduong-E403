"""Task 2 - Crawl news articles and save structured JSON."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://tuoitre.vn/ca-si-miu-le-bi-bat-qua-tang-su-dung-ma-tuy-o-hai-phong-20260511172700149.htm",
    "https://vietnamnet.vn/de-nghi-truy-to-ca-si-chi-dan-cung-anh-trai-vi-to-chuc-su-dung-ma-tuy-2434484.html",
    "https://baovanhoa.vn/giai-tri/long-nhat-bi-bat-ma-tuy-dang-len-sau-vao-showbiz-viet-230095.html",
    "https://tintuconline.com.vn/truy-bat-vu-xuan-b-duong-tinh-ma-tuy-lai-o-to-chay-chen-ep-tren-cao-toc-5057624.html",
    "https://nld.com.vn/showbiz-viet-nhung-nghe-si-gay-soc-vi-be-boi-ma-tuy-196250725113547841.htm",
]


def setup_directory() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


async def crawl_article(url: str) -> dict:
    """Crawl one article using Crawl4AI and return normalized fields."""
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

    if getattr(result, "success", True) is False:
        raise RuntimeError(getattr(result, "error_message", "Crawl failed"))

    metadata = getattr(result, "metadata", None) or {}
    markdown = getattr(result, "markdown", "") or ""
    if hasattr(markdown, "raw_markdown"):
        markdown = markdown.raw_markdown

    return {
        "url": url,
        "title": metadata.get("title", "Unknown"),
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": str(markdown),
    }


async def crawl_all() -> list[Path]:
    """Crawl every configured article and return paths written."""
    setup_directory()
    written = []

    for index, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{index}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
            filename = f"article_{index:02d}.json"
        except Exception as exc:
            article = {
                "url": url,
                "title": "CRAWL_FAILED",
                "date_crawled": datetime.now().isoformat(),
                "content_markdown": "",
                "error": str(exc),
            }
            filename = f"article_{index:02d}_failed.json"

        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(filepath)
        print(f"Saved: {filepath}")

    return written


if __name__ == "__main__":
    asyncio.run(crawl_all())
