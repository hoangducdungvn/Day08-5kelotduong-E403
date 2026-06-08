"""
Script crawl lại bài báo về nghệ sĩ Việt dính tới ma túy.
Dùng requests + trafilatura thay vì crawl4ai (vì anti-bot).
"""

import json
import time
import requests
import trafilatura
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# Danh sách URL bài báo thật, đã kiểm tra
ARTICLE_URLS = [
    # 1. Ca sĩ Chi Dân bị bắt vì ma túy (VnExpress)
    "https://vnexpress.net/ca-si-chi-dan-bi-khoi-to-4818133.html",
    # 2. Người mẫu An Tây bị bắt vì ma túy (VnExpress)
    "https://vnexpress.net/nguoi-mau-an-tay-bi-bat-tam-giam-4818896.html",
    # 3. Diễn viên Hữu Tín bị bắt vì ma túy (VietnamNet)
    "https://vietnamnet.vn/dien-vien-huu-tin-bi-phat-7-nam-6-thang-tu-2135041.html",
    # 4. Ca sĩ Châu Việt Cường bị tuyên án (Thanh Niên)
    "https://thanhnien.vn/ca-si-chau-viet-cuong-linh-13-nam-tu-185190307130017028.htm",
    # 5. Ca sĩ Long Nhật bị bắt vì ma túy 2026 (Tuổi Trẻ)
    "https://tuoitre.vn/ca-si-long-nhat-son-ngoc-minh-bi-bat-vi-to-chuc-su-dung-ma-tuy-20260520.htm",
    # 6. Diễn viên Lệ Hằng bị bắt vì mua bán ma túy (Dân trí)
    "https://dantri.com.vn/phap-luat/dien-vien-le-hang-bi-khoi-to-vi-mua-ban-ma-tuy-20230425074551297.htm",
    # 7. Tổng hợp nghệ sĩ dính tới ma túy (VietnamNet)
    "https://vietnamnet.vn/nhung-nghe-si-viet-vuong-vong-lao-ly-vi-ma-tuy-2174539.html",
    # 8. Chuyên án VN10 truy tố 227 bị can (Thanh Niên)
    "https://thanhnien.vn/chuyen-an-ma-tuy-truy-to-227-bi-can-co-chi-dan-an-tay-185260416.htm",
    # backup URLs
    "https://dantri.com.vn/phap-luat/vu-4-tiep-vien-hang-khong-buon-ma-tuy-truy-to-227-bi-can-20260416.htm",
    "https://vietnamnet.vn/ca-si-chi-dan-bi-truy-to-toi-to-chuc-su-dung-trai-phep-chat-ma-tuy-2388425.html",
]


def crawl_article(url: str) -> dict:
    """Crawl một bài báo, trả về dict có metadata + content."""
    print(f"  Fetching: {url}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
        resp.raise_for_status()

        # Kiểm tra xem có bị redirect tới trang chủ không
        if resp.url != url and len(resp.url.split("/")) <= 4:
            print(f"    ⚠ Bị redirect tới: {resp.url}")
            return None

        html = resp.text

        # Dùng trafilatura extract nội dung chính
        content = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            output_format="txt",
            favor_precision=True,
        )

        if not content or len(content) < 200:
            print(f"    ⚠ Nội dung quá ngắn ({len(content) if content else 0} chars)")
            return None

        # Extract title
        title = trafilatura.extract(html, output_format="xml")
        import re
        title_match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        title_text = title_match.group(1).strip() if title_match else "Unknown Title"
        # Cleanup title
        title_text = title_text.split(" - ")[0].split(" | ")[0].strip()

        return {
            "url": url,
            "title": title_text,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": content,
        }
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return None


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for url in ARTICLE_URLS:
        article = crawl_article(url)
        if article:
            results.append(article)
            print(f"    ✓ OK ({len(article['content_markdown'])} chars)")
        time.sleep(2)  # Tránh bị chặn

        if len(results) >= 10:
            break

    print(f"\n=== Crawl được {len(results)} bài ===")

    if len(results) < 5:
        print("⚠ Không đủ 5 bài, sẽ dùng fallback...")

    # Lưu file
    for i, article in enumerate(results, 1):
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  Saved: {filepath.name}")


if __name__ == "__main__":
    main()
