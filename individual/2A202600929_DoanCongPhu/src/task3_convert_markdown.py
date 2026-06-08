"""
Task 3 - Convert toan bo file trong data/landing/ thanh Markdown.

Su dung MarkItDown cua Microsoft:
    https://github.com/microsoft/markitdown

Cai dat:
    pip install markitdown

Huong dan:
    1. Scan toan bo file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Luu vao data/standardized/ giu nguyen cau truc thu muc
"""

import json
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()

    for filepath in legal_dir.rglob("*"):
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            result = md.convert(str(filepath))
            relative_parent = filepath.relative_to(legal_dir).parent
            target_dir = output_dir / relative_parent
            target_dir.mkdir(parents=True, exist_ok=True)
            output_path = target_dir / f"{filepath.stem}.md"
            output_path.write_text(result.text_content, encoding="utf-8")
            print(f"  Saved: {output_path}")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in news_dir.rglob("*.json"):
        print(f"Converting: {filepath.name}")
        data = json.loads(filepath.read_text(encoding="utf-8"))
        relative_parent = filepath.relative_to(news_dir).parent
        target_dir = output_dir / relative_parent
        target_dir.mkdir(parents=True, exist_ok=True)
        output_path = target_dir / f"{filepath.stem}.md"

        header = f"# {data.get('title', 'Unknown')}\n\n"
        header += f"**Source:** {data.get('url', 'N/A')}\n"
        header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"

        content = header + data.get("content_markdown", "")
        output_path.write_text(content, encoding="utf-8")
        print(f"  Saved: {output_path}")


def convert_all():
    """Convert toan bo files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\nDone! Output tai:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
