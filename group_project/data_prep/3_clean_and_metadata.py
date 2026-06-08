import re
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent.parent / "data" / "standardized"
PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"

def clean_text(text: str) -> str:
    # Xóa các dòng trống liên tiếp (nhiều hơn 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Xóa ký tự zero-width space
    text = text.replace('\u200b', '')
    # Trim
    return text.strip()

def extract_metadata_legal(text: str, filename: str) -> dict:
    title = filename.replace('.md', '')
    # Thử tìm tiêu đề trong văn bản (Heading đầu tiên)
    match = re.search(r'^#\s+(.+)', text, flags=re.MULTILINE)
    if match:
        title = match.group(1).strip().replace('**', '')
        
    category = "Luật" if "luat" in filename.lower() else "Nghị định" if "nghi" in filename.lower() else "Văn bản pháp luật"
    return {
        "title": f'"{title}"',
        "source": f'"{filename}"',
        "category": f'"{category}"'
    }

def extract_metadata_news(text: str, filename: str) -> dict:
    title = filename.replace('.md', '')
    match = re.search(r'^#\s+(.+)', text, flags=re.MULTILINE)
    if match:
        title = match.group(1).strip().replace('**', '')
        
    return {
        "title": f'"{title}"',
        "source": f'"{filename}"',
        "category": '"Tin tức"'
    }

def process_files(category_folder: str, metadata_extractor):
    in_dir = STANDARDIZED_DIR / category_folder
    out_dir = PROCESSED_DIR / category_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not in_dir.exists():
        print(f"Skipping {category_folder}, directory does not exist.")
        return

    for filepath in in_dir.iterdir():
        if filepath.suffix == ".md":
            print(f"Processing: {filepath.name}")
            raw_text = filepath.read_text(encoding="utf-8")
            
            # Làm sạch văn bản
            cleaned_text = clean_text(raw_text)
            
            # Trích xuất metadata
            metadata = metadata_extractor(cleaned_text, filepath.name)
            
            # Tạo frontmatter (YAML)
            frontmatter = "---\n"
            for k, v in metadata.items():
                frontmatter += f"{k}: {v}\n"
            frontmatter += "---\n\n"
            
            # Ghi ra file
            final_text = frontmatter + cleaned_text
            out_path = out_dir / filepath.name
            out_path.write_text(final_text, encoding="utf-8")
            print(f"  [OK] Saved to processed")

if __name__ == "__main__":
    print("Cleaning Legal docs and adding metadata...")
    process_files("legal", extract_metadata_legal)
    print("Cleaning News articles and adding metadata...")
    process_files("news", extract_metadata_news)
    print("Done!")
