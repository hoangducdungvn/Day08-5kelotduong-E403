import os
import re
from pathlib import Path
import shutil

LANDING_DIR = Path(__file__).parent.parent.parent / "data" / "landing"

def standardize_name(filename: str) -> str:
    name, ext = os.path.splitext(filename)
    # Loại bỏ dấu tiếng việt, ký tự đặc biệt
    # Chuyển về snake_case
    name = re.sub(r'[^a-zA-Z0-9]+', '_', name)
    name = name.strip('_').lower()
    # Fix lại phần mở rộng
    ext = ext.lower()
    return name + ext

def standardize_directory(directory: Path):
    if not directory.exists():
        return
    for filepath in directory.iterdir():
        if filepath.is_file() and filepath.name != ".gitkeep":
            new_name = standardize_name(filepath.name)
            if new_name != filepath.name:
                new_path = filepath.parent / new_name
                print(f"Renaming: {filepath.name} -> {new_name}")
                filepath.rename(new_path)

if __name__ == "__main__":
    print("Standardizing legal documents...")
    standardize_directory(LANDING_DIR / "legal")
    print("Standardizing news articles...")
    standardize_directory(LANDING_DIR / "news")
    print("Done!")
