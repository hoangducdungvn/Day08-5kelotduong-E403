import json
from pathlib import Path
from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent.parent / "data" / "landing"
STANDARDIZED_DIR = Path(__file__).parent.parent.parent / "data" / "standardized"

def convert_to_markdown():
    md = MarkItDown()
    
    # Process Legal
    legal_landing = LANDING_DIR / "legal"
    legal_std = STANDARDIZED_DIR / "legal"
    legal_std.mkdir(parents=True, exist_ok=True)
    
    if legal_landing.exists():
        for filepath in legal_landing.iterdir():
            if filepath.is_file() and filepath.suffix in (".pdf", ".docx", ".doc"):
                print(f"Converting: {filepath.name}")
                target_path = str(filepath.resolve())
                if filepath.suffix == ".doc":
                    try:
                        import win32com.client
                        word = win32com.client.Dispatch("Word.Application")
                        doc = word.Documents.Open(target_path)
                        docx_path = str(filepath.with_suffix(".docx").resolve())
                        doc.SaveAs2(docx_path, FileFormat=16)
                        doc.Close()
                        word.Quit()
                        target_path = docx_path
                        print(f"  [OK] Converted .doc to .docx using win32com")
                    except Exception as e:
                        print(f"  [X] Failed to convert .doc to .docx: {e}")
                        continue
                        
                try:
                    result = md.convert(target_path)
                    out_path = legal_std / f"{filepath.stem}.md"
                    out_path.write_text(result.text_content, encoding="utf-8")
                    print(f"  [OK] Saved: {out_path.name}")
                except Exception as e:
                    print(f"  [X] Failed to convert {filepath.name}: {e}")

    # Process News
    news_landing = LANDING_DIR / "news"
    news_std = STANDARDIZED_DIR / "news"
    news_std.mkdir(parents=True, exist_ok=True)
    
    if news_landing.exists():
        for filepath in news_landing.iterdir():
            if filepath.is_file() and filepath.suffix == ".json":
                print(f"Converting: {filepath.name}")
                try:
                    data = json.loads(filepath.read_text(encoding="utf-8"))
                    out_path = news_std / f"{filepath.stem}.md"
                    
                    content = f"# {data.get('title', 'Unknown')}\n\n"
                    content += data.get('content_markdown', '')
                    
                    out_path.write_text(content, encoding="utf-8")
                    print(f"  [OK] Saved: {out_path.name}")
                except Exception as e:
                    print(f"  [X] Failed to convert {filepath.name}: {e}")

if __name__ == "__main__":
    print("Converting to Markdown...")
    convert_to_markdown()
    print("Done converting!")
