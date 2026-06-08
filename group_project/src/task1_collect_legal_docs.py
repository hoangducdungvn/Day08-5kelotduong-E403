"""Task 1 - Collect legal documents into data/landing/legal."""

from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


def setup_directory() -> Path:
    """Create and return the legal landing directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Directory ready: {DATA_DIR}")
    return DATA_DIR


def download_file(url: str, filename: str, timeout: int = 60) -> Path:
    """Download one legal document and return its local path."""
    setup_directory()
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    filepath = DATA_DIR / Path(filename).name
    filepath.write_bytes(response.content)
    print(f"Downloaded: {filepath}")
    return filepath


def list_legal_documents() -> list[Path]:
    """Return all supported legal source documents."""
    setup_directory()
    extensions = {".pdf", ".doc", ".docx"}
    return sorted(path for path in DATA_DIR.iterdir() if path.suffix.lower() in extensions)


if __name__ == "__main__":
    documents = list_legal_documents()
    print(f"Found {len(documents)} legal documents")
