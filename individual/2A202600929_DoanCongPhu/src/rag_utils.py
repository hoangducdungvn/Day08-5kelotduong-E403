"""Small offline helpers shared by the task modules.

The course tasks mention external services such as Weaviate, PageIndex and LLM
APIs. These helpers keep the personal pipeline runnable locally for tests and
demo, while the public task functions can still be swapped to managed services.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"


def tokenize(text: str) -> list[str]:
    """Tokenize Vietnamese-ish text with a dependency-free Unicode regex."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def load_markdown_documents(base_dir: Path = STANDARDIZED_DIR) -> list[dict]:
    documents = []
    if not base_dir.exists():
        return documents

    for md_file in sorted(base_dir.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            continue
        relative_path = md_file.relative_to(base_dir)
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "path": str(relative_path).replace("\\", "/"),
                    "type": relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown",
                },
            }
        )
    return documents


def simple_chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            step = max(1, chunk_size - chunk_overlap)
            for start in range(0, len(paragraph), step):
                piece = paragraph[start : start + chunk_size].strip()
                if piece:
                    chunks.append(piece)
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            overlap_text = current[-chunk_overlap:].strip() if chunk_overlap and current else ""
            current = f"{overlap_text}\n\n{paragraph}".strip() if overlap_text else paragraph

    if current:
        chunks.append(current.strip())

    return chunks


def chunk_documents_offline(
    documents: list[dict], chunk_size: int = 500, chunk_overlap: int = 50
) -> list[dict]:
    chunks = []
    for doc in documents:
        for index, content in enumerate(simple_chunk_text(doc["content"], chunk_size, chunk_overlap)):
            chunks.append(
                {
                    "content": content,
                    "metadata": {
                        **doc.get("metadata", {}),
                        "chunk_index": index,
                    },
                }
            )
    return chunks


def hashed_embedding(text: str, dim: int = 384) -> list[float]:
    vector = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    left_norm = math.sqrt(sum(value * value for value in left[:size]))
    right_norm = math.sqrt(sum(value * value for value in right[:size]))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(left[i] * right[i] for i in range(size)) / (left_norm * right_norm)


def default_chunks(chunk_size: int = 500, chunk_overlap: int = 50) -> list[dict]:
    return chunk_documents_offline(load_markdown_documents(), chunk_size, chunk_overlap)

