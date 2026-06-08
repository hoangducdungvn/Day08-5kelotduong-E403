"""Small offline helpers shared by the task modules.

The course tasks mention external services such as Weaviate, PageIndex and LLM
APIs. These helpers keep the personal pipeline runnable locally for tests and
demo, while the public task functions can still be swapped to managed services.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"


def tokenize(text: str) -> list[str]:
    """Tokenize Vietnamese-ish text with a dependency-free Unicode regex."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def normalize_for_match(text: str) -> str:
    """Fold Vietnamese accents for robust legal-reference matching."""
    normalized = unicodedata.normalize("NFD", text or "").lower()
    normalized = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    return normalized.replace("đ", "d")


def strip_markdown_markup(text: str) -> str:
    """Remove heading/emphasis markers while keeping legal numbering intact."""
    text = re.sub(r"^#{1,6}\s*", "", text or "").strip()
    return text.strip("*_`~ ")


def extract_legal_references(text: str) -> dict[str, set[str]]:
    """Extract article, clause, chapter, and legal-document references."""
    raw = text or ""
    folded = normalize_for_match(raw)
    law_ids = {
        match.lower()
        for match in re.findall(r"\b\d{1,3}/\d{4}/[A-Za-zĐđ-]+(?:/[A-Za-zĐđ-]+)?\b", raw)
    }
    law_ids.update(
        match.lower()
        for match in re.findall(r"\b\d{1,3}/\d{4}/[a-z-]+(?:/[a-z-]+)?\b", folded)
    )

    if "bo luat hinh su" in folded or re.search(r"\bblhs\s*2015\b", folded):
        law_ids.add("bo-luat-hinh-su-2015")
    if "luat phong chong ma tuy" in folded or "luat phong, chong ma tuy" in folded:
        law_ids.add("luat-phong-chong-ma-tuy")
    if "nghi dinh 105" in folded or "105/2021/nd-cp" in folded:
        law_ids.add("105/2021/nd-cp")

    return {
        "article_numbers": set(re.findall(r"\bdieu\s+(\d+[a-z]?)\b", folded)),
        "clause_numbers": set(re.findall(r"\bkhoan\s+(\d+[a-z]?)\b", folded)),
        "chapter_numbers": set(re.findall(r"\bchuong\s+([ivxlcdm]+|\d+)\b", folded)),
        "law_ids": law_ids,
    }


def legal_reference_match_score(query: str, metadata: dict, content: str = "") -> float:
    """Score exact legal-reference matches using metadata before text fallback.

    Metadata matches are intentionally stronger than content-only matches because
    they identify the provision itself, not a passing mention in commentary.
    """
    refs = extract_legal_references(query)
    if not any(refs.values()):
        return 0.0

    metadata = metadata or {}
    content_refs = extract_legal_references(content)
    score = 0.0

    article = str(metadata.get("article_number", "")).lower()
    clause = str(metadata.get("clause_number", "")).lower()
    clauses = {str(value).lower() for value in metadata.get("clause_numbers", [])}
    if clause:
        clauses.add(clause)
    chapter = str(metadata.get("chapter_number", "")).lower()
    document_ids = {str(value).lower() for value in metadata.get("document_ids", [])}

    if refs["article_numbers"]:
        if article in refs["article_numbers"]:
            score += 6.0
        elif refs["article_numbers"] & content_refs["article_numbers"]:
            score += 1.0

    if refs["clause_numbers"]:
        if clauses & refs["clause_numbers"]:
            score += 3.0
        elif refs["clause_numbers"] & content_refs["clause_numbers"]:
            score += 0.75

    if refs["chapter_numbers"]:
        if chapter.lower() in {item.lower() for item in refs["chapter_numbers"]}:
            score += 2.0
        elif refs["chapter_numbers"] & content_refs["chapter_numbers"]:
            score += 0.5

    if refs["law_ids"]:
        if document_ids & refs["law_ids"]:
            score += 3.0
        elif refs["law_ids"] & content_refs["law_ids"]:
            score += 1.0

    return score


def _document_identifiers(content: str, metadata: dict) -> set[str]:
    identifiers: set[str] = set()
    path_text = f"{metadata.get('source', '')} {metadata.get('path', '')}"
    header_text = "\n".join((content or "").splitlines()[:25])
    header_ids = set(extract_legal_references(header_text)["law_ids"])
    header_ids.discard("luat-phong-chong-ma-tuy")
    identifiers.update(header_ids)
    identifiers.update(extract_legal_references(path_text)["law_ids"])
    folded_path = normalize_for_match(path_text).replace("_", "-")
    folded_header = normalize_for_match(header_text)
    if "luat-73-2021-qh14" in folded_path or "73-2021-qh14" in folded_path or "luat so: 73/2021/qh14" in folded_header:
        identifiers.update({"73/2021/qh14", "luat-phong-chong-ma-tuy"})
    if "105-2021-ndcp" in folded_path or "105-2021-nd-cp" in folded_path:
        identifiers.add("105/2021/nd-cp")
    if "hinh-su" in folded_path or "2018" in folded_path:
        identifiers.add("bo-luat-hinh-su-2015")
    return identifiers


def _article_heading(line: str) -> tuple[str, str] | None:
    clean = strip_markdown_markup(line)
    folded = normalize_for_match(clean)
    match = re.match(r"^dieu\s+(\d+[a-z]?)\s*[\.:]\s*(.*)$", folded)
    if not match:
        return None
    title = re.sub(r"^Điều\s+\d+[A-Za-z]?\s*[\.:]\s*", "", clean, flags=re.IGNORECASE)
    return match.group(1), title.strip()


def _chapter_heading(line: str) -> tuple[str, str] | None:
    clean = strip_markdown_markup(line)
    folded = normalize_for_match(clean)
    match = re.match(r"^chuong\s+([ivxlcdm]+|\d+)\b\s*(.*)$", folded)
    if not match:
        return None
    title = re.sub(r"^Chương\s+(?:[IVXLCDM]+|\d+)\b\s*", "", clean, flags=re.IGNORECASE)
    return match.group(1).upper(), title.strip()


def _clause_number(line: str) -> str | None:
    match = re.match(r"^\s*(\d+[a-z]?)\.\s+", strip_markdown_markup(line), flags=re.IGNORECASE)
    return normalize_for_match(match.group(1)) if match else None


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
        folded_paragraph = normalize_for_match(strip_markdown_markup(paragraph))
        is_heading = bool(re.match(r"^#{1,6}\s+", paragraph)) or bool(
            re.match(r"^(dieu|chuong|muc)\s+\d+", folded_paragraph)
        )
        if is_heading and current:
            chunks.append(current.strip())
            current = paragraph
            continue

        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            for piece in split_long_paragraph(paragraph, chunk_size):
                chunks.append(piece)
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = paragraph

    if current:
        chunks.append(current.strip())

    return chunks


def split_long_paragraph(paragraph: str, chunk_size: int) -> list[str]:
    """Split long paragraphs at sentence/word boundaries, never mid-word."""
    sentences = re.split(r"(?<=[.!?。])\s+", paragraph)
    pieces: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > chunk_size:
            if current:
                pieces.append(current.strip())
                current = ""
            pieces.extend(split_long_sentence(sentence, chunk_size))
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                pieces.append(current.strip())
            current = sentence

    if current:
        pieces.append(current.strip())

    return pieces


def split_long_sentence(sentence: str, chunk_size: int) -> list[str]:
    words = sentence.split()
    pieces: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                pieces.append(current.strip())
            current = word
    if current:
        pieces.append(current.strip())
    return pieces


def _append_sized_chunks(content: str, metadata: dict, chunks: list[dict], chunk_size: int) -> None:
    """Append text while respecting the existing chunk-size contract."""
    content = content.strip()
    if not content:
        return
    pieces = [content] if len(content) <= chunk_size else split_long_paragraph(content, chunk_size)
    for piece_index, piece in enumerate(pieces, 1):
        piece_metadata = metadata.copy()
        if len(pieces) > 1:
            piece_metadata["piece_index"] = piece_index
        chunks.append({"content": piece, "metadata": piece_metadata})


def _article_units(lines: list[str]) -> list[tuple[str | None, str]]:
    """Group article body lines by Khoản-style numbered clauses."""
    units: list[tuple[str | None, str]] = []
    current_clause: str | None = None
    current_lines: list[str] = []

    for line in lines:
        if not line.strip() and not current_lines:
            continue
        clause = _clause_number(line)
        if clause and current_lines:
            unit_text = "\n".join(current_lines).strip()
            if unit_text:
                units.append((current_clause, unit_text))
            current_lines = []
        if clause:
            current_clause = clause
        current_lines.append(line)

    if current_lines:
        unit_text = "\n".join(current_lines).strip()
        if unit_text:
            units.append((current_clause, unit_text))
    return units


def _split_legal_unit(text: str, chunk_size: int) -> list[str]:
    """Split legal clauses without treating ``1.`` as a sentence boundary."""
    protected = re.sub(r"^(\s*\d+[a-z]?)\.\s+", r"\1§ ", text.strip(), flags=re.IGNORECASE)
    pieces = split_long_paragraph(protected, chunk_size)
    return [piece.replace("§", ".") for piece in pieces]


def _group_short_article_units(
    units: list[tuple[str | None, str]],
    chunk_size: int,
) -> list[tuple[list[str], str]]:
    """Pack adjacent short clauses together to make top-3 context complete."""
    grouped: list[tuple[list[str], str]] = []
    current_clauses: list[str] = []
    current_text = ""

    def flush_current() -> None:
        nonlocal current_clauses, current_text
        if current_text.strip():
            grouped.append((current_clauses[:], current_text.strip()))
        current_clauses = []
        current_text = ""

    for clause, unit in units:
        if len(unit) > chunk_size:
            flush_current()
            for piece in _split_legal_unit(unit, chunk_size):
                grouped.append(([clause] if clause else [], piece))
            continue

        candidate = f"{current_text}\n\n{unit}".strip() if current_text else unit
        if current_text and len(candidate) > chunk_size:
            flush_current()
            current_text = unit
            current_clauses = [clause] if clause else []
        else:
            current_text = candidate
            if clause:
                current_clauses.append(clause)

    flush_current()
    return grouped


def chunk_legal_document(doc: dict, chunk_size: int = 500) -> list[dict]:
    """Chunk legal Markdown by chapter, article, and clause boundaries.

    Each legal-provision chunk carries structured metadata. Exact references can
    therefore be boosted without relying on noisy full-text similarity.
    """
    content = doc.get("content", "")
    base_metadata = doc.get("metadata", {})
    document_ids = sorted(_document_identifiers(content, base_metadata))
    chunks: list[dict] = []
    preamble_lines: list[str] = []
    article_lines: list[str] = []
    article_number: str | None = None
    article_title = ""
    chapter_number: str | None = None
    chapter_title = ""

    def flush_preamble() -> None:
        nonlocal preamble_lines
        if preamble_lines:
            metadata = {
                **base_metadata,
                "chunk_kind": "legal_preamble",
                "document_ids": document_ids,
                "chapter_number": chapter_number,
                "chapter_title": chapter_title,
            }
            _append_sized_chunks("\n".join(preamble_lines), metadata, chunks, chunk_size)
            preamble_lines = []

    def flush_article() -> None:
        nonlocal article_lines, article_number, article_title
        if article_number is None or not article_lines:
            return

        heading = article_lines[0].strip()
        units = _article_units(article_lines[1:]) or [(None, "")]
        body_limit = max(80, chunk_size - len(heading) - 2)
        grouped_units = _group_short_article_units(units, body_limit)
        for clause_numbers, unit in grouped_units:
            body_pieces = [""] if not unit else _split_legal_unit(unit, body_limit)
            for piece_index, body_piece in enumerate(body_pieces, 1):
                chunk_content = f"{heading}\n\n{body_piece}".strip()
                metadata = {
                    **base_metadata,
                    "chunk_kind": "legal_provision",
                    "document_ids": document_ids,
                    "article_number": article_number,
                    "article_title": article_title,
                    "chapter_number": chapter_number,
                    "chapter_title": chapter_title,
                }
                if clause_numbers:
                    metadata["clause_numbers"] = clause_numbers
                if len(clause_numbers) == 1:
                    metadata["clause_number"] = clause_numbers[0]
                if len(body_pieces) > 1:
                    metadata["piece_index"] = piece_index
                chunks.append({"content": chunk_content, "metadata": metadata})

        article_lines = []
        article_number = None
        article_title = ""

    for line in content.splitlines():
        chapter = _chapter_heading(line)
        article = _article_heading(line)

        if chapter:
            flush_article()
            flush_preamble()
            chapter_number, chapter_title = chapter
            preamble_lines.append(line)
            continue

        if article:
            flush_article()
            flush_preamble()
            article_number, article_title = article
            article_lines = [line]
            continue

        if article_lines:
            article_lines.append(line)
        else:
            clean_line = strip_markdown_markup(line)
            if chapter_number and clean_line and clean_line.upper() == clean_line:
                chapter_title = clean_line
            preamble_lines.append(line)

    flush_article()
    flush_preamble()
    return chunks


def chunk_documents_offline(
    documents: list[dict], chunk_size: int = 500, chunk_overlap: int = 50
) -> list[dict]:
    chunks = []
    for doc in documents:
        if doc.get("metadata", {}).get("type") == "legal":
            doc_chunks = chunk_legal_document(doc, chunk_size)
        else:
            doc_chunks = [
                {"content": content, "metadata": doc.get("metadata", {})}
                for content in simple_chunk_text(doc["content"], chunk_size, chunk_overlap)
            ]

        for index, chunk in enumerate(doc_chunks):
            chunks.append(
                {
                    "content": chunk["content"],
                    "metadata": {
                        **chunk.get("metadata", {}),
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
