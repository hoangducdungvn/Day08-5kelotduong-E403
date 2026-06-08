"""Task 10 - Citation-focused generation."""

import os

from dotenv import load_dotenv

try:
    from .task9_retrieval_pipeline import retrieve
except ImportError:
    from task9_retrieval_pipeline import retrieve

load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

SYSTEM_PROMPT = """Answer in Vietnamese using only the provided context.
Every factual claim must include a citation in square brackets.
If the context is insufficient, say that the information cannot be verified."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Place strongest chunks at the beginning and end of the context."""
    if len(chunks) <= 2:
        return list(chunks)

    reordered = []
    tail = []
    for index, chunk in enumerate(chunks):
        if index % 2 == 0:
            reordered.append(chunk)
        else:
            tail.append(chunk)
    reordered.extend(reversed(tail))
    return reordered


def format_context(chunks: list[dict]) -> str:
    """Format chunks with stable citation labels."""
    parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source") or metadata.get("path") or f"Source {index}"
        doc_type = metadata.get("type", "unknown")
        parts.append(
            f"[Document {index} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk.get('content', '')}"
        )
    return "\n\n---\n\n".join(parts)


def _offline_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "Toi khong the xac minh thong tin nay tu nguon hien co."

    lines = [f"Tra loi dua tren cac nguon da truy xuat cho cau hoi: {query}"]
    for index, chunk in enumerate(chunks[:3], 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source") or metadata.get("path") or f"Document {index}"
        excerpt = " ".join(chunk.get("content", "").split())[:350]
        lines.append(f"- {excerpt} [{source}]")
    return "\n".join(lines)


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Retrieve evidence, format context, and return an answer with citations."""
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    answer = ""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = response.choices[0].message.content or ""
        except Exception:
            answer = ""

    if not answer:
        answer = _offline_answer(query, reordered)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "none") if chunks else "none",
    }


if __name__ == "__main__":
    result = generate_with_citation("Hinh phat tang tru ma tuy?")
    print(result["answer"])
