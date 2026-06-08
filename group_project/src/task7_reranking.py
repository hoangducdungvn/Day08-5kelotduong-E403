"""Task 7 - Local reranking with lexical scoring, MMR and RRF."""

import re

try:
    from .rag_utils import cosine_similarity, hashed_embedding, legal_reference_match_score, normalize_for_match, tokenize
except ImportError:
    from rag_utils import cosine_similarity, hashed_embedding, legal_reference_match_score, normalize_for_match, tokenize


def _keyword_overlap_score(query: str, text: str) -> float:
    query_tokens = set(tokenize(query))
    text_tokens = set(tokenize(text))
    if not query_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def _article_numbers(text: str) -> set[str]:
    """Extract Vietnamese legal article numbers, for example ``Điều 249``."""
    return set(re.findall(r"(?:điều|dieu)\s*(\d+)", (text or "").lower()))


def _legal_reference_bonus(query: str, metadata: dict, text: str) -> float:
    return legal_reference_match_score(query, metadata, text)


def _penalty_answer_bonus(query: str, text: str) -> float:
    normalized_query = normalize_for_match(query)
    normalized_text = normalize_for_match(text)
    asks_penalty = "hinh phat" in normalized_query or "khung hinh phat" in normalized_query
    has_penalty_range = bool(re.search(r"phat\s+tu\s+tu\s+\d+", normalized_text))
    return 0.7 if asks_penalty and has_penalty_range else 0.0


def _basic_clause_bonus(query: str, metadata: dict) -> float:
    normalized_query = normalize_for_match(query)
    asks_basic_frame = "co ban" in normalized_query or "khoan 1" in normalized_query
    clauses = {str(value) for value in metadata.get("clause_numbers", [])}
    clause = str(metadata.get("clause_number", ""))
    if clause:
        clauses.add(clause)
    return 1.0 if asks_basic_frame and "1" in clauses else 0.0


def _candidate_key(item: dict) -> str:
    """Identify a chunk without collapsing every chunk from the same file."""
    metadata = item.get("metadata", {}) or {}
    path = metadata.get("path")
    chunk_index = metadata.get("chunk_index")
    if path is not None and chunk_index is not None:
        return f"{path}#{chunk_index}"
    return item.get("content", "")


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Lightweight local replacement for a cross-encoder reranker."""
    reranked = []
    for candidate in candidates:
        item = candidate.copy()
        original = float(candidate.get("score", 0.0))
        content = candidate.get("content", "")
        metadata = item.get("metadata", {})
        overlap = _keyword_overlap_score(query, content)
        # Exact Điều/Khoản/Chương/document matches improve Context Precision:
        # metadata hits identify the requested provision more reliably than
        # broad semantic overlap.
        reference_bonus = _legal_reference_bonus(query, metadata, content)
        penalty_bonus = _penalty_answer_bonus(query, content)
        basic_clause_bonus = _basic_clause_bonus(query, metadata)
        item["score"] = reference_bonus + penalty_bonus + basic_clause_bonus + 0.7 * overlap + 0.3 * original
        item["metadata"] = metadata
        reranked.append(item)

    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Select relevant and diverse candidates with Maximal Marginal Relevance."""
    if not candidates or top_k <= 0:
        return []

    enriched = []
    for candidate in candidates:
        item = candidate.copy()
        item["embedding"] = item.get("embedding") or hashed_embedding(item.get("content", ""))
        enriched.append(item)

    selected: list[int] = []
    remaining = list(range(len(enriched)))
    while remaining and len(selected) < top_k:
        best_index = remaining[0]
        best_score = float("-inf")
        for index in remaining:
            relevance = cosine_similarity(query_embedding, enriched[index]["embedding"])
            diversity_penalty = max(
                (cosine_similarity(enriched[index]["embedding"], enriched[chosen]["embedding"]) for chosen in selected),
                default=0.0,
            )
            score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if score > best_score:
                best_score = score
                best_index = index
        enriched[best_index]["score"] = float(best_score)
        selected.append(best_index)
        remaining.remove(best_index)

    return [enriched[index] for index in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[dict]:
    """Fuse multiple ranked lists with Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    weights = weights or [1.0] * len(ranked_lists)
    for list_index, ranked_list in enumerate(ranked_lists):
        weight = weights[list_index] if list_index < len(weights) else 1.0
        for rank, item in enumerate(ranked_list, 1):
            key = _candidate_key(item)
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            items[key] = item

    fused = []
    for key, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]:
        item = items[key].copy()
        item["score"] = float(score)
        item["metadata"] = item.get("metadata", {})
        fused.append(item)
    return fused


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """Unified reranking interface."""
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "mmr":
        return rerank_mmr(hashed_embedding(query), candidates, top_k)
    if method == "rrf":
        return rerank_rrf([candidates], top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    sample = [{"content": "Toi tang tru ma tuy", "score": 0.8, "metadata": {}}]
    print(rerank("hinh phat ma tuy", sample))
