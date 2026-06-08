"""Task 7 - Local reranking with lexical scoring, MMR and RRF."""

try:
    from .rag_utils import cosine_similarity, hashed_embedding, tokenize
except ImportError:
    from rag_utils import cosine_similarity, hashed_embedding, tokenize


def _keyword_overlap_score(query: str, text: str) -> float:
    query_tokens = set(tokenize(query))
    text_tokens = set(tokenize(text))
    if not query_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Lightweight local replacement for a cross-encoder reranker."""
    reranked = []
    for candidate in candidates:
        item = candidate.copy()
        original = float(candidate.get("score", 0.0))
        overlap = _keyword_overlap_score(query, candidate.get("content", ""))
        item["score"] = 0.65 * overlap + 0.35 * original
        item["metadata"] = item.get("metadata", {})
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


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    """Fuse multiple ranked lists with Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item.get("metadata", {}).get("path") or item.get("content", "")
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
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
