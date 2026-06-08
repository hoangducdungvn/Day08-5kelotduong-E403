"""Task 9 - Hybrid retrieval pipeline with fallback."""

try:
    from .rag_utils import extract_legal_references, tokenize
    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search
    from .task7_reranking import rerank, rerank_rrf
    from .task8_pageindex_vectorless import pageindex_search
except ImportError:
    from rag_utils import extract_legal_references, tokenize
    from task5_semantic_search import semantic_search
    from task6_lexical_search import lexical_search
    from task7_reranking import rerank, rerank_rrf
    from task8_pageindex_vectorless import pageindex_search

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"


def _content_overlap(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _article_key(item: dict) -> str | None:
    metadata = item.get("metadata", {}) or {}
    path = metadata.get("path")
    article = metadata.get("article_number")
    if path and article:
        return f"{path}#article-{article}"
    return None


def deduplicate_and_diversify(
    candidates: list[dict],
    top_k: int = DEFAULT_TOP_K,
    overlap_threshold: float = 0.82,
    max_per_article: int = 2,
) -> list[dict]:
    """Remove duplicate contexts and avoid overloading one article.

    This improves Context Precision by keeping near-identical chunks out of the
    prompt, while still allowing up to two clauses from the same article when a
    specific provision needs multiple pieces of evidence.
    """
    selected: list[dict] = []
    article_counts: dict[str, int] = {}
    deferred: list[dict] = []

    for candidate in candidates:
        content = candidate.get("content", "")
        if any(_content_overlap(content, item.get("content", "")) >= overlap_threshold for item in selected):
            continue

        article_key = _article_key(candidate)
        if article_key and article_counts.get(article_key, 0) >= max_per_article:
            deferred.append(candidate)
            continue

        selected.append(candidate)
        if article_key:
            article_counts[article_key] = article_counts.get(article_key, 0) + 1
        if len(selected) >= top_k:
            return selected

    for candidate in deferred:
        content = candidate.get("content", "")
        if any(_content_overlap(content, item.get("content", "")) >= overlap_threshold for item in selected):
            continue
        selected.append(candidate)
        if len(selected) >= top_k:
            break

    return selected


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Run semantic + lexical retrieval, fuse, rerank and fallback if needed."""
    if top_k <= 0:
        return []

    query_refs = extract_legal_references(query)
    has_legal_reference = any(query_refs.values())

    # Retrieve moderately broadly, then cut aggressively after reranking. Legal
    # reference queries give BM25/RRF extra weight because exact Điều/Khoản hits
    # are more trustworthy than semantic similarity for statute lookup.
    candidate_k = top_k * 3
    dense_results = semantic_search(query, top_k=candidate_k)
    sparse_results = lexical_search(query, top_k=candidate_k)
    fusion_weights = [1.0, 2.5] if has_legal_reference else [1.0, 1.0]
    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 4, weights=fusion_weights)

    for item in merged:
        item["source"] = "hybrid"

    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k * 2, method=RERANK_METHOD)
        for item in final_results:
            item["source"] = "hybrid"
    else:
        final_results = merged[:top_k]

    max_per_article = 3 if query_refs["article_numbers"] else 2
    final_results = deduplicate_and_diversify(
        final_results,
        top_k=top_k,
        max_per_article=max_per_article,
    )

    if not final_results or final_results[0].get("score", 0.0) < score_threshold:
        fallback = pageindex_search(query, top_k=top_k)
        return deduplicate_and_diversify(
            fallback,
            top_k=top_k,
            max_per_article=max_per_article,
        )

    return final_results[:top_k]


if __name__ == "__main__":
    for result in retrieve("hinh phat ma tuy", top_k=3):
        print(f"[{result['score']:.3f}] [{result['source']}] {result['content'][:100]}...")
