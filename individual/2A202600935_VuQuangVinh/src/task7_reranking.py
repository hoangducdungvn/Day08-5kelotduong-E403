"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

import os
from typing import Optional

import numpy as np


# =============================================================================
# Helper
# =============================================================================

def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity giữa 2 vectors (đã normalize thì = dot product)."""
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# =============================================================================
# Method 1: Cross-encoder (Jina API)
# =============================================================================

def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    import requests

    JINA_API_KEY = os.getenv("JINA_API_KEY", "")
    if not JINA_API_KEY:
        raise ValueError("JINA_API_KEY không có trong .env — dùng method='rrf' thay thế")

    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={"Authorization": f"Bearer {JINA_API_KEY}"},
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": [c["content"] for c in candidates],
            "top_n": top_k,
        },
    )
    response.raise_for_status()
    reranked = response.json()["results"]

    return [
        {**candidates[r["index"]], "score": round(r["relevance_score"], 4)}
        for r in reranked
    ]


# =============================================================================
# Method 2: MMR — Maximal Marginal Relevance
# =============================================================================

def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if not candidates:
        return []

    selected = []       # indices đã chọn
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            # Relevance to query
            relevance = _cosine_sim(query_embedding, candidates[idx]["embedding"])

            # Max similarity to already selected docs (diversity penalty)
            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sim = _cosine_sim(
                    candidates[idx]["embedding"],
                    candidates[sel_idx]["embedding"],
                )
                max_sim_to_selected = max(max_sim_to_selected, sim)

            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [
        {**candidates[i], "score": round(
            lambda_param * _cosine_sim(query_embedding, candidates[i]["embedding"]), 4
        )}
        for i in selected
    ]


# =============================================================================
# Method 3: RRF — Reciprocal Rank Fusion  ← default
# =============================================================================

def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}   # content → rrf score
    content_map: dict[str, dict] = {}   # content → full item

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = round(score, 6)
        results.append(item)

    return results


# =============================================================================
# Unified interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",  # default: rrf (không cần API key / embedding)
    # extra args cho từng method
    ranked_lists: Optional[list[list[dict]]] = None,
    query_embedding: Optional[list[float]] = None,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: "cross_encoder" | "mmr" | "rrf"
        ranked_lists: (rrf only) list of ranked lists từ nhiều retriever
        query_embedding: (mmr only) vector embedding của query
        lambda_param: (mmr only) relevance vs diversity trade-off

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)

    elif method == "mmr":
        if query_embedding is None:
            raise ValueError("mmr cần query_embedding")
        if not candidates or "embedding" not in candidates[0]:
            raise ValueError("mmr cần candidates có key 'embedding'")
        return rerank_mmr(query_embedding, candidates, top_k, lambda_param)

    elif method == "rrf":
        # Nếu không truyền ranked_lists → wrap candidates thành 1 list
        lists = ranked_lists if ranked_lists is not None else [candidates]
        return rerank_rrf(lists, top_k)

    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test RRF với 2 ranked lists (semantic + lexical)
    semantic_results = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
        {"content": "Nghệ sĩ Chi Dân bị bắt vì sử dụng ma tuý", "score": 0.5, "metadata": {}},
    ]
    lexical_results = [
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 19.4, "metadata": {}},
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 16.3, "metadata": {}},
        {"content": "Nghệ sĩ Chi Dân bị bắt vì sử dụng ma tuý", "score": 8.1, "metadata": {}},
    ]

    results = rerank(
        query="hình phạt tàng trữ ma tuý",
        candidates=[],
        top_k=3,
        method="rrf",
        ranked_lists=[semantic_results, lexical_results],
    )
    for r in results:
        print(f"[{r['score']:.6f}] {r['content']}")