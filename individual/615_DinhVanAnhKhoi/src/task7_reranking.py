"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

import sys
from pathlib import Path
# Add project root to sys.path to prevent ModuleNotFoundError
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import requests
import numpy as np
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


def cosine_sim(vec_a: list, vec_b: list) -> float:
    """Calculates cosine similarity between two lists representing vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a > 0 and norm_b > 0:
        return float(dot / (norm_a * norm_b))
    return 0.0


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
    if not candidates:
        return []

    jina_key = os.getenv("JINA_API_KEY", "")

    # Option A: Jina Reranker API
    if jina_key and jina_key != "jina_xxx":
        try:
            response = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Authorization": f"Bearer {jina_key}"},
                json={
                    "model": "jina-reranker-v2-base-multilingual",
                    "query": query,
                    "documents": [c["content"] for c in candidates],
                    "top_n": top_k
                },
                timeout=10
            )
            if response.status_code == 200:
                reranked = response.json().get("results", [])
                return [
                    {**candidates[r["index"]], "score": float(r["relevance_score"])}
                    for r in reranked
                ]
            else:
                print(f"[Warning] Jina Rerank API returned status {response.status_code}. Falling back to local similarity.")
        except Exception as e:
            print(f"[Warning] Jina Rerank API failed: {e}. Falling back to local similarity.")

    # Fallback Option: Local similarity using cached SentenceTransformer model
    try:
        from src.task5_semantic_search import get_model

        model = get_model()
        query_vector = model.encode(query)

        results = []
        for c in candidates:
            c_vector = model.encode(c["content"])
            sim = cosine_sim(query_vector, c_vector)
            item = c.copy()
            item["score"] = sim
            results.append(item)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    except Exception as fallback_err:
        print(f"[Error] Fallback local similarity calculation failed: {fallback_err}")
        # Default safety fallback: return candidates in original order with scores preserved
        output = [c.copy() for c in candidates]
        for item in output:
            if "score" not in item:
                item["score"] = 0.0
        output.sort(key=lambda x: x["score"], reverse=True)
        return output[:top_k]


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

    selected = []
    remaining = list(range(len(candidates)))
    model = None

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float('-inf')

        for idx in remaining:
            cand = candidates[idx]
            # Ensure candidate has embedding
            if "embedding" not in cand or cand["embedding"] is None:
                if model is None:
                    from src.task5_semantic_search import get_model
                    model = get_model()
                cand["embedding"] = model.encode(cand["content"]).tolist()

            # Relevance to query
            relevance = cosine_sim(query_embedding, cand["embedding"])

            # Max similarity to already selected candidates
            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sel_cand = candidates[sel_idx]
                if "embedding" not in sel_cand or sel_cand["embedding"] is None:
                    if model is None:
                        from src.task5_semantic_search import get_model
                        model = get_model()
                    sel_cand["embedding"] = model.encode(sel_cand["content"]).tolist()
                sim = cosine_sim(cand["embedding"], sel_cand["embedding"])
                max_sim_to_selected = max(max_sim_to_selected, sim)

            # MMR score calculation
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    # Convert results and ensure score matches final rank relevance or original relevance
    output = []
    for rank, i in enumerate(selected):
        item = candidates[i].copy()
        # Ensure we keep a valid score
        if "score" not in item:
            item["score"] = float(1.0 - rank * 0.1)
        output.append(item)
    return output


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores = {}  # content -> score
    content_map = {}  # content -> full dict

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in content_map:
                content_map[key] = item

    # Sort by RRF score descending
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = score
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if not candidates:
        return []

    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        try:
            from src.task5_semantic_search import get_model
            model = get_model()
            query_embedding = model.encode(query).tolist()
            return rerank_mmr(query_embedding, candidates, top_k)
        except Exception as e:
            print(f"[Warning] MMR embedding computation failed: {e}. Falling back to default list.")
            return candidates[:top_k]
    elif method == "rrf":
        return rerank_rrf([candidates], top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    print("Testing rerank with cross_encoder:")
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2, method="cross_encoder")
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")

