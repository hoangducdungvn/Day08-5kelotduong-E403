"""Task 6 - Dependency-free BM25 lexical search."""

from __future__ import annotations

import math

try:
    from .rag_utils import default_chunks, tokenize
except ImportError:
    from rag_utils import default_chunks, tokenize

CORPUS: list[dict] = default_chunks()


class SimpleBM25:
    def __init__(self, corpus: list[dict], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.tokenized = [tokenize(doc["content"]) for doc in corpus]
        self.doc_lengths = [len(tokens) for tokens in self.tokenized]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.idf = self._build_idf()

    def _build_idf(self) -> dict[str, float]:
        doc_count = len(self.tokenized)
        dfs: dict[str, int] = {}
        for tokens in self.tokenized:
            for token in set(tokens):
                dfs[token] = dfs.get(token, 0) + 1
        return {
            token: math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
            for token, df in dfs.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = []
        for tokens, doc_len in zip(self.tokenized, self.doc_lengths):
            frequencies: dict[str, int] = {}
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1

            score = 0.0
            for token in query_tokens:
                tf = frequencies.get(token, 0)
                if tf == 0:
                    continue
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1))
                score += self.idf.get(token, 0.0) * (tf * (self.k1 + 1)) / denominator
            scores.append(score)
        return scores


def build_bm25_index(corpus: list[dict]):
    """Build a BM25 index from a list of chunk dictionaries."""
    return SimpleBM25(corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Return keyword matches sorted by BM25 score descending."""
    if top_k <= 0:
        return []

    corpus = CORPUS or default_chunks()
    bm25 = build_bm25_index(corpus)
    scores = bm25.get_scores(tokenize(query))
    ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)

    results = []
    for index in ranked_indices[:top_k]:
        if scores[index] <= 0:
            continue
        results.append(
            {
                "content": corpus[index]["content"],
                "score": float(scores[index]),
                "metadata": corpus[index].get("metadata", {}),
            }
        )
    return results


if __name__ == "__main__":
    for result in lexical_search("ma tuy", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
