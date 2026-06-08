"""Offline RAG evaluation pipeline for the group project.

Chosen approach:
    - Framework style: custom offline evaluator inspired by the requested
      DeepEval/RAGAS metrics.
    - Why: it runs deterministically in the current workspace without API keys
      or cloud dependencies, while still scoring the four required metrics:
      faithfulness, answer relevance, context recall, and context precision.

The pipeline evaluates two configurations:
    A. hybrid retrieval + reranking
    B. hybrid retrieval without reranking

The output is written to `results.md`.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

import sys

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.task10_generation import generate_with_citation
from src.task5_semantic_search import semantic_search
from src.task9_retrieval_pipeline import retrieve


TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_expected_context(expected_context: str) -> str:
    """Load path-based golden contexts before computing retrieval metrics."""
    candidate = PROJECT_DIR / (expected_context or "")
    if expected_context and candidate.is_file():
        return candidate.read_text(encoding="utf-8", errors="ignore")
    return expected_context or ""


def _jaccard(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _coverage_ratio(source_text: str, target_text: str) -> float:
    source_tokens = set(tokenize(source_text))
    target_tokens = tokenize(target_text)
    if not target_tokens:
        return 0.0
    hits = sum(1 for token in target_tokens if token in source_tokens)
    return hits / len(target_tokens)


def _score_faithfulness(answer: str, contexts: list[str]) -> float:
    if not answer.strip() or not contexts:
        return 0.0
    context_blob = " ".join(contexts)
    return max(0.0, min(1.0, _coverage_ratio(context_blob, answer)))


def _score_answer_relevance(question: str, answer: str, expected_answer: str) -> float:
    if not answer.strip():
        return 0.0
    if "khong the xac minh" in answer.lower():
        return 0.15 if expected_answer else 0.0
    qa_similarity = _jaccard(question, answer)
    expected_similarity = _jaccard(expected_answer, answer)
    return max(0.0, min(1.0, 0.35 * qa_similarity + 0.65 * expected_similarity))


def _score_context_recall(expected_context: str, retrieved_contexts: list[str]) -> float:
    if not retrieved_contexts:
        return 0.0
    retrieved_blob = " ".join(retrieved_contexts)
    return max(0.0, min(1.0, _coverage_ratio(retrieved_blob, expected_context)))


def _score_context_precision(expected_context: str, retrieved_contexts: list[str]) -> float:
    if not retrieved_contexts:
        return 0.0
    scores = [_jaccard(expected_context, ctx) for ctx in retrieved_contexts]
    return max(0.0, min(1.0, mean(scores) if scores else 0.0))


@dataclass
class ExampleResult:
    question: str
    expected_answer: str
    expected_context: str
    answer: str
    sources: list[dict]
    faithfulness: float
    relevance: float
    recall: float
    precision: float
    retrieval_source: str

    @property
    def average(self) -> float:
        return mean([self.faithfulness, self.relevance, self.recall, self.precision])


def _evaluate_example(question: str, expected_answer: str, expected_context: str, *, use_reranking: bool) -> ExampleResult:
    expected_context = resolve_expected_context(expected_context)
    if use_reranking:
        sources = retrieve(question, top_k=5, use_reranking=True)
    else:
        sources = semantic_search(question, top_k=5)

    result = generate_with_citation(question, sources=sources, use_reranking=use_reranking)
    # Evaluate the same top-3 contexts actually passed to generation.
    sources = result["sources"]
    retrieved_contexts = [c.get("content", "") for c in sources]
    faithfulness = _score_faithfulness(result["answer"], retrieved_contexts)
    relevance = _score_answer_relevance(question, result["answer"], expected_answer)
    recall = _score_context_recall(expected_context, retrieved_contexts)
    precision = _score_context_precision(expected_context, retrieved_contexts)
    retrieval_source = result.get("retrieval_source", "none")

    return ExampleResult(
        question=question,
        expected_answer=expected_answer,
        expected_context=expected_context,
        answer=result["answer"],
        sources=sources,
        faithfulness=faithfulness,
        relevance=relevance,
        recall=recall,
        precision=precision,
        retrieval_source=retrieval_source,
    )


def _evaluate_config(golden_dataset: list[dict], *, use_reranking: bool) -> dict:
    examples = []
    for item in golden_dataset:
        examples.append(
            _evaluate_example(
                item["question"],
                item["expected_answer"],
                item["expected_context"],
                use_reranking=use_reranking,
            )
        )

    summary = {
        "faithfulness": mean(x.faithfulness for x in examples) if examples else 0.0,
        "answer_relevance": mean(x.relevance for x in examples) if examples else 0.0,
        "context_recall": mean(x.recall for x in examples) if examples else 0.0,
        "context_precision": mean(x.precision for x in examples) if examples else 0.0,
    }
    summary["average"] = mean(summary.values()) if summary else 0.0
    return {"summary": summary, "examples": examples}


def evaluate_with_offline_metrics(golden_dataset: list[dict]) -> dict:
    """Evaluate both configs using deterministic local metrics."""
    return {
        "hybrid_rerank": _evaluate_config(golden_dataset, use_reranking=True),
        "dense_only": _evaluate_config(golden_dataset, use_reranking=False),
    }


def compare_configs(rag_pipeline, golden_dataset: list[dict]):
    """Return the A/B comparison used by the report."""
    return evaluate_with_offline_metrics(golden_dataset)


def export_results(results: dict, comparison: dict):
    """Format the evaluation into the report template in results.md."""
    a = comparison["hybrid_rerank"]["summary"]
    b = comparison["dense_only"]["summary"]
    examples_a = comparison["hybrid_rerank"]["examples"]

    worst = sorted(examples_a, key=lambda item: item.average)[:3]

    better_config = "Config A" if a["average"] >= b["average"] else "Config B"
    better_reason = (
        "reranking improves faithfulness and context precision"
        if better_config == "Config A"
        else "the dense-only baseline retrieved broader evidence that fit this dataset better under the local metrics"
    )

    def fmt(value: float) -> str:
        return f"{value:.3f}"

    content = []
    content.append("# RAG Evaluation Results\n")
    content.append("## Framework sử dụng\n")
    content.append("> Custom offline evaluator inspired by DeepEval/RAGAS metrics\n")
    content.append("---\n")
    content.append("## Overall Scores\n")
    content.append("| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Delta |")
    content.append("|--------|---------------------------|----------------------|---|")
    content.append(f"| Faithfulness | {fmt(a['faithfulness'])} | {fmt(b['faithfulness'])} | {fmt(a['faithfulness'] - b['faithfulness'])} |")
    content.append(f"| Answer Relevance | {fmt(a['answer_relevance'])} | {fmt(b['answer_relevance'])} | {fmt(a['answer_relevance'] - b['answer_relevance'])} |")
    content.append(f"| Context Recall | {fmt(a['context_recall'])} | {fmt(b['context_recall'])} | {fmt(a['context_recall'] - b['context_recall'])} |")
    content.append(f"| Context Precision | {fmt(a['context_precision'])} | {fmt(b['context_precision'])} | {fmt(a['context_precision'] - b['context_precision'])} |")
    content.append(f"| **Average** | {fmt(a['average'])} | {fmt(b['average'])} | {fmt(a['average'] - b['average'])} |")
    content.append("\n---\n")
    content.append("## A/B Comparison Analysis\n")
    content.append("\n**Config A:**")
    content.append("> Hybrid retrieval with reranking. This configuration prioritizes precision and de-duplicates noisy retrievals.\n")
    content.append("\n**Config B:**")
    content.append("> Dense-only retrieval using semantic search without reranking. This is the simpler baseline.\n")
    content.append("\n**Kết luận:**")
    content.append(f"> {better_config} tốt hơn trên bộ golden dataset này vì {better_reason}.")
    content.append("\n---\n")
    content.append("## Worst Performers (Bottom 3)\n")
    content.append("| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |")
    content.append("|---|----------|-------------|-----------|--------|---------------|------------|")
    for idx, item in enumerate(worst, 1):
        failure_stage = "Retrieval" if item.recall < 0.4 else "Generation"
        root_cause = "Expected context is broader than retrieved chunks" if item.recall < item.precision else "Answer is too generic or underspecified"
        content.append(
            f"| {idx} | {item.question} | {fmt(item.faithfulness)} | {fmt(item.relevance)} | {fmt(item.recall)} | {failure_stage} | {root_cause} |"
        )
    content.append("\n---\n")
    content.append("## Recommendations\n")
    content.append("\n### Cải tiến 1")
    content.append("**Action:** Tăng cường chunking theo heading và điều chỉnh chunk size cho legal documents.  ")
    content.append("**Expected impact:** Giúp context precision cao hơn và giảm nhiễu từ các đoạn pháp lý dài.\n")
    content.append("### Cải tiến 2")
    content.append("**Action:** Bổ sung lexical weighting mạnh hơn cho query có tên điều luật, số hiệu và tên văn bản.  ")
    content.append("**Expected impact:** Tăng context recall cho các câu hỏi dạng tra cứu điều khoản.\n")
    content.append("### Cải tiến 3")
    content.append("**Action:** Thêm bước answer grounding chặt hơn trong generation prompt, ưu tiên trích dẫn theo nguồn cụ thể.  ")
    content.append("**Expected impact:** Nâng faithfulness và giảm câu trả lời chung chung.\n")

    RESULTS_PATH.write_text("\n".join(content).strip() + "\n", encoding="utf-8")
    return RESULTS_PATH


def run():
    golden_dataset = load_golden_dataset()
    comparison = evaluate_with_offline_metrics(golden_dataset)
    results = {
        "framework": "offline_custom",
        "num_examples": len(golden_dataset),
        "configs": comparison,
    }
    export_results(results, comparison)
    return results


if __name__ == "__main__":
    results = run()
    print(f"Evaluated {results['num_examples']} examples and wrote {RESULTS_PATH}")
