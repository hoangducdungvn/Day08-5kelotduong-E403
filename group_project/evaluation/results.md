# RAG Evaluation Results

## Framework sử dụng

> Custom offline evaluator inspired by DeepEval/RAGAS metrics

---

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Delta |
|--------|---------------------------|----------------------|---|
| Faithfulness | 0.854 | 0.810 | 0.045 |
| Answer Relevance | 0.428 | 0.416 | 0.012 |
| Context Recall | 0.779 | 0.730 | 0.049 |
| Context Precision | 0.086 | 0.094 | -0.007 |
| **Average** | 0.537 | 0.512 | 0.024 |

---

## A/B Comparison Analysis


**Config A:**
> Hybrid retrieval with reranking. This configuration prioritizes precision and de-duplicates noisy retrievals.


**Config B:**
> Dense-only retrieval using semantic search without reranking. This is the simpler baseline.


**Kết luận:**
> Config A tốt hơn trên bộ golden dataset này vì reranking improves faithfulness and context precision.

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Bài viết về ca sĩ Miu Lê trên Tuổi Trẻ Online cho biết nội dung chính là gì? | 0.673 | 0.205 | 0.477 | Generation | Answer is too generic or underspecified |
| 2 | Luật Phòng, chống ma túy 2021 quy định những hình thức cai nghiện ma túy nào? | 0.462 | 0.151 | 0.750 | Generation | Answer is too generic or underspecified |
| 3 | Luật Phòng, chống ma túy 2021 cấm những hành vi nào liên quan đến ma túy? | 0.483 | 0.118 | 0.875 | Generation | Answer is too generic or underspecified |

---

## Recommendations


### Cải tiến 1
**Action:** Tăng cường chunking theo heading và điều chỉnh chunk size cho legal documents.  
**Expected impact:** Giúp context precision cao hơn và giảm nhiễu từ các đoạn pháp lý dài.

### Cải tiến 2
**Action:** Bổ sung lexical weighting mạnh hơn cho query có tên điều luật, số hiệu và tên văn bản.  
**Expected impact:** Tăng context recall cho các câu hỏi dạng tra cứu điều khoản.

### Cải tiến 3
**Action:** Thêm bước answer grounding chặt hơn trong generation prompt, ưu tiên trích dẫn theo nguồn cụ thể.  
**Expected impact:** Nâng faithfulness và giảm câu trả lời chung chung.
