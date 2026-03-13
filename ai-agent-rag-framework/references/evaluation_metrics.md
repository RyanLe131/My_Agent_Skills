# Evaluation Metrics for RAG

How to measure and improve RAG pipeline quality.

## Metric Categories

### Retrieval Metrics (Is the right context found?)

| Metric | What It Measures | Formula |
|--------|-----------------|---------|
| **Context Precision** | Are retrieved docs relevant? | relevant_retrieved / total_retrieved |
| **Context Recall** | Are all relevant docs retrieved? | relevant_retrieved / total_relevant |
| **MRR** | How high is the first relevant result? | 1 / rank_of_first_relevant |
| **NDCG** | Are results ordered by relevance? | Normalized cumulative gain |
| **Hit Rate** | Is any relevant doc in top-k? | queries_with_hit / total_queries |

### Generation Metrics (Is the answer good?)

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **Faithfulness** | Is every claim supported by context? | > 0.9 |
| **Answer Relevance** | Does the answer address the question? | > 0.8 |
| **Answer Correctness** | Does the answer match ground truth? | > 0.8 |
| **Hallucination Rate** | % of claims not in context | < 0.1 |

### End-to-End Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Latency (p50/p95)** | Response time |
| **Abstention Rate** | How often the system says "I don't know" |
| **User Satisfaction** | Thumbs up/down, CSAT scores |
| **Cost per Query** | Token usage × price |

## Evaluation Frameworks

### RAGAS

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

results = evaluate(dataset=eval_dataset, metrics=[
    faithfulness, answer_relevancy, context_precision, context_recall
])
```

### DeepEval

```python
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

evaluate(test_cases=cases, metrics=[
    FaithfulnessMetric(threshold=0.7),
    AnswerRelevancyMetric(threshold=0.7),
])
```

### LLM-as-Judge (Custom)

Use a strong LLM to evaluate answers on specific criteria. Useful when frameworks don't cover your needs.

## Diagnostic Flowchart

```
Problem: Low overall quality
│
├─ Low Context Precision (retrieved docs irrelevant)?
│  → Better chunking (smaller, semantic)
│  → Add re-ranking (cross-encoder)
│  → Improve embedding model
│
├─ Low Context Recall (missing relevant docs)?
│  → Increase top-k
│  → Try hybrid search (BM25 + dense)
│  → Increase chunk overlap
│  → Multi-query retrieval
│
├─ Low Faithfulness (hallucinations)?
│  → Strengthen grounding in prompt
│  → Set temperature = 0
│  → Reduce context noise (re-rank first)
│
└─ Low Answer Relevance (off-topic answers)?
   → Improve prompt template
   → Check if context quality is the root cause
   → Add output format constraints
```

## Test Dataset Construction

A good evaluation dataset has:
- **20+ questions minimum** (50+ for production)
- **Diverse question types**: factual, reasoning, multi-hop, comparison
- **Ground truth answers**: Human-verified reference answers
- **Relevant document IDs**: For measuring retrieval quality

```json
[
  {
    "question": "What is the return policy for electronics?",
    "ground_truth": "Electronics can be returned within 30 days.",
    "relevant_doc_ids": ["doc_42"],
    "difficulty": "simple"
  },
  {
    "question": "Compare the warranty terms for laptops vs phones.",
    "ground_truth": "Laptops have 2-year warranty, phones have 1-year.",
    "relevant_doc_ids": ["doc_15", "doc_22"],
    "difficulty": "comparison"
  }
]
```

## Iteration Cadence

1. **Weekly**: Run full RAGAS evaluation on test set
2. **Per-change**: Quick evaluation on subset after pipeline changes
3. **Continuous**: Monitor production metrics (latency, abstention rate)
4. **Monthly**: Update test dataset with new failure cases from production
