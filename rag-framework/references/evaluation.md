# RAG Evaluation

Measure and iterate on RAG quality across retrieval accuracy, answer faithfulness, and end-to-end relevance.

## Metrics Overview

| Metric | Measures | Range | Good Target |
|--------|----------|-------|-------------|
| **Context Precision** | Are retrieved docs relevant? | 0–1 | > 0.8 |
| **Context Recall** | Are all needed docs retrieved? | 0–1 | > 0.8 |
| **Faithfulness** | Is the answer supported by context? | 0–1 | > 0.9 |
| **Answer Relevance** | Does the answer address the question? | 0–1 | > 0.8 |
| **Answer Correctness** | Is the answer factually correct? | 0–1 | > 0.8 |
| **Hallucination Rate** | % of claims not in context | 0–1 | < 0.1 |

### Retrieval Metrics

- **Context Precision**: Of the retrieved chunks, how many are actually relevant?
- **Context Recall**: Of all the relevant chunks in the corpus, how many were retrieved?
- **MRR (Mean Reciprocal Rank)**: How high does the first relevant result rank?

### Generation Metrics

- **Faithfulness**: Every claim in the answer is supported by retrieved context
- **Answer Relevance**: The answer directly addresses the question asked
- **Answer Correctness**: Compared against a ground truth reference answer

## RAGAS Evaluation

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas import EvaluationDataset, SingleTurnSample

# Prepare evaluation samples
samples = []
for item in test_data:
    # Run your RAG pipeline
    result = rag_chain.invoke(item["question"])
    retrieved_docs = retriever.invoke(item["question"])
    
    samples.append(SingleTurnSample(
        user_input=item["question"],
        response=result,
        retrieved_contexts=[doc.page_content for doc in retrieved_docs],
        reference=item["ground_truth"],
    ))

dataset = EvaluationDataset(samples=samples)

results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)
print(results)
# Access per-sample scores
df = results.to_pandas()
```

## DeepEval

```python
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)

test_cases = []
for item in test_data:
    result = rag_chain.invoke(item["question"])
    retrieved_docs = retriever.invoke(item["question"])
    
    test_cases.append(LLMTestCase(
        input=item["question"],
        actual_output=result,
        expected_output=item["ground_truth"],
        retrieval_context=[doc.page_content for doc in retrieved_docs],
    ))

metrics = [
    FaithfulnessMetric(threshold=0.7),
    AnswerRelevancyMetric(threshold=0.7),
    ContextualPrecisionMetric(threshold=0.7),
    ContextualRecallMetric(threshold=0.7),
]

evaluate(test_cases=test_cases, metrics=metrics)
```

## Building a Test Dataset

```python
# Manual curation (highest quality)
test_data = [
    {
        "question": "What is the return policy for electronics?",
        "ground_truth": "Electronics can be returned within 30 days with receipt.",
        "relevant_doc_ids": ["doc_42", "doc_43"],
    },
]

# LLM-generated test questions from your documents
from ragas.testset import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

generator = TestsetGenerator.from_langchain(
    generator_llm=ChatOpenAI(model="gpt-4o"),
    critic_llm=ChatOpenAI(model="gpt-4o"),
    embeddings=OpenAIEmbeddings(),
)

testset = generator.generate_with_langchain_docs(
    documents=docs,
    test_size=20,
    distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
)
test_df = testset.to_pandas()
```

## Custom Evaluation

```python
from langchain_openai import ChatOpenAI

eval_llm = ChatOpenAI(model="gpt-4o", temperature=0)

FAITHFULNESS_PROMPT = """Given the context and the answer, determine if every claim 
in the answer is supported by the context.

Context: {context}
Answer: {answer}

Score (0.0 to 1.0) and explain:"""

def evaluate_faithfulness(answer: str, context: str) -> float:
    response = eval_llm.invoke(
        FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    )
    # Parse score from response
    import re
    match = re.search(r'(\d+\.?\d*)', response.content)
    return float(match.group(1)) if match else 0.0
```

## Diagnostic Workflow

When evaluation scores are low, diagnose and fix systematically:

```
Low Context Precision → Retrieved docs are irrelevant
  Fix: Improve chunking, add re-ranking, tune embedding model

Low Context Recall → Missing relevant docs
  Fix: Increase top-k, try hybrid search, improve chunking overlap

Low Faithfulness → Answer contains unsupported claims
  Fix: Strengthen prompt ("only use provided context"), lower temperature

Low Answer Relevance → Answer doesn't address the question
  Fix: Improve prompt template, check context quality first
```

## Monitoring in Production

```python
import logging
import time

logger = logging.getLogger("rag_monitor")

def monitored_query(chain, question: str):
    """Wrap RAG queries with latency and quality logging."""
    start = time.time()
    result = chain.invoke({"query": question})
    latency = time.time() - start
    
    num_sources = len(result.get("source_documents", []))
    answer_length = len(result.get("result", ""))
    abstained = "don't have enough information" in result.get("result", "").lower()
    
    logger.info(
        "rag_query",
        extra={
            "latency_s": round(latency, 2),
            "num_sources": num_sources,
            "answer_length": answer_length,
            "abstained": abstained,
        },
    )
    return result
```

## Checklist

- [ ] Test dataset created (20+ questions with ground truth)
- [ ] Retrieval metrics measured (context precision and recall)
- [ ] Generation metrics measured (faithfulness and answer relevance)
- [ ] Low-scoring samples diagnosed and root-caused
- [ ] Pipeline parameters tuned based on evaluation results
- [ ] Monitoring set up for production deployment
- [ ] Evaluation runs automated in CI/CD pipeline
