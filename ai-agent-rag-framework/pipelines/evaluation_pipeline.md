# Evaluation Pipeline

End-to-end workflow for measuring and improving RAG quality.

## Pipeline Steps

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Build   │ →  │   Run    │ →  │  Score   │ →  │ Diagnose │ →  │  Iterate │
│ Test Set │    │ Pipeline │    │ Metrics  │    │ Failures │    │ & Improve│
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

## Step 1: Build Test Dataset

Use the provided template or create your own:

```python
import json

# Load example questions
with open("./assets/evaluation_questions.json") as f:
    test_data = json.load(f)

# Or generate from documents
from ragas.testset import TestsetGenerator
generator = TestsetGenerator.from_langchain(
    generator_llm=llm, critic_llm=llm, embeddings=embeddings
)
testset = generator.generate_with_langchain_docs(documents=docs, test_size=30)
```

## Step 2: Run RAG Pipeline

```python
from scripts.generate_embeddings import load_vectorstore
from scripts.retrieve_context import create_rag_chain

vectorstore = load_vectorstore(provider="chromadb", persist_dir="./vector_db")
chain = create_rag_chain(vectorstore)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

## Step 3: Compute Metrics

```python
from scripts.evaluate_answers import evaluate_rag

results = evaluate_rag(
    rag_chain=chain,
    retriever=retriever,
    test_data=test_data,
)

print(results["scores"])
# {
#   'faithfulness': 0.92,
#   'answer_relevancy': 0.85,
#   'context_precision': 0.78,
#   'context_recall': 0.81
# }
```

## Step 4: Diagnose Failures

```python
from scripts.evaluate_answers import generate_evaluation_report

report = generate_evaluation_report(results, output_path="./eval_report.md")

# Review per-sample scores
df = results["dataframe"]
low_faith = df[df["faithfulness"] < 0.7]
print(f"Questions with low faithfulness: {len(low_faith)}")
print(low_faith[["user_input", "faithfulness"]])
```

## Step 5: Iterate

Based on diagnosis, apply targeted fixes:

| Low Metric | Actions to Try |
|-----------|---------------|
| Context Precision < 0.8 | Smaller chunks, add re-ranking, better embeddings |
| Context Recall < 0.8 | Increase top-k, hybrid search, more overlap |
| Faithfulness < 0.9 | Strengthen prompt, temperature=0, filter noisy context |
| Answer Relevance < 0.8 | Improve prompt template, check context quality |

Then re-run evaluation to measure impact.

## Automation

```python
# Run as part of CI/CD
# scripts/run_evaluation.py
if __name__ == "__main__":
    results = evaluate_rag(chain, retriever, "./assets/evaluation_questions.json")
    scores = results["scores"]
    
    # Fail CI if quality drops below threshold
    assert scores["faithfulness"] >= 0.85, f"Faithfulness too low: {scores['faithfulness']}"
    assert scores["context_precision"] >= 0.75, f"Precision too low: {scores['context_precision']}"
    
    generate_evaluation_report(results)
    print("✅ All quality gates passed")
```
