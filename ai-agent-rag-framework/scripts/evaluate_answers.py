"""
RAG Evaluation Script
=====================
Measure retrieval and generation quality using RAGAS metrics.

Usage:
    from scripts.evaluate_answers import evaluate_rag, quick_evaluate

    results = evaluate_rag(rag_chain, retriever, test_data)
    score = quick_evaluate(answer="...", context="...", question="...")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_test_data(path: str) -> list[dict]:
    """Load evaluation questions from a JSON file.

    Expected format: [{"question": "...", "ground_truth": "..."}, ...]
    """
    with open(path) as f:
        data = json.load(f)
    logger.info("Loaded %d test questions from %s", len(data), path)
    return data


def evaluate_rag(
    rag_chain,
    retriever,
    test_data: list[dict] | str,
    metrics: list[str] | None = None,
) -> dict:
    """Run full RAGAS evaluation on a RAG pipeline.

    Args:
        rag_chain: The RAG chain to evaluate (must accept query string).
        retriever: The retriever to get context documents.
        test_data: List of {"question", "ground_truth"} dicts, or path to JSON.
        metrics: Which metrics to compute. Default: all.

    Returns:
        Dictionary of metric scores and per-sample results.
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas import EvaluationDataset, SingleTurnSample

    if isinstance(test_data, str):
        test_data = load_test_data(test_data)

    samples = []
    for item in test_data:
        question = item["question"]
        result = rag_chain.invoke(question)
        retrieved_docs = retriever.invoke(question)

        # Handle chain output formats
        answer = result if isinstance(result, str) else result.get("result", str(result))

        samples.append(SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=[doc.page_content for doc in retrieved_docs],
            reference=item.get("ground_truth", ""),
        ))

    dataset = EvaluationDataset(samples=samples)

    all_metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    results = evaluate(dataset=dataset, metrics=all_metrics)

    logger.info("RAGAS evaluation complete: %s", results)
    return {
        "scores": dict(results),
        "dataframe": results.to_pandas(),
    }


def quick_evaluate(
    answer: str,
    context: str,
    question: str,
    model: str = "gpt-4o",
) -> dict:
    """Quick LLM-as-judge evaluation for a single Q&A pair.

    Args:
        answer: The generated answer.
        context: The retrieved context used.
        question: The original question.
        model: LLM to use as evaluator.

    Returns:
        Dict with faithfulness and relevance scores (0.0–1.0).
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model, temperature=0)

    faithfulness_prompt = (
        "You are an evaluation judge. Given the context and answer below, "
        "score how well the answer is supported by the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:\n{answer}\n\n"
        "Score from 0.0 (not supported) to 1.0 (fully supported). "
        "Respond with ONLY a number."
    )

    relevance_prompt = (
        "You are an evaluation judge. Given the question and answer below, "
        "score how well the answer addresses the question.\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        "Score from 0.0 (irrelevant) to 1.0 (perfectly relevant). "
        "Respond with ONLY a number."
    )

    faith_score = float(llm.invoke(faithfulness_prompt).content.strip())
    rel_score = float(llm.invoke(relevance_prompt).content.strip())

    return {
        "faithfulness": round(faith_score, 2),
        "relevance": round(rel_score, 2),
    }


def generate_evaluation_report(results: dict, output_path: str = "./eval_report.md"):
    """Generate a markdown evaluation report."""
    scores = results["scores"]
    report = ["# RAG Evaluation Report\n"]
    report.append("## Overall Scores\n")
    report.append("| Metric | Score |")
    report.append("|--------|-------|")
    for metric, score in scores.items():
        status = "✅" if score >= 0.8 else "⚠️" if score >= 0.6 else "❌"
        report.append(f"| {metric} | {score:.3f} {status} |")

    report.append("\n## Diagnosis\n")
    if scores.get("context_precision", 1) < 0.8:
        report.append("- **Low Context Precision**: Retrieved docs may be irrelevant → improve chunking or add re-ranking\n")
    if scores.get("context_recall", 1) < 0.8:
        report.append("- **Low Context Recall**: Missing relevant docs → increase top-k or try hybrid search\n")
    if scores.get("faithfulness", 1) < 0.9:
        report.append("- **Low Faithfulness**: Answer contains unsupported claims → strengthen prompt grounding\n")
    if scores.get("answer_relevancy", 1) < 0.8:
        report.append("- **Low Answer Relevance**: Answer doesn't address question → review prompt template\n")

    content = "\n".join(report)
    Path(output_path).write_text(content)
    logger.info("Evaluation report saved to %s", output_path)
    return content
