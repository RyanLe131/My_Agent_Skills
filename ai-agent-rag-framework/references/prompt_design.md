# Prompt Design for RAG

Best practices for designing prompts that produce grounded, cited, and accurate answers from retrieved context.

## Core Principles

1. **Ground in context**: Explicitly instruct the LLM to use only provided context
2. **Enable abstention**: LLM must say "I don't know" when context is insufficient
3. **Require citations**: Link claims back to specific sources
4. **Control format**: Define expected output structure
5. **Minimize hallucination**: Set temperature to 0, constrain the answer scope

## Prompt Templates

### Basic RAG Prompt

```
Answer the question based only on the provided context.
If the context doesn't contain enough information, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:
```

### Cited RAG Prompt

```
You are a helpful assistant. Answer the question using ONLY the provided sources.
After each claim, cite the source using [Source N] notation.
If no sources support an answer, say "I don't have enough information."

Sources:
{sources}

Question: {question}

Answer (with citations):
```

### Structured Output Prompt

```
You are a knowledge assistant. Answer the question using only the context below.

Context:
{context}

Question: {question}

Respond in this format:
**Answer**: [Your answer here]
**Confidence**: [High/Medium/Low based on context support]
**Sources**: [List source documents used]
**Limitations**: [What the context doesn't cover]
```

### Multi-Turn Conversational Prompt

```
You are a helpful assistant with access to a knowledge base.
Use the conversation history and retrieved context to answer.
If the context doesn't help, say so — don't make up information.

Chat History:
{chat_history}

Retrieved Context:
{context}

Current Question: {question}

Answer:
```

## System Prompt Design

The system prompt sets the agent's behavior for the entire conversation:

```
You are an expert assistant for [DOMAIN].
Your role is to answer questions accurately using retrieved documents.

Rules:
1. Only use information from the provided context
2. If the context is insufficient, say "I don't have enough information"
3. Cite sources using [Source: filename] after each key claim
4. Be concise but thorough
5. If asked about topics outside your knowledge base, redirect politely
```

## Context Formatting

How you format retrieved documents matters:

```python
# Numbered sources (best for citation)
def format_as_sources(docs):
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        parts.append(f"[Source {i}] ({source}):\n{doc.page_content}")
    return "\n\n".join(parts)

# Separator-delimited (simple, clear)
def format_with_separators(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)
```

## Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| No grounding instruction | LLM uses parametric knowledge | Add "only use provided context" |
| No abstention | LLM hallucinates when context is weak | Add "say I don't know" instruction |
| Too much context | Exceeds context window or dilutes signal | Limit to top-k, truncate |
| Vague system prompt | Inconsistent behavior | Be specific about role and rules |
| No output format | Unparseable responses | Define expected structure |

## Temperature Settings

| Use Case | Temperature | Rationale |
|----------|-------------|-----------|
| Factual Q&A | 0 | Deterministic, grounded |
| Summarization | 0–0.3 | Slight flexibility in phrasing |
| Creative writing | 0.7–1.0 | Not typical for RAG |
| Code generation | 0 | Precision matters |
