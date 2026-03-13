# Answer Format

When generating answers from retrieved context, follow this structure:

## Standard Answer

```
**Answer**: [Direct, concise answer to the question]

**Details**: [Supporting details from the context, if needed]

**Sources**: [List of source documents used]
```

## Extended Answer (for complex questions)

```
**Answer**: [Direct answer in 1-2 sentences]

**Explanation**:
[Detailed explanation drawing from multiple sources]
- Point 1 [Source: doc_name]
- Point 2 [Source: doc_name]

**Confidence**: [High / Medium / Low]
- High: Multiple sources directly support the answer
- Medium: One source supports, or answer requires inference
- Low: Partial support only — flag for human review

**Sources Used**:
1. [document_name] — [relevant section]
2. [document_name] — [relevant section]

**Limitations**:
[What the knowledge base doesn't cover about this topic]
```

## When You Don't Know

```
I don't have enough information in my knowledge base to answer this question.

**What I found**: [Brief summary of related but insufficient context]
**Suggestion**: [Where the user might find the answer]
```
