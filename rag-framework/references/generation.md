# Answer Generation

Feed retrieved context and user queries to an LLM to produce grounded, cited answers.

## Prompt Design

The prompt template is the most critical component. It must clearly separate context from the question and instruct the LLM on how to use the context.

### Standard RAG Prompt

```python
RAG_PROMPT = """Answer the question based only on the provided context. 
If the context doesn't contain enough information, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""
```

### Cited RAG Prompt

```python
CITED_RAG_PROMPT = """Answer the question using only the provided sources. 
Cite sources using [Source N] notation after each claim.
If no sources support the answer, say "I don't have enough information."

Sources:
{sources}

Question: {question}

Answer (with citations):"""

def format_sources(docs):
    """Format retrieved documents as numbered sources."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        parts.append(f"[Source {i}] ({source}):\n{doc.page_content}")
    return "\n\n".join(parts)
```

## LangChain Generation

```python
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Simple chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": PromptTemplate.from_template(RAG_PROMPT),
    },
)
result = qa_chain.invoke({"query": "What is the refund policy?"})
print(result["result"])
print(result["source_documents"])
```

### LCEL (LangChain Expression Language)

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

answer = rag_chain.invoke("What is the refund policy?")
```

### Streaming

```python
for chunk in rag_chain.stream("What is the refund policy?"):
    print(chunk, end="", flush=True)
```

## LlamaIndex Generation

```python
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI

Settings.llm = OpenAI(model="gpt-4o", temperature=0)

# Query engine with custom prompt
from llama_index.core import PromptTemplate

query_engine = index.as_query_engine(
    similarity_top_k=4,
    text_qa_template=PromptTemplate(RAG_PROMPT),
)
response = query_engine.query("What is the refund policy?")
print(response.response)
print(response.source_nodes)
```

## Context Window Management

```python
def fit_context(docs, max_tokens=6000, tokenizer_name="cl100k_base"):
    """Truncate context to fit within token budget."""
    import tiktoken
    enc = tiktoken.get_encoding(tokenizer_name)
    
    fitted = []
    total = 0
    for doc in docs:
        tokens = len(enc.encode(doc.page_content))
        if total + tokens > max_tokens:
            break
        fitted.append(doc)
        total += tokens
    return fitted
```

## Conversational RAG (Chat with Memory)

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=5,  # Keep last 5 exchanges
)

conv_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    memory=memory,
    return_source_documents=True,
)
result = conv_chain.invoke({"question": "What is the refund policy?"})
```

## Guardrails

```python
def validate_answer(answer: str, context: str) -> str:
    """Basic guardrail: flag answers that may hallucinate."""
    if "I don't have enough information" in answer:
        return answer  # Appropriate abstention
    # Check that answer references concepts in context
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())
    overlap = len(answer_words & context_words) / max(len(answer_words), 1)
    if overlap < 0.1:
        return f"⚠️ Low grounding detected. {answer}"
    return answer
```

## Checklist

- [ ] Prompt template instructs LLM to use only provided context
- [ ] Abstention behavior defined for insufficient context
- [ ] Source attribution / citations implemented
- [ ] Context window budget calculated and enforced
- [ ] Temperature set to 0 for factual tasks
- [ ] Streaming configured for user-facing applications
- [ ] Conversation memory added if multi-turn needed
