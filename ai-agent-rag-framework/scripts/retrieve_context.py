"""
Context Retrieval Script
========================
Query the vector store to find the most relevant chunks for a user question.
Supports dense, hybrid, MMR, and re-ranked retrieval.

Usage:
    from scripts.retrieve_context import retrieve, create_rag_chain

    docs = retrieve(vectorstore, "What is the refund policy?")
    chain = create_rag_chain(vectorstore)
    answer = chain.invoke("What is the refund policy?")
"""

from __future__ import annotations

import logging
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def retrieve(
    vectorstore,
    query: str,
    search_type: str = "similarity",
    top_k: int = 5,
    score_threshold: float | None = None,
    filters: dict | None = None,
) -> list[Document]:
    """Retrieve relevant documents from the vector store.

    Args:
        vectorstore: LangChain VectorStore instance.
        query: User question or search query.
        search_type: "similarity", "mmr", or "similarity_score_threshold".
        top_k: Number of documents to retrieve.
        score_threshold: Minimum similarity score (for threshold search).
        filters: Metadata filters to apply.

    Returns:
        List of relevant Document objects.
    """
    search_kwargs = {"k": top_k}
    if filters:
        search_kwargs["filter"] = filters

    if search_type == "similarity_score_threshold" and score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold

    if search_type == "mmr":
        search_kwargs["fetch_k"] = top_k * 4
        search_kwargs["lambda_mult"] = 0.5

    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )
    docs = retriever.invoke(query)
    logger.info("Retrieved %d documents for query: %.80s...", len(docs), query)
    return docs


def create_rag_chain(
    vectorstore,
    model: str = "gpt-4o",
    top_k: int = 5,
    prompt_template: str | None = None,
):
    """Create a full RAG chain (retrieve → generate).

    Args:
        vectorstore: LangChain VectorStore instance.
        model: LLM model name.
        top_k: Number of documents to retrieve.
        prompt_template: Custom prompt template string.

    Returns:
        Runnable RAG chain.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    if prompt_template is None:
        prompt_template = (
            "Answer the question based only on the provided context. "
            "If the context doesn't contain enough information, say "
            "\"I don't have enough information to answer this question.\"\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(model=model, temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    def format_docs(docs: list[Document]) -> str:
        return "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for d in docs
        )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def create_conversational_chain(
    vectorstore,
    model: str = "gpt-4o",
    top_k: int = 5,
    memory_window: int = 5,
):
    """Create a conversational RAG chain with chat history.

    Args:
        vectorstore: LangChain VectorStore instance.
        model: LLM model name.
        top_k: Documents to retrieve per turn.
        memory_window: Number of past exchanges to retain.

    Returns:
        Conversational retrieval chain.
    """
    from langchain_openai import ChatOpenAI
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferWindowMemory

    llm = ChatOpenAI(model=model, temperature=0)
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=memory_window,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": top_k}),
        memory=memory,
        return_source_documents=True,
    )
    return chain
