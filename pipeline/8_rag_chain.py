from typing import List
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    Runnable,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from utils.logger import setup_logger

logger = setup_logger("rag_chain")


def _deduplicate_docs(docs: List[Document]) -> List[Document]:
    """Remove duplicate chunks by content hash, keeping order (full-doc first)."""
    seen = set()
    unique = []
    for doc in docs:
        key = hash(doc.page_content.strip())
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def format_docs(docs: List[Document]) -> str:
    """Formats retrieved documents into a single context string for the LLM.

    Full-document chunks (chunk_type == 'full_document') are placed at the TOP
    of the context so the LLM always sees the complete resume first.
    Section-level chunks follow for additional detail.

    Args:
        docs: Retrieved Document chunks.

    Returns:
        str: Formatted context string.
    """
    if not docs:
        return ""

    # Sort: full-document chunk first, then the rest
    full_docs = [d for d in docs if d.metadata.get("chunk_type") == "full_document"]
    section_docs = [d for d in docs if d.metadata.get("chunk_type") != "full_document"]

    ordered = full_docs + section_docs
    ordered = _deduplicate_docs(ordered)

    parts = []
    for i, doc in enumerate(ordered):
        label = (
            "[ FULL RESUME ]"
            if doc.metadata.get("chunk_type") == "full_document"
            else f"[ Section {i} | Page {doc.metadata.get('page', 0) + 1} ]"
        )
        parts.append(f"{label}\n{doc.page_content.strip()}")

    context = "\n\n" + "\n\n---\n\n".join(parts) + "\n"
    logger.debug(f"Context length: {len(context)} chars, {len(ordered)} chunks.")
    return context


def create_rag_chain(
    retriever: BaseRetriever,
    prompt: ChatPromptTemplate,
    llm: BaseChatModel,
) -> Runnable:
    """Assembles and returns the complete RAG chain using LCEL.

    Input:  {"question": str, "chat_history": List[BaseMessage]}
    Output: {"answer": str, "context": List[Document]}

    Args:
        retriever: Hybrid or BM25 retriever from stage 5.
        prompt: RAG prompt template from stage 6.
        llm: ChatGroq language model from stage 7.

    Returns:
        Runnable: The fully assembled LangChain pipeline.
    """
    logger.info("Assembling RAG pipeline (LCEL)...")

    try:
        def retrieve_context(inputs: dict) -> List[Document]:
            """Retrieves and deduplicates relevant chunks for the question."""
            docs = retriever.invoke(inputs["question"])
            docs = _deduplicate_docs(docs)
            logger.info(
                f"Retrieved {len(docs)} unique chunks for: {inputs['question'][:60]}"
            )
            return docs

        # Step 1: Retrieve docs + pass through question and chat_history
        retrieve_and_prepare = RunnableParallel(
            {
                "context": RunnableLambda(retrieve_context),
                "question": RunnableLambda(lambda x: x["question"]),
                "chat_history": RunnableLambda(lambda x: x["chat_history"]),
            }
        )

        # Step 2: Format context → run prompt → run LLM → parse output
        generate_answer = (
            RunnableParallel(
                {
                    "context": RunnableLambda(lambda x: format_docs(x["context"])),
                    "question": RunnableLambda(lambda x: x["question"]),
                    "chat_history": RunnableLambda(lambda x: x["chat_history"]),
                }
            )
            | prompt
            | llm
            | StrOutputParser()
        )

        # Step 3: Return answer + raw source docs
        chain = retrieve_and_prepare | RunnableParallel(
            {
                "answer": generate_answer,
                "context": RunnableLambda(lambda x: x["context"]),
            }
        )

        logger.info("RAG pipeline assembled successfully.")
        return chain

    except Exception as e:
        logger.exception(f"Failed to assemble RAG pipeline: {e}")
        raise
