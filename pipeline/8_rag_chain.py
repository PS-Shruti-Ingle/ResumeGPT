from typing import List
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, Runnable, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from utils.logger import setup_logger

logger = setup_logger("rag_chain")

def format_docs(docs: List[Document]) -> str:
    """Helper function to format document list into a single context string.

    Args:
        docs: List of Document chunks retrieved.

    Returns:
        str: Formatted context string.
    """
    return "\n\n".join([doc.page_content for doc in docs])

def create_rag_chain(
    retriever: BaseRetriever,
    prompt: ChatPromptTemplate,
    llm: BaseChatModel
) -> Runnable:
    """Assembles and returns the complete RAG chain using LCEL.

    The assembled chain takes a dictionary input:
        {"question": str, "chat_history": List[BaseMessage]}
    And returns a dictionary output:
        {"answer": str, "context": List[Document]}

    Args:
        retriever: Configured retriever (BM25 or vector store).
        prompt: RAG prompt template.
        llm: ChatGroq language model.

    Returns:
        Runnable: The combined LangChain executable pipeline.
    """
    logger.info("Assembling RAG pipeline elements using LCEL...")
    
    try:
        # Use retriever.invoke() — compatible with both BM25Retriever and VectorStoreRetriever
        # in LangChain 1.x where get_relevant_documents is removed from public API.
        def retrieve_context(inputs: dict) -> List[Document]:
            return retriever.invoke(inputs["question"])

        # Step 1: Retrieve docs + pass through question and chat_history
        retrieve_and_prepare = RunnableParallel(
            {
                "context": RunnableLambda(retrieve_context),
                "question": RunnableLambda(lambda x: x["question"]),
                "chat_history": RunnableLambda(lambda x: x["chat_history"])
            }
        )

        # Step 2: Format context string for the prompt, then generate answer
        generate_answer = (
            RunnableParallel(
                {
                    "context": RunnableLambda(lambda x: format_docs(x["context"])),
                    "question": RunnableLambda(lambda x: x["question"]),
                    "chat_history": RunnableLambda(lambda x: x["chat_history"])
                }
            )
            | prompt
            | llm
            | StrOutputParser()
        )

        # Step 3: Final chain — returns both generated answer and the raw source docs
        chain = retrieve_and_prepare | RunnableParallel(
            {
                "answer": generate_answer,
                "context": RunnableLambda(lambda x: x["context"])
            }
        )

        logger.info("RAG pipeline successfully assembled.")
        return chain
    except Exception as e:
        logger.exception(f"Failed to assemble RAG pipeline: {e}")
        raise
