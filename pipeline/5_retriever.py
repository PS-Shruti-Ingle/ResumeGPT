from typing import List
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from utils.logger import setup_logger

logger = setup_logger("retriever")

def get_retriever(chunks: List[Document], k: int = 4) -> BM25Retriever:
    """Initializes and returns a BM25Retriever from a list of document chunks.

    BM25 is a pure-Python keyword-based retrieval algorithm that requires no
    ML models, neural networks, or native DLL dependencies. It is an excellent
    choice for structured text like resumes.

    Args:
        chunks: List of Document chunks from the PDF.
        k: The number of top documents to retrieve. Defaults to 4.

    Returns:
        BM25Retriever: The configured retriever.
    """
    logger.info(f"Creating BM25Retriever from {len(chunks)} chunks, k={k}")
    try:
        retriever = BM25Retriever.from_documents(chunks)
        retriever.k = k
        logger.info("BM25Retriever created successfully.")
        return retriever
    except Exception as e:
        logger.exception(f"Error creating BM25Retriever: {e}")
        raise
