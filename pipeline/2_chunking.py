from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logger import setup_logger

logger = setup_logger("chunking")

def chunk_documents(
    documents: List[Document], 
    chunk_size: int = 500, 
    chunk_overlap: int = 100
) -> List[Document]:
    """Splits a list of documents into chunks using RecursiveCharacterTextSplitter.

    Args:
        documents: The input list of documents to chunk.
        chunk_size: Maximum size of each chunk (characters). Defaults to 500.
        chunk_overlap: Number of characters to overlap between chunks. Defaults to 100.

    Returns:
        List[Document]: The chunked documents.
    """
    logger.info(f"Splitting {len(documents)} documents with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True  # useful for tracing exact positions if needed
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split completed. Created {len(chunks)} chunks.")
    return chunks
