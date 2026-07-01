import shutil
import gc
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from utils.logger import setup_logger

logger = setup_logger("vector_store")

def clear_vector_store(persist_directory: str | Path) -> None:
    """Safely removes the persistent Chroma vector store directory.

    Args:
        persist_directory: Path to the Chroma database directory.
    """
    path = Path(persist_directory)
    if path.exists():
        logger.info(f"Clearing vector store at: {path.resolve()}")
        # Force garbage collection to release file handlers held by Chroma
        gc.collect()
        try:
            shutil.rmtree(path)
            logger.info("Successfully deleted vector store directory.")
        except Exception as e:
            logger.error(f"Failed to delete vector store directory: {e}")
            raise OSError(
                f"Could not clear vector store directory at {path.resolve()}. "
                f"It might be locked by another process. Details: {e}"
            )

def create_vector_store(
    documents: List[Document],
    embeddings: Embeddings,
    persist_directory: str | Path
) -> Chroma:
    """Creates a new Chroma vector store from documents, overwriting any existing store.

    Args:
        documents: List of Document objects to index.
        embeddings: The Embeddings model to use.
        persist_directory: Path to persist the Chroma database.

    Returns:
        Chroma: The initialized Chroma vector store.
    """
    path = Path(persist_directory)
    
    # Clean previous database if it exists
    if path.exists():
        clear_vector_store(path)
        
    logger.info(f"Creating new Chroma vector store at {path.resolve()} with {len(documents)} chunks...")
    try:
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(path.resolve())
        )
        
        # Call persist if the method exists (depends on LangChain/Chroma version)
        if hasattr(vector_store, "persist"):
            try:
                vector_store.persist()
            except Exception:
                pass
                
        logger.info("Chroma vector store successfully created and persisted.")
        return vector_store
    except Exception as e:
        logger.exception(f"Error creating vector store: {e}")
        raise ValueError(f"Failed to initialize vector database. Details: {e}")

def load_vector_store(
    embeddings: Embeddings,
    persist_directory: str | Path
) -> Chroma:
    """Loads an existing Chroma vector store from the persist directory.

    Args:
        embeddings: The Embeddings model to use.
        persist_directory: Path to the Chroma database directory.

    Returns:
        Chroma: The loaded Chroma vector store.
        
    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    path = Path(persist_directory)
    if not path.exists():
        logger.warning(f"Vector store directory does not exist: {path}")
        raise FileNotFoundError(f"No vector database found at {path}")
        
    logger.info(f"Loading existing Chroma vector store from {path.resolve()}")
    try:
        return Chroma(
            persist_directory=str(path.resolve()),
            embedding_function=embeddings
        )
    except Exception as e:
        logger.exception(f"Error loading vector store: {e}")
        raise ValueError(f"Failed to load vector database. Details: {e}")
