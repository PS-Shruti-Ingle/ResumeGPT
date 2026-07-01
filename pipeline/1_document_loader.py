from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from utils.logger import setup_logger

logger = setup_logger("document_loader")

def load_pdf(file_path: str | Path) -> List[Document]:
    """Loads a PDF file and returns a list of Document objects.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List[Document]: List of extracted Document chunks (usually one per page).
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a PDF or is corrupted.
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"PDF file not found at {path}")
        
    if path.suffix.lower() != ".pdf":
        logger.error(f"Invalid file extension: {path.suffix}")
        raise ValueError(f"Uploaded file must be a PDF, got: {path.suffix}")
        
    try:
        logger.info(f"Loading PDF from: {path.resolve()}")
        loader = PyPDFLoader(str(path.resolve()))
        documents = loader.load()
        logger.info(f"Successfully loaded PDF. Total pages: {len(documents)}")
        return documents
    except Exception as e:
        logger.exception(f"Error loading PDF file: {e}")
        raise ValueError(f"Could not load or parse PDF file. Details: {str(e)}")
