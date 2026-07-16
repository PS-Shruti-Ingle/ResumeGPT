from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from utils.logger import setup_logger

logger = setup_logger("document_loader")


def _extract_with_pypdf(path: Path) -> List[Document]:
    """Primary extraction using PyPDFLoader."""
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(str(path.resolve()))
    docs = loader.load()
    logger.info(f"PyPDFLoader extracted {len(docs)} pages.")
    return docs


def _extract_with_pdfplumber(path: Path) -> List[Document]:
    """Fallback extraction using pdfplumber — better with formatted/columnar PDFs."""
    import pdfplumber
    from langchain_core.documents import Document

    docs = []
    with pdfplumber.open(str(path.resolve())) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            # Also try table extraction and append table text
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    if row:
                        row_text = " | ".join(cell or "" for cell in row if cell)
                        if row_text.strip():
                            text += "\n" + row_text
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": str(path), "page": page_num},
                )
            )
    logger.info(f"pdfplumber extracted {len(docs)} pages.")
    return docs


def load_pdf(file_path) -> List[Document]:
    """Loads a PDF resume using PyPDFLoader with pdfplumber as fallback.

    Tries PyPDF first. If the extracted text looks empty or garbled
    (< 100 meaningful characters per page on average), falls back to pdfplumber
    which handles formatted/two-column and ATS-style resumes much better.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List[Document]: One Document per page with full text.

    Raises:
        FileNotFoundError: If the PDF doesn't exist.
        ValueError: If the PDF cannot be parsed by either method.
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"PDF file not found at {path}")

    if path.suffix.lower() != ".pdf":
        logger.error(f"Invalid extension: {path.suffix}")
        raise ValueError(f"Uploaded file must be a PDF, got: {path.suffix}")

    # ── Attempt 1: PyPDF ──────────────────────────────────────────────────────
    docs: Optional[List[Document]] = None
    try:
        docs = _extract_with_pypdf(path)
    except Exception as e:
        logger.warning(f"PyPDFLoader failed: {e}")

    # Check quality — if average page text is too short, fallback
    avg_len = (
        sum(len(d.page_content) for d in docs) / len(docs) if docs else 0
    )
    logger.info(f"PyPDF avg chars per page: {avg_len:.0f}")

    if avg_len < 150:
        logger.warning(
            "PyPDF extraction appears low-quality. Trying pdfplumber as fallback..."
        )
        try:
            fallback_docs = _extract_with_pdfplumber(path)
            fallback_avg = (
                sum(len(d.page_content) for d in fallback_docs) / len(fallback_docs)
                if fallback_docs
                else 0
            )
            logger.info(f"pdfplumber avg chars per page: {fallback_avg:.0f}")
            if fallback_avg > avg_len:
                docs = fallback_docs
                logger.info("Using pdfplumber output (better quality).")
        except Exception as e:
            logger.warning(f"pdfplumber fallback also failed: {e}")

    if not docs:
        raise ValueError("Could not extract text from PDF using any available method.")

    # Log a preview of what was extracted (for debugging)
    total_chars = sum(len(d.page_content) for d in docs)
    logger.info(
        f"Final extraction: {len(docs)} pages, {total_chars} total characters."
    )
    for i, doc in enumerate(docs):
        safe_preview = repr(doc.page_content[:200]).encode('ascii', 'replace').decode('ascii')
        logger.info(f"  Page {i+1} preview: {safe_preview}")

    return docs
