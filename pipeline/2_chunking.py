from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logger import setup_logger

logger = setup_logger("chunking")

# Resume-optimised separators — try to split on section boundaries first,
# then paragraphs, then lines, then sentences, then words.
_RESUME_SEPARATORS = [
    "\n\n\n",   # large whitespace block (section gap)
    "\n\n",     # paragraph break
    "\n•",      # bullet list item
    "\n-",      # dash list item
    "\n–",      # en-dash list item
    "\n",       # single newline
    ". ",       # sentence boundary
    ", ",       # clause boundary
    " ",        # word boundary
    "",         # character boundary (last resort)
]


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Splits resume documents into well-sized chunks.

    Resume-specific design decisions:
    - chunk_size=1000  (default) — large enough to keep a full section together
      (e.g. "Experience at Company X" with bullets fits in one chunk).
    - chunk_overlap=200 — ensures that if a section boundary falls mid-chunk,
      the next chunk still has the section heading for context.
    - Custom separators favour semantic boundaries (section gaps, bullet points)
      over arbitrary character counts.
    - FULL-DOC PASS-THROUGH: If the entire document is ≤ 8 000 chars (fits in
      one LLM context window comfortably) it is returned as a single chunk so
      the LLM always sees the complete resume and never misses a field.

    Args:
        documents: List of Document objects (one per page from the loader).
        chunk_size: Max chunk size in characters.
        chunk_overlap: Character overlap between adjacent chunks.

    Returns:
        List[Document]: Chunked documents ready for indexing.
    """
    logger.info(
        f"Chunking {len(documents)} pages | chunk_size={chunk_size} | "
        f"chunk_overlap={chunk_overlap}"
    )

    # ── Full-document pass-through for short resumes ──────────────────────────
    total_chars = sum(len(d.page_content) for d in documents)
    logger.info(f"Total document characters: {total_chars}")

    if total_chars <= 8_000:
        # Resume fits comfortably in the LLM context — return whole-doc chunk
        # PLUS per-page chunks so BM25 still has granular matches.
        logger.info(
            "Short resume detected — adding a full-document chunk for complete context."
        )
        # Create a merged "full resume" document
        full_text = "\n\n".join(d.page_content for d in documents)
        full_doc = Document(
            page_content=full_text,
            metadata={
                "source": documents[0].metadata.get("source", "resume"),
                "page": 0,
                "chunk_type": "full_document",
            },
        )
        # Also keep normal per-section chunks for precise retrieval
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=_RESUME_SEPARATORS,
            length_function=len,
            add_start_index=True,
        )
        section_chunks = splitter.split_documents(documents)
        all_chunks = [full_doc] + section_chunks
        logger.info(
            f"Final chunks: 1 full-doc + {len(section_chunks)} section chunks = "
            f"{len(all_chunks)} total."
        )
        return all_chunks

    # ── Standard chunking for longer documents ────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_RESUME_SEPARATORS,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks.")
    return chunks
