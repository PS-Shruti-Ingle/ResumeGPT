from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from utils.logger import setup_logger

logger = setup_logger("retriever")


def get_retriever(chunks: List[Document], k: int = 6) -> BaseRetriever:
    """Builds a Hybrid Retriever: BM25 (keyword) + Chroma (semantic) combined
    via LangChain's EnsembleRetriever.

    Design notes
    ────────────
    • BM25  (weight 0.4) — excels at exact token matches: names, email addresses,
      phone numbers, skill acronyms (Python, SQL, AWS …).
    • Semantic (weight 0.6) — excels at conceptual/paraphrased queries: "leadership
      skills", "data science background", "what languages does he speak?".
    • k is intentionally higher (default 6) to cast a wider net over a small resume.
    • A full-document chunk (injected by the chunker for short resumes) is always
      included: it ranks highly for any broad query and guarantees the LLM sees
      the complete resume for simple factual questions (name, contact, languages).
    • Falls back gracefully to BM25-only if the embedding stack is unavailable.

    Args:
        chunks: Document chunks from the chunking stage (may include a full-doc chunk).
        k: Number of top documents to return per sub-retriever. Defaults to 6.

    Returns:
        BaseRetriever: EnsembleRetriever (hybrid) or BM25Retriever (fallback).
    """
    logger.info(
        f"Building hybrid retriever from {len(chunks)} chunks (k={k} per sub-retriever)"
    )

    # ── 1. BM25 – keyword retriever ──────────────────────────────────────────
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k
    logger.info("BM25Retriever initialised.")

    # ── 2. Semantic – vector retriever (HuggingFace + Chroma in-memory) ──────
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma

        # EnsembleRetriever location varies by LangChain version:
        # - LangChain 1.3.x → langchain_classic.retrievers.ensemble
        # - Older versions  → langchain.retrievers or langchain_community.retrievers
        _ensemble_cls = None
        for _mod_path, _attr in [
            ("langchain_classic.retrievers.ensemble", "EnsembleRetriever"),
            ("langchain_classic.retrievers", "EnsembleRetriever"),
            ("langchain.retrievers.ensemble", "EnsembleRetriever"),
            ("langchain.retrievers", "EnsembleRetriever"),
            ("langchain_community.retrievers", "EnsembleRetriever"),
        ]:
            try:
                import importlib as _il
                _m = _il.import_module(_mod_path)
                _ensemble_cls = getattr(_m, _attr)
                logger.info(f"EnsembleRetriever found at: {_mod_path}")
                break
            except (ImportError, AttributeError):
                continue

        if _ensemble_cls is None:
            raise ImportError("EnsembleRetriever not found in any LangChain package.")

        import tempfile

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("HuggingFace embedding model loaded.")

        # Temp dir — isolated per session, never conflicts with old resume data
        tmp_dir = tempfile.mkdtemp(prefix="resume_chroma_")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=tmp_dir,
        )
        semantic_retriever = vector_store.as_retriever(
            search_type="mmr",           # Maximal Marginal Relevance — diverse results
            search_kwargs={"k": k, "fetch_k": k * 3},
        )
        logger.info("Chroma MMR semantic retriever initialised.")

        # ── 3. Ensemble: merge BM25 + Semantic scores ────────────────────────
        hybrid_retriever = _ensemble_cls(
            retrievers=[bm25_retriever, semantic_retriever],
            weights=[0.4, 0.6],
        )
        logger.info("Hybrid EnsembleRetriever ready.")
        return hybrid_retriever

    except Exception as e:
        logger.warning(
            f"Semantic retriever failed — using BM25 only. Reason: {e}"
        )
        return bm25_retriever
