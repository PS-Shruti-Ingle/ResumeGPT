from langchain_core.embeddings import Embeddings
from config import Config
from utils.logger import setup_logger

logger = setup_logger("embeddings")

def get_embeddings_model() -> Embeddings:
    """Initializes and returns the Embeddings model using configurations.

    Returns:
        Embeddings: The instantiated LangChain embeddings model.
    """
    if Config.OPENAI_API_KEY:
        from langchain_openai import OpenAIEmbeddings
        logger.info(f"Initializing OpenAIEmbeddings with model: {Config.OPENAI_EMBEDDING_MODEL}")
        try:
            return OpenAIEmbeddings(
                model=Config.OPENAI_EMBEDDING_MODEL,
                openai_api_key=Config.OPENAI_API_KEY
            )
        except Exception as e:
            logger.exception(f"Failed to initialize OpenAIEmbeddings: {e}")
            raise
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        logger.info(f"Initializing HuggingFaceEmbeddings with model: {Config.EMBEDDING_MODEL_NAME}")
        try:
            return HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL_NAME,
                model_kwargs={'device': 'cpu'}
            )
        except Exception as e:
            logger.exception(f"Failed to initialize HuggingFaceEmbeddings: {e}")
            raise
