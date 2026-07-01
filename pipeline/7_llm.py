from langchain_groq import ChatGroq
from config import Config
from utils.logger import setup_logger

logger = setup_logger("llm")

def get_llm(model_name: str = None, temperature: float = 0.0) -> ChatGroq:
    """Initializes and returns the Groq chat language model.

    Args:
        model_name: The Groq chat model to use. Defaults to Config.GROQ_MODEL_NAME.
        temperature: Temperature setting for LLM response generation. Defaults to 0.0.

    Returns:
        ChatGroq: The initialized LangChain ChatGroq object.
        
    Raises:
        ValueError: If the Groq API key is missing.
    """
    if not Config.GROQ_API_KEY:
        logger.error("Groq API Key is missing from config.")
        raise ValueError(
            "Groq API Key is missing. Please configure GROQ_API_KEY in your .env file."
        )
        
    active_model = model_name or Config.GROQ_MODEL_NAME
    logger.info(f"Initializing ChatGroq with model={active_model}, temperature={temperature}")
    try:
        return ChatGroq(
            model=active_model,
            temperature=temperature,
            groq_api_key=Config.GROQ_API_KEY
        )
    except Exception as e:
        logger.exception(f"Failed to initialize ChatGroq: {e}")
        raise
