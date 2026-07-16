import re
from langchain_groq import ChatGroq
from config import Config
from utils.logger import setup_logger

logger = setup_logger("llm")


def get_llm(model_name: str = None, temperature: float = 0.0) -> ChatGroq:
    """Initialises and returns the Groq chat language model.

    Default model: llama-3.3-70b-versatile. Temperature is set to 0.0 for 
    maximum factual accuracy when extracting from resumes.

    Args:
        model_name: Override model name. Defaults to Config.GROQ_MODEL_NAME.
        temperature: Sampling temperature. Defaults to 0.0.

    Returns:
        ChatGroq: The configured LangChain ChatGroq client.

    Raises:
        ValueError: If GROQ_API_KEY is not set.
    """
    if not Config.GROQ_API_KEY:
        logger.error("Groq API Key is missing.")
        raise ValueError(
            "Groq API Key is missing. Please configure GROQ_API_KEY in your .env file."
        )

    active_model = model_name or Config.GROQ_MODEL_NAME
    logger.info(f"Initialising ChatGroq | model={active_model} | temperature={temperature}")

    try:
        return ChatGroq(
            model=active_model,
            temperature=temperature,
            groq_api_key=Config.GROQ_API_KEY,
        )
    except Exception as e:
        logger.exception(f"Failed to initialise ChatGroq: {e}")
        raise


def strip_thinking_tags(text: str) -> str:
    """Strips <think>...</think> reasoning tokens produced by DeepSeek-R1 models.

    Reasoning models like deepseek-r1-distill-llama-70b prefix their response
    with an internal monologue wrapped in <think> tags. We remove these before
    displaying to the user.

    Args:
        text: Raw LLM output potentially containing <think>...</think> blocks.

    Returns:
        str: Clean answer with thinking blocks removed, stripped of extra whitespace.
    """
    # Remove <think>...</think> blocks (may span multiple lines)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()
