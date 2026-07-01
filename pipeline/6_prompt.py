from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from utils.logger import setup_logger

logger = setup_logger("prompt")

def get_rag_prompt() -> ChatPromptTemplate:
    """Creates and returns the ChatPromptTemplate for the RAG chain.

    Ensures the LLM answers strictly based on context and avoids hallucinations.
    
    Returns:
        ChatPromptTemplate: The prompt template.
    """
    logger.info("Initializing RAG prompt template...")
    
    system_prompt = (
        "You are a precise, professional AI assistant specialized in analyzing resumes.\n"
        "Your task is to answer the user's question using ONLY the provided resume context.\n"
        "Do not use any external knowledge. Do not assume or extrapolate beyond what is explicitly written.\n\n"
        "STRICT RULES:\n"
        "1. Base your answer strictly on the provided context.\n"
        "2. If the context does not contain the information to answer the question, or if the answer is not "
        "explicitly stated in the context, you MUST respond with this exact phrase: "
        "\"I couldn't find this information in the uploaded resume.\"\n"
        "3. When the answer is not found, do not add any additional comments, explanations, or text. "
        "Just return that exact sentence.\n\n"
        "Provided Resume Context:\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    return prompt
