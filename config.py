import os
from pathlib import Path
from dotenv import load_dotenv

# Find the workspace root
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file
load_dotenv(dotenv_path=BASE_DIR / ".env")

class Config:
    """Application configuration loaded from environment variables."""
    
    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
    
    # Model Configurations
    GROQ_MODEL_NAME: str = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile").strip()
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2").strip()
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()
    
    # Path Configurations
    CHROMA_DB_DIR: str = os.getenv("CHROMA_DB_DIR", "vector_db").strip()
    UPLOADS_DIR: str = os.getenv("UPLOADS_DIR", "uploads").strip()
    
    # Pipeline Parameters
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "4"))
    
    @classmethod
    def get_absolute_path(cls, path_str: str) -> Path:
        """Helper to get absolute path from a potentially relative path config."""
        path = Path(path_str)
        if not path.is_absolute():
            return BASE_DIR / path
        return path

    @classmethod
    def validate(cls) -> bool:
        """Validates if essential configurations are present.

        Returns:
            bool: True if configuration is valid, False otherwise.
        """
        if not cls.GROQ_API_KEY:
            return False
        return True

