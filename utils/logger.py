import logging
import sys

def setup_logger(name: str = "resume_rag") -> logging.Logger:
    """Sets up a standardized logger for the application.

    Args:
        name: Name of the logger, defaults to 'resume_rag'

    Returns:
        logging.Logger configured to write to stdout.
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if setup is called multiple times
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create console handler and set level to info
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Add formatter to handler
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        
    return logger
