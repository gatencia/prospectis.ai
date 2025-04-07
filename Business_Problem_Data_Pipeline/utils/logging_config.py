"""Logging configuration for business problems pipeline."""

import logging
import logging.handlers
import os
from Business_Problem_Data_Pipeline.config import LOG_FILE

def setup_logging(name, level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        name (str): Logger name
        level (int): Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=10485760, backupCount=5
    )
    file_handler.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger