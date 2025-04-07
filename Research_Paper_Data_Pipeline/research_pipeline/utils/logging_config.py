"""
Logging configuration for the research pipeline.
"""

import logging
import sys
from pathlib import Path
from loguru import logger

from research_pipeline.config import LOG_LEVEL_NUM, LOG_FILE


def setup_logging():
    """
    Set up logging configuration using loguru.
    """
    # Remove default logger
    logger.remove()
    
    # Add console logger with appropriate level
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=LOG_LEVEL_NUM,
        colorize=True
    )
    
    # Add file logger
    log_file = Path(LOG_FILE)
    log_file.parent.mkdir(exist_ok=True)
    
    logger.add(
        LOG_FILE,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=LOG_LEVEL_NUM,
        rotation="10 MB",
        retention="1 month"
    )
    
    # Intercept standard library logging
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            
            # Find caller from where this record was issued
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            
            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
    
    # Setup standard library logging to use loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    logger.info(f"Logging initialized with level {logging.getLevelName(LOG_LEVEL_NUM)}")


def get_logger(name: str):
    """
    Get a logger with the given name.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    return logger.bind(name=name)