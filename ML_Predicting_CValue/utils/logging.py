"""
Logging utilities for Prospectis ML system.
Provides standardized logging configuration across all components.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
import json
from datetime import datetime
from typing import Optional, Dict, Any, Union

# Base log directory
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Log levels
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

# Standard log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_LOG_FORMAT = "[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s"

# Default log level
DEFAULT_LOG_LEVEL = logging.INFO


def setup_logger(name: str, 
                log_file: Optional[Union[str, Path]] = None,
                level: Union[int, str] = DEFAULT_LOG_LEVEL,
                log_format: str = DEFAULT_LOG_FORMAT,
                console: bool = True,
                rotating: bool = True,
                max_bytes: int = 10 * 1024 * 1024,  # 10MB
                backup_count: int = 5) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (or None to auto-generate)
        level: Log level (can be string 'debug', 'info', etc. or logging constant)
        log_format: Log format string
        console: Whether to log to console
        rotating: Whether to use rotating file handler
        max_bytes: Maximum file size for rotation
        backup_count: Number of backup files to keep
    
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Convert string log level to int if needed
    if isinstance(level, str):
        level = LOG_LEVELS.get(level.lower(), DEFAULT_LOG_LEVEL)
        
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if log_file is provided or auto-generate one
    if log_file is not None or name != "root":
        if log_file is None:
            # Auto-generate log filename from logger name
            safe_name = name.replace(".", "_").lower()
            log_file = LOG_DIR / f"{safe_name}_{datetime.now().strftime('%Y%m%d')}.log"
        
        log_file = Path(log_file)
        log_file.parent.mkdir(exist_ok=True, parents=True)
        
        if rotating:
            handler = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count
            )
        else:
            handler = logging.FileHandler(log_file)
            
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def setup_daily_logger(name: str,
                     log_dir: Optional[Union[str, Path]] = None,
                     level: Union[int, str] = DEFAULT_LOG_LEVEL,
                     log_format: str = DEFAULT_LOG_FORMAT,
                     console: bool = True,
                     backup_count: int = 30) -> logging.Logger:
    """
    Set up a logger with daily rotation.
    
    Args:
        name: Logger name
        log_dir: Directory for log files (or None for default)
        level: Log level
        log_format: Log format string
        console: Whether to log to console
        backup_count: Number of backup files to keep
    
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Convert string log level to int if needed
    if isinstance(level, str):
        level = LOG_LEVELS.get(level.lower(), DEFAULT_LOG_LEVEL)
        
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Set up log directory
    if log_dir is None:
        log_dir = LOG_DIR
    else:
        log_dir = Path(log_dir)
        log_dir.mkdir(exist_ok=True, parents=True)
    
    # Create daily rotating file handler
    safe_name = name.replace(".", "_").lower()
    log_file = log_dir / f"{safe_name}.log"
    
    handler = TimedRotatingFileHandler(
        log_file, when="midnight", interval=1, backupCount=backup_count
    )
    handler.suffix = "%Y%m%d"  # e.g., app.log.20231015
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def log_execution_time(logger: logging.Logger, start_time: float, operation: str) -> None:
    """
    Log the execution time of an operation.
    
    Args:
        logger: Logger to use
        start_time: Start time from time.time()
        operation: Description of the operation
    """
    import time
    elapsed = time.time() - start_time
    logger.info(f"{operation} completed in {elapsed:.2f} seconds")


def log_exception(logger: logging.Logger, e: Exception, context: str = "") -> None:
    """
    Log an exception with context.
    
    Args:
        logger: Logger to use
        e: Exception to log
        context: Context description
    """
    import traceback
    message = f"Exception occurred"
    if context:
        message += f" during {context}"
    logger.error(f"{message}: {str(e)}")
    logger.debug(traceback.format_exc())


def log_dict(logger: logging.Logger, data: Dict[str, Any], 
            message: str = "Data", level: str = "info") -> None:
    """
    Log a dictionary as JSON.
    
    Args:
        logger: Logger to use
        data: Dictionary to log
        message: Message prefix
        level: Log level
    """
    log_func = getattr(logger, level.lower())
    json_str = json.dumps(data, indent=2)
    log_func(f"{message}:\n{json_str}")


# Set up a default root logger
root_logger = setup_logger("prospectis", LOG_DIR / "prospectis.log")