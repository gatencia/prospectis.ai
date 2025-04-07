"""
Configuration and environment variables for the research pipeline.
"""

import os
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Database settings
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "prospectis")
MONGODB_COLLECTION_PAPERS = os.getenv("MONGODB_COLLECTION_PAPERS", "research_papers")

# API keys
IEEE_API_KEY = os.getenv("IEEE_API_KEY")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

# Pipeline settings
PAPERS_FETCH_LIMIT = int(os.getenv("PAPERS_FETCH_LIMIT", "100"))
PAPERS_DAYS_LOOKBACK = int(os.getenv("PAPERS_DAYS_LOOKBACK", "3"))
SCHEDULER_INTERVAL_HOURS = int(os.getenv("SCHEDULER_INTERVAL_HOURS", "24"))

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", str(LOGS_DIR / "prospectis.log"))

# Convert string log level to logging constant
LOG_LEVEL_NUM = getattr(logging, LOG_LEVEL, logging.INFO)

# Sources to use in the pipeline
SOURCES = ["arxiv", "crossref", "ieee", "semantic_scholar"]