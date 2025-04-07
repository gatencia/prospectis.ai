"""Configuration settings for the Business Problem Data Pipeline."""

import os
from dotenv import load_dotenv

load_dotenv()

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "prospectis")
COLLECTION_NAME = os.getenv("PROBLEM_COLLECTION", "business_problems")

# API Keys and Credentials
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "python:prospectis:v0.1")

TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")

STACK_EXCHANGE_KEY = os.getenv("STACK_EXCHANGE_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_API_KEY = os.getenv("ADZUNA_API_KEY")

PATENTS_VIEW_API_KEY = os.getenv("PATENTS_VIEW_API_KEY")

# Source Configuration
REDDIT_SUBREDDITS = [
    "techsupport", "programming", "webdev", "datascience",
    "machinelearning", "cscareerquestions", "sysadmin", "devops",
    "startups", "cybersecurity"
]

STACK_EXCHANGE_SITES = [
    "stackoverflow", "serverfault", "dba", "datascience",
    "ai", "stats", "security", "superuser"
]

NEWS_KEYWORDS = [
    "technology problem", "software issue", "business challenge",
    "tech startup", "ai challenge", "data breach", "security vulnerability",
    "supply chain issue", "digital transformation"
]

# Data settings
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Ensure data directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# API Rate Limits
REDDIT_RATE_LIMIT = 60  # requests per minute
TWITTER_RATE_LIMIT = 15  # requests per 15 minutes
STACK_EXCHANGE_RATE_LIMIT = 30  # requests per second
NEWS_API_RATE_LIMIT = 10  # requests per minute
ADZUNA_RATE_LIMIT = 10  # requests per minute
PATENTS_VIEW_RATE_LIMIT = 45  # requests per minute
HACKER_NEWS_RATE_LIMIT = 10  # requests per minute

# Logging Configuration
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "business_problems.log")