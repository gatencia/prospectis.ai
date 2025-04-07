#!/usr/bin/env python3
"""Script to fetch high-value business problems from Reddit."""

import os
import sys
import logging
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Business_Problem_Data_Pipeline.db.connection import mongo
from Business_Problem_Data_Pipeline.apis.reddit import RedditClient
from Business_Problem_Data_Pipeline.processors.text_processor import TextProcessor
from Business_Problem_Data_Pipeline.processors.keyword_extractor import KeywordExtractor
from Business_Problem_Data_Pipeline.utils.logging_config import setup_logging
from Business_Problem_Data_Pipeline.config import REDDIT_SUBREDDITS

# 🔍 Add your domain-specific keywords here
KEYWORDS = [
    "algorithm", "detection", "classification", "automation", "security", "data", "analysis",
    "machine learning", "model", "training", "inference", "ai", "bias", "recommendation system",
    "forecasting", "nlp", "vision", "language model", "optimization", "prediction", "scalability"
]

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Fetch high-value problems from Reddit')
    parser.add_argument('--subreddits', nargs='+', default=None,
                        help='Subreddits to fetch from (default: use config)')
    parser.add_argument('--time-filter', default='day', choices=['hour', 'day', 'week', 'month', 'year', 'all'],
                        help='Time filter for Reddit posts')
    parser.add_argument('--limit', type=int, default=100,
                        help='Maximum number of posts to fetch per subreddit')
    parser.add_argument('--process', action='store_true',
                        help='Process the fetched problems')
    return parser.parse_args()

def contains_keywords(text):
    """Check if the text contains any relevant keywords."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in KEYWORDS)

def main():
    """Main function to fetch and store problems from Reddit."""
    logger = logging.getLogger('reddit')
    logger.info("Starting Reddit fetcher")

    args = parse_args()

    if not mongo.connect():
        logger.error("Failed to connect to MongoDB")
        return

    reddit_client = RedditClient()
    text_processor = TextProcessor() if args.process else None
    keyword_extractor = KeywordExtractor() if args.process else None

    logger.info(f"Fetching problems from Reddit with time filter '{args.time_filter}'")
    raw_problems = reddit_client.fetch_problems(
        subreddits=args.subreddits,
        time_filter=args.time_filter,
        limit=args.limit
    )

    if not raw_problems:
        logger.warning("No problems fetched from Reddit")
        mongo.close()
        return

    # Filter based on keyword content
    problems = [p for p in raw_problems if contains_keywords(p.text)]
    logger.info(f"Filtered down to {len(problems)} relevant problems")

    if not problems:
        logger.warning("No relevant problems matched keyword filters")
        mongo.close()
        return

    if args.process:
        logger.info("Processing problems")
        for problem in problems:
            problem = text_processor.process(problem)
            problem = keyword_extractor.extract_keywords(problem)

    logger.info(f"Saving {len(problems)} problems to MongoDB")
    collection = mongo.get_collection()

    for problem in problems:
        problem_dict = problem.to_dict()
        existing = collection.find_one({"source": problem.source, "source_id": problem.source_id})

        if existing:
            collection.update_one(
                {"source": problem.source, "source_id": problem.source_id},
                {"$set": problem_dict}
            )
        else:
            collection.insert_one(problem_dict)

    logger.info("Reddit fetcher completed")
    mongo.close()

if __name__ == '__main__':
    logger = setup_logging('reddit')
    try:
        main()
    except Exception as e:
        logger.exception(f"Reddit fetcher failed: {e}")
        sys.exit(1)