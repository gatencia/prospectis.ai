#!/usr/bin/env python3
"""Script to fetch problems from NewsAPI."""

import os
import sys
import logging
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from db.connection import mongo
from apis.news_api import NewsAPIClient
from processors.text_processor import TextProcessor
from processors.keyword_extractor import KeywordExtractor
from utils.logging_config import setup_logging
from config import NEWS_KEYWORDS

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Fetch problems from NewsAPI')
    parser.add_argument('--keywords', nargs='+', default=None,
                        help='Keywords to search for (default: use config)')
    parser.add_argument('--days', type=int, default=7,
                        help='Number of days to look back')
    parser.add_argument('--limit', type=int, default=100,
                        help='Maximum number of articles to fetch per keyword')
    parser.add_argument('--process', action='store_true',
                        help='Process the fetched problems')
    return parser.parse_args()

def main():
    """Main function to fetch problems from NewsAPI."""
    logger = logging.getLogger('news_api')
    logger.info("Starting NewsAPI fetcher")
    
    # Parse arguments
    args = parse_args()
    
    # Connect to MongoDB
    if not mongo.connect():
        logger.error("Failed to connect to MongoDB")
        return
    
    # Initialize clients and processors
    news_client = NewsAPIClient()
    text_processor = TextProcessor() if args.process else None
    keyword_extractor = KeywordExtractor() if args.process else None
    
    # Fetch problems
    logger.info(f"Fetching problems from NewsAPI from the last {args.days} days")
    problems = news_client.fetch_problems(
        keywords=args.keywords,
        days=args.days,
        limit=args.limit
    )
    
    if not problems:
        logger.warning("No problems fetched from NewsAPI")
        mongo.close()
        return
    
    logger.info(f"Fetched {len(problems)} problems from NewsAPI")
    
    # Process problems if requested
    if args.process:
        logger.info("Processing problems")
        for problem in problems:
            problem = text_processor.process(problem)
            problem = keyword_extractor.extract_keywords(problem)
    
    # Save problems to MongoDB
    logger.info(f"Saving {len(problems)} problems to MongoDB")
    collection = mongo.get_collection()
    
    for problem in problems:
        # Convert to dictionary
        problem_dict = problem.to_dict()
        
        # Check if problem already exists
        existing = collection.find_one({"source": problem.source, "source_id": problem.source_id})
        
        if existing:
            # Update existing problem
            collection.update_one(
                {"source": problem.source, "source_id": problem.source_id},
                {"$set": problem_dict}
            )
        else:
            # Insert new problem
            collection.insert_one(problem_dict)
    
    logger.info(f"Saved {len(problems)} problems to MongoDB")
    
    # Close connections
    mongo.close()
    logger.info("NewsAPI fetcher completed")

if __name__ == '__main__':
    # Set up logging
    logger = setup_logging('news_api')
    
    try:
        main()
    except Exception as e:
        logger.exception(f"NewsAPI fetcher failed: {e}")
        sys.exit(1)