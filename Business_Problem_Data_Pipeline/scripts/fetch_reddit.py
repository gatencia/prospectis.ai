#!/usr/bin/env python3
"""Script to fetch problems from Reddit."""

import os
import sys
import logging
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Business_Problem_Data_Pipeline.db.connection import mongo
from Business_Problem_Data_Pipeline.apis.reddit import RedditClient
from Business_Problem_Data_Pipeline.processors.text_processor import TextProcessor
from Business_Problem_Data_Pipeline.processors.keyword_extractor import KeywordExtractor
from Business_Problem_Data_Pipeline.utils.logging_config import setup_logging
from Business_Problem_Data_Pipeline.config import REDDIT_SUBREDDITS

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Fetch problems from Reddit')
    parser.add_argument('--subreddits', nargs='+', default=None,
                        help='Subreddits to fetch from (default: use config)')
    parser.add_argument('--time-filter', default='day', choices=['hour', 'day', 'week', 'month', 'year', 'all'],
                        help='Time filter for Reddit posts')
    parser.add_argument('--limit', type=int, default=100,
                        help='Maximum number of posts to fetch per subreddit')
    parser.add_argument('--process', action='store_true',
                        help='Process the fetched problems')
    return parser.parse_args()

def main():
    """Main function to fetch problems from Reddit."""
    logger = logging.getLogger('reddit')
    logger.info("Starting Reddit fetcher")
    
    # Parse arguments
    args = parse_args()
    
    # Connect to MongoDB
    if not mongo.connect():
        logger.error("Failed to connect to MongoDB")
        return
    
    # Initialize clients and processors
    reddit_client = RedditClient()
    text_processor = TextProcessor() if args.process else None
    keyword_extractor = KeywordExtractor() if args.process else None
    
    # Fetch problems
    logger.info(f"Fetching problems from Reddit with time filter '{args.time_filter}'")
    problems = reddit_client.fetch_problems(
        subreddits=args.subreddits,
        time_filter=args.time_filter,
        limit=args.limit
    )
    
    if not problems:
        logger.warning("No problems fetched from Reddit")
        mongo.close()
        return
    
    logger.info(f"Fetched {len(problems)} problems from Reddit")
    
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
    logger.info("Reddit fetcher completed")

if __name__ == '__main__':
    # Set up logging
    logger = setup_logging('reddit')
    
    try:
        main()
    except Exception as e:
        logger.exception(f"Reddit fetcher failed: {e}")
        sys.exit(1)